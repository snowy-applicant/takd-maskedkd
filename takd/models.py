"""DeiT variants for the TAKD x MaskedKD experiment.

Parameter names follow the official DeiT checkpoints (and the reference runner's
torch-only DeiT), so ImageNet weights load without timm. Differences from the
reference module, all numerically equivalent up to kernel rounding:

* attention uses ``F.scaled_dot_product_attention`` (flash kernels on Ampere);
* only when a mask must be chosen does the last block also compute the CLS row
  of its attention map, ``softmax(q_cls k^T / sqrt(d_h))`` averaged over heads,
  which is exactly the ``attn.mean(dim=1)[:, 0, 1:]`` used by MaskedKD;
* every block can report its token representation for feature distillation.

Position embeddings are added before tokens are gathered and CLS is always
kept, as in MaskedKD. Dropped tokens are removed, not replaced by mask tokens.
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .data import IMAGE_SIZE, NUM_PATCHES, PATCH

# Teacher assistant "DeiT-S (width x2)": the student architecture with every
# layer twice as wide (embedding 384 -> 768, MLP 1536 -> 3072). The head width is
# kept at 64, so the head count doubles (6 -> 12), as in the DeiT size family.
ARCHS = {
    "deit_tiny": dict(dim=192, depth=12, heads=3),
    "deit_small": dict(dim=384, depth=12, heads=6),
    "deit_small_w2": dict(dim=768, depth=12, heads=12),
    "deit_base": dict(dim=768, depth=12, heads=12),
    # Tiny stand-ins for CPU smoke tests; same code paths, same width ratios.
    "debug_small": dict(dim=32, depth=2, heads=2),
    "debug_small_w2": dict(dim=64, depth=2, heads=4),
    "debug_base": dict(dim=64, depth=2, heads=4),
}
IMAGENET_URLS = {
    "deit_tiny": "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    "deit_small": "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
    "deit_base": "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
}
MLP_RATIO = 4


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} is not divisible by heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, need_cls_attention=False):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        cls_attention = None
        if need_cls_attention:
            # fp32 on purpose: under autocast this small matmul would otherwise run in bf16
            # (8-bit mantissa) and add rank noise to the top-k patch choice.
            with torch.autocast(device_type=q.device.type, enabled=False):
                scores = (q[:, :, :1].float() * self.scale) @ k.float().transpose(-2, -1)
                cls_attention = scores.softmax(dim=-1).mean(dim=1)[:, 0, 1:]
        return self.proj(out.transpose(1, 2).reshape(b, n, c)), cls_attention


class Block(nn.Module):
    def __init__(self, dim, heads, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * MLP_RATIO)

    def forward(self, x, need_cls_attention=False):
        y, cls_attention = self.attn(self.norm1(x), need_cls_attention)
        x = x + self.drop_path(y)
        return x + self.drop_path(self.mlp(self.norm2(x))), cls_attention


class PatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


@dataclass
class Output:
    logits: torch.Tensor
    embedding: torch.Tensor = None              # final normalised CLS token, the input of the head
    cls_attention: torch.Tensor = None          # (B, 196) patch attention of the CLS query, last block
    features: list = field(default_factory=list)  # per-block pooled representation, (B, dim) each


class DeiT(nn.Module):
    def __init__(self, dim, depth, heads, num_classes, drop_path=0.0):
        super().__init__()
        self.dim, self.depth, self.heads, self.num_classes = dim, depth, heads, num_classes
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_PATCHES + 1, dim))
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([Block(dim, heads, rate) for rate in rates])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x, keep=None, need_cls_attention=False, feature_pool=None):
        """keep: (B, k) patch indices the model may see (None = all 196 patches).

        feature_pool: None, "cls" (CLS token after each block) or "mean"
        (average of the visible patch tokens after each block).
        """
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("This experiment requires 224 x 224 input")
        tokens = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), tokens), dim=1) + self.pos_embed
        if keep is not None:
            visible = x[:, 1:].gather(1, keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            x = torch.cat((x[:, :1], visible), dim=1)
        features, cls_attention = [], None
        last = len(self.blocks) - 1
        for index, block in enumerate(self.blocks):
            x, attention = block(x, need_cls_attention and index == last)
            if attention is not None:
                cls_attention = attention
            if feature_pool == "cls":
                features.append(x[:, 0])
            elif feature_pool == "mean":
                features.append(x[:, 1:].mean(dim=1))
            elif feature_pool is not None:
                raise ValueError(feature_pool)
        embedding = self.norm(x)[:, 0]
        return Output(self.head(embedding), embedding, cls_attention, features)


def build(arch, num_classes, drop_path=0.0):
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture {arch}; choose from {sorted(ARCHS)}")
    return DeiT(**ARCHS[arch], num_classes=num_classes, drop_path=drop_path)


def imagenet_state(arch):
    """Official non-distilled DeiT ImageNet weights, SHA-prefix verified by torch.hub."""
    checkpoint = torch.hub.load_state_dict_from_url(IMAGENET_URLS[arch], map_location="cpu",
                                                    check_hash=True, weights_only=True)
    return {k: v for k, v in checkpoint["model"].items() if not k.startswith("head.")}


def imagenet_file(arch):
    return Path(torch.hub.get_dir()) / "checkpoints" / IMAGENET_URLS[arch].rsplit("/", 1)[-1]


def load_backbone(model, state):
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {"head.weight", "head.bias"} or result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint mismatch: {result}")


def widen_state(state, source_cfg, target_cfg, noise=0.0, generator=None):
    """Function-preserving widening (Net2WiderNet) of a DeiT backbone by an integer factor.

    Every unit of the residual stream, of q/k/v (hence every head) and of the MLP
    hidden layer is copied ``f`` times; a layer reading duplicated inputs divides
    its weights by ``f``. Duplicating a LayerNorm input keeps its mean and
    variance, so with ``noise=0`` the widened network computes exactly the same
    function. Gaussian noise (``noise`` x per-tensor std) on the weight matrices
    breaks the symmetry between copies, which gradient descent (and Adam in
    particular) would otherwise preserve forever.
    """
    d, big = source_cfg["dim"], target_cfg["dim"]
    factor = big // d
    if factor * d != big or target_cfg["depth"] != source_cfg["depth"]:
        raise ValueError("Widening needs the same depth and an integer width factor")
    if big // target_cfg["heads"] != d // source_cfg["heads"]:
        raise ValueError("Widening keeps the head width; the head count must scale with the width")

    def rows(t):  # duplicate output units
        return t.repeat(factor, *([1] * (t.ndim - 1)))

    def cols(t):  # duplicated inputs share the original weight
        return t.repeat(1, factor) / factor

    def perturb(t):
        if noise <= 0:
            return t
        return t + torch.randn(t.shape, generator=generator, dtype=t.dtype) * (noise * t.std())

    out = {}
    for key, value in state.items():
        value = value.float()
        if key in ("cls_token", "pos_embed"):
            out[key] = value.repeat(1, 1, factor)
        elif key == "patch_embed.proj.weight":
            out[key] = perturb(rows(value))
        elif key.endswith("attn.qkv.weight"):
            q, k, v = value.chunk(3, dim=0)
            out[key] = perturb(torch.cat([cols(rows(part)) for part in (q, k, v)], dim=0))
        elif key.endswith("attn.qkv.bias"):
            out[key] = torch.cat([rows(part) for part in value.chunk(3, dim=0)], dim=0)
        elif value.ndim == 2:  # attn.proj, mlp.fc1, mlp.fc2
            out[key] = perturb(cols(rows(value)))
        elif value.ndim == 1:  # biases and LayerNorm affine parameters
            out[key] = rows(value)
        else:
            raise KeyError(f"Cannot widen parameter {key} with shape {tuple(value.shape)}")
    return out


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def describe(arch):
    cfg = ARCHS[arch]
    head_dim = cfg["dim"] // cfg["heads"]
    return f"{arch}(dim={cfg['dim']}, depth={cfg['depth']}, heads={cfg['heads']}, head_dim={head_dim})"
