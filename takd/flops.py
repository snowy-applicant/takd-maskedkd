"""Analytic training FLOPs (primary metric) with a FlopCounterMode cross-check.

Convention (the usual "model FLOPs"): one multiply-add = 2 FLOPs; only matrix
products are counted (linear layers, the patch-embedding convolution and the two
attention products). LayerNorm, GELU, softmax, the top-k selection and the
optimizer update are ignored. A backward pass costs twice its forward pass,
except the patch embedding whose input needs no gradient (1x). Recomputation
inside fused attention kernels is hardware work, not model work, and is not
counted here; the torch FlopCounterMode measurement recorded next to it does
count it, which is why it comes out slightly higher on CUDA.
"""
from .data import NUM_PATCHES, PATCH
from .models import ARCHS, MLP_RATIO


def patch_embed_flops(dim):
    return 2 * NUM_PATCHES * dim * (3 * PATCH * PATCH)


def block_flops(dim, tokens):
    hidden = dim * MLP_RATIO
    linear = 2 * tokens * dim * (3 * dim) + 2 * tokens * dim * dim + 2 * 2 * tokens * dim * hidden
    attention = 2 * 2 * tokens * tokens * dim  # q k^T and attn v, summed over heads
    return linear + attention


def forward_flops(arch, num_classes, visible_patches=NUM_PATCHES, cls_attention=False):
    """One sample; the patch embedding always runs on all 196 patches before any are dropped."""
    cfg = ARCHS[arch]
    dim, tokens = cfg["dim"], visible_patches + 1
    total = patch_embed_flops(dim) + cfg["depth"] * block_flops(dim, tokens) + 2 * dim * num_classes
    if cls_attention:
        total += 2 * tokens * dim  # q_cls k^T of the last block, used only to pick the mask
    return total


def train_flops(arch, num_classes, cls_attention=False):
    """Forward + backward of a trainee that sees all patches."""
    forward = forward_flops(arch, num_classes, cls_attention=cls_attention)
    extra = 2 * (NUM_PATCHES + 1) * ARCHS[arch]["dim"] if cls_attention else 0  # not back-propagated
    return 3 * (forward - extra) - patch_embed_flops(ARCHS[arch]["dim"]) + extra


def step_flops_per_sample(trainee_arch, teacher_arch, num_classes, keep_patches):
    """FLOPs of one optimisation step per sample, split by where they are spent.

    teacher_arch=None -> plain supervised fine-tuning (the teacher itself).
    keep_patches < 196 -> MaskedKD: the teacher forward sees only the kept patches
    and the trainee's last block also computes its CLS attention row.
    """
    masked = teacher_arch is not None and keep_patches < NUM_PATCHES
    trainee = train_flops(trainee_arch, num_classes, cls_attention=masked)
    teacher = forward_flops(teacher_arch, num_classes, keep_patches) if teacher_arch else 0
    return {"trainee": trainee, "teacher": teacher, "total": trainee + teacher}


def measure_step_flops(trainee, teacher, images, keep, feature_pool, loss_fn):
    """Count FLOPs of one real step with torch's FlopCounterMode (per sample).

    Returns None when the counter is unavailable or fails, so it can never stop a run.
    """
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    try:
        counter = FlopCounterMode(display=False)
        with counter:
            loss = loss_fn(trainee, teacher, images, keep, feature_pool)
            loss.backward()
        trainee.zero_grad(set_to_none=True)
        return counter.get_total_flops() / images.shape[0]
    except Exception as error:  # noqa: BLE001 - a diagnostic must not kill training
        print(f"FlopCounterMode cross-check failed: {error!r}", flush=True)
        return None
