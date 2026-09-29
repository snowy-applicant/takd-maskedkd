import torch

from takd.models import ARCHS, build, widen_state

OFFICIAL_KEYS = {"cls_token", "pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias", "norm.weight",
                 "norm.bias", "head.weight", "head.bias"} | {
    f"blocks.{i}.{name}" for i in range(12) for name in (
        "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
        "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")}


def test_parameter_names_match_official_deit_checkpoints():
    for arch in ("deit_small", "deit_base", "deit_small_w2"):
        assert set(build(arch, 10).state_dict()) == OFFICIAL_KEYS


def test_sizes():
    count = lambda arch: sum(p.numel() for p in build(arch, 1000).parameters())
    assert abs(count("deit_small") - 22.05e6) < 0.1e6
    assert abs(count("deit_base") - 86.57e6) < 0.1e6
    # DeiT-S with every layer twice as wide has DeiT-B's size (documented in the README).
    assert count("deit_small_w2") == count("deit_base")


def _widened_pair(source_arch, target_arch, noise):
    torch.manual_seed(0)
    small = build(source_arch, 10).double().eval()
    state = {k: v for k, v in small.state_dict().items() if not k.startswith("head.")}
    wide = build(target_arch, 10).double().eval()
    widened = widen_state(state, ARCHS[source_arch], ARCHS[target_arch], noise, torch.Generator().manual_seed(1))
    factor = ARCHS[target_arch]["dim"] // ARCHS[source_arch]["dim"]
    widened["head.weight"] = small.head.weight.detach().repeat(1, factor) / factor
    widened["head.bias"] = small.head.bias.detach().clone()
    wide.load_state_dict({k: v.double() for k, v in widened.items()})
    return small, wide


def test_widening_preserves_the_function_exactly_without_noise():
    for source, target in (("debug_small", "debug_small_w2"), ("deit_small", "deit_small_w2")):
        small, wide = _widened_pair(source, target, 0.0)
        x = torch.randn(2, 3, 224, 224, dtype=torch.float64)
        with torch.no_grad():
            a, b = small(x, feature_pool="cls"), wide(x, feature_pool="cls")
        assert torch.allclose(a.logits, b.logits, atol=1e-9)
        dim = ARCHS[source]["dim"]
        # the residual stream is duplicated: [x; x]
        assert torch.allclose(b.features[-1][:, :dim], a.features[-1], atol=1e-9)
        assert torch.allclose(b.features[-1][:, dim:], a.features[-1], atol=1e-9)


def test_widening_noise_breaks_symmetry_but_stays_close():
    small, wide = _widened_pair("debug_small", "debug_small_w2", 0.01)
    x = torch.randn(4, 3, 224, 224, dtype=torch.float64)
    with torch.no_grad():
        a, b = small(x).logits, wide(x).logits
    assert not torch.allclose(a, b)
    assert (a - b).abs().max() < 0.05 * a.abs().max() + 1e-3
    qkv = wide.blocks[0].attn.qkv.weight
    half = qkv.shape[1] // 2
    assert not torch.equal(qkv[:, :half], qkv[:, half:])


def test_cls_attention_matches_explicit_attention():
    torch.manual_seed(0)
    model = build("debug_small", 10).eval()
    x = torch.randn(3, 3, 224, 224)
    captured = {}
    last = model.blocks[-1]

    def hook(module, inputs, output):
        captured["x"] = inputs[0]

    handle = last.attn.register_forward_hook(hook)
    with torch.no_grad():
        out = model(x, need_cls_attention=True)
    handle.remove()
    attn = last.attn
    b, n, c = captured["x"].shape
    qkv = attn.qkv(captured["x"]).reshape(b, n, 3, attn.num_heads, c // attn.num_heads).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    reference = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(-1).mean(dim=1)[:, 0, 1:]  # MaskedKD upstream
    assert out.cls_attention.shape == (3, 196)
    assert torch.allclose(out.cls_attention, reference, atol=1e-5)


def test_token_dropping_keeps_positions_and_is_order_invariant():
    torch.manual_seed(0)
    model = build("debug_base", 10).eval()
    x = torch.randn(2, 3, 224, 224)
    all_patches = torch.arange(196).expand(2, -1)
    with torch.no_grad():
        full = model(x).logits
        same = model(x, keep=all_patches).logits
        keep = torch.stack([torch.randperm(196)[:98] for _ in range(2)])
        masked = model(x, keep=keep).logits
        shuffled = model(x, keep=keep[:, torch.randperm(98)]).logits
    assert torch.allclose(full, same, atol=1e-5)
    assert torch.allclose(masked, shuffled, atol=1e-5)
    assert not torch.allclose(full, masked, atol=1e-3)


def test_cls_attention_is_fp32_under_autocast():
    torch.manual_seed(0)
    model = build("debug_small", 10).eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, need_cls_attention=True)
    with torch.no_grad():
        reference = model(x, need_cls_attention=True).cls_attention
    assert out.cls_attention.dtype == torch.float32
    assert bool((out.cls_attention.sum(1) < 1).all())  # the CLS->CLS column is excluded, as upstream
    assert torch.allclose(out.cls_attention, reference, atol=5e-3)  # only the bf16 q/k inputs differ
