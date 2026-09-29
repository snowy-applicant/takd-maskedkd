import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from takd.flops import forward_flops, step_flops_per_sample, train_flops
from takd.models import build


def counted(fn):
    counter = FlopCounterMode(display=False)
    with counter, sdpa_kernel(SDPBackend.MATH):
        fn()
    return counter.get_total_flops()


@pytest.mark.parametrize("arch", ["debug_small", "debug_base", "deit_small"])
@pytest.mark.parametrize("keep", [196, 98])
def test_forward_flops_match_flop_counter(arch, keep):
    torch.manual_seed(0)
    model = build(arch, 10).eval()
    x = torch.randn(2, 3, 224, 224)
    indices = None if keep == 196 else torch.stack([torch.randperm(196)[:keep] for _ in range(2)])
    with torch.no_grad():
        measured = counted(lambda: model(x, keep=indices)) / 2
    assert measured == forward_flops(arch, 10, keep)


@pytest.mark.parametrize("arch", ["debug_small", "deit_small"])
@pytest.mark.parametrize("cls_attention", [False, True])
def test_train_flops_match_flop_counter(arch, cls_attention):
    torch.manual_seed(0)
    model = build(arch, 10).train()
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([1, 2])

    def step():
        out = model(x, need_cls_attention=cls_attention)
        if cls_attention:
            out.cls_attention.detach().topk(98, dim=1)
        F.cross_entropy(out.logits, y).backward()

    assert counted(step) / 2 == train_flops(arch, 10, cls_attention=cls_attention)


def test_masking_halves_the_teacher_forward():
    full = step_flops_per_sample("deit_small", "deit_base", 10, 196)
    masked = step_flops_per_sample("deit_small", "deit_base", 10, 98)
    ratio = masked["teacher"] / full["teacher"]
    assert 0.49 < ratio < 0.51  # patch embedding still runs on all 196 patches
    # DeiT-B forward ~17.6 GMACs = ~35.1 GFLOPs (MaskedKD Tab. 10 counts MACs)
    assert abs(full["teacher"] / 2e9 - 17.58) < 0.05
