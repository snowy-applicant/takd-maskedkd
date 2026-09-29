import json
from pathlib import Path

import torch
from torch.nn import functional as F

from takd.losses import Objective, kernel_probabilities, pkt
from takd.models import Output

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))


def test_probabilities_exclude_diagonal_and_normalise():
    f = torch.randn(8, 5)
    for kernel in ("cosine", "student_t"):
        p = kernel_probabilities(f, kernel)
        assert torch.allclose(p.diagonal(), torch.zeros(8))
        assert torch.allclose(p.sum(1), torch.ones(8), atol=1e-6)


def test_pkt_properties():
    torch.manual_seed(0)
    a, b = torch.randn(16, 32), torch.randn(16, 64, requires_grad=True)
    assert pkt(a, a).abs() < 1e-6                              # identical similarity structure
    assert pkt(a, b) > 0
    assert torch.allclose(pkt(a, b.detach()), pkt(b.detach(), a), atol=1e-6)  # Jeffreys is symmetric
    pkt(a, b).backward()
    assert b.grad is not None and b.grad.abs().sum() > 0       # widths may differ, no projector needed


def _outputs(batch, classes, dims, seed):
    g = torch.Generator().manual_seed(seed)
    return Output(logits=torch.randn(batch, classes, generator=g),
                  embedding=torch.randn(batch, dims, generator=g),
                  features=[torch.randn(batch, dims, generator=g) for _ in range(12)])


def test_hkd_objective_stages_and_critical_period():
    cfg = config("hkd")
    labels = torch.arange(8) % 10
    student, teacher = _outputs(8, 10, 384, 0), _outputs(8, 10, 768, 1)
    stage = Objective(cfg, "student", 10)
    _, parts = stage(student, teacher, labels, epoch_index=0)
    assert {"ce", "pkt_block3", "pkt_block6", "pkt_block9", "pkt_final", "flow"} <= set(parts)
    assert stage.weights(0) == {3: 100.0, 6: 100.0, 9: 100.0}
    assert abs(stage.weights(1)[3] - 70.0) < 1e-9
    expected = parts["ce"] + 100.0 * (parts["pkt_block3"] + parts["pkt_block6"] + parts["pkt_block9"]) + parts["pkt_final"]
    total, _ = stage(student, teacher, labels, epoch_index=0)
    assert torch.allclose(total, expected)
    assistant = Objective(cfg, "assistant", 10)
    _, parts = assistant(student, teacher, labels, epoch_index=0)
    assert "pkt_final" in parts and not any(k.startswith("pkt_block") for k in parts)


def test_logit_objective_is_reference_maskedkd_loss():
    cfg = config("logit")
    labels = torch.arange(8) % 10
    student, teacher = _outputs(8, 10, 384, 0), _outputs(8, 10, 768, 1)
    total, parts = Objective(cfg, "student", 10)(student, teacher, labels, epoch_index=3)
    ce = F.cross_entropy(student.logits, labels, label_smoothing=0.1)
    kd = F.kl_div(F.log_softmax(student.logits, 1), F.softmax(teacher.logits, 1), reduction="batchmean")
    assert torch.allclose(total, 0.5 * ce + 0.5 * kd, atol=1e-6)
    assert "flow" not in parts


def test_stepper_computes_the_objective_outside_autocast(monkeypatch):
    """PKT kernels must be real fp32 matmuls; inside autocast they would be re-cast to bf16."""
    from takd import engine
    from takd.models import build

    seen = {}
    cfg = config("hkd")
    cfg["loss"]["stages"]["student"]["taps"] = [1]
    objective = Objective(cfg, "student", 10)
    original = objective.__call__

    def spy(student, teacher, labels, epoch_index):
        seen["autocast"] = torch.is_autocast_enabled("cpu")
        return original(student, teacher, labels, epoch_index)

    monkeypatch.setattr(engine, "autocast", lambda device, precision: torch.autocast("cpu", dtype=torch.bfloat16))
    objective_proxy = type("P", (), {"__call__": staticmethod(spy), "feature_pool": "cls"})()
    trainee, teacher = build("debug_small", 10), build("debug_base", 10).eval()
    stepper = engine.Stepper(trainee, teacher, objective_proxy, 98, "bf16", torch.device("cpu"))
    stepper(torch.randn(4, 3, 224, 224), torch.arange(4), 0)
    assert seen["autocast"] is False
