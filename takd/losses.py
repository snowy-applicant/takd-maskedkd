"""Distillation objectives.

One formula covers every stage of both suites:

    L = w_ce * CE(student, y; label smoothing)
      + w_logit * T^2 * KL(softmax(t / T) || softmax(s / T))
      + w_flow * sum_l alpha_l(epoch) * PKT_l(teacher, student)

``hkd`` suite (Passalis et al., CVPR 2020, "Heterogeneous Knowledge
Distillation using Information Flow Modeling"):

* PKT_l is the paper's Eq. 7: Jeffreys divergence between the batch conditional
  probabilities of a cosine kernel plus the same for a Student-t kernel (d = 1).
  Only B x B matrices are compared, so teacher and student widths may differ.
* Eq. 10 critical-period weights: the final representation always has weight 1;
  every intermediate tap has alpha_init * gamma^k at epoch k (k = 0 first),
  with the paper's alpha_init = 100 and gamma = 0.7.
* The assistant is trained from the teacher with plain PKT on the final
  representation only (paper App. A.2); the student learns all tapped layers one
  to one (Eq. 9). The paper adds a CE term for classification (Sec. 4, Tab. 2).
* Deviation from the released code, following the paper instead: the diagonal
  (i = j) is excluded; the divergence is summed over j and averaged over anchors
  i (a "batchmean" scale, so it is comparable to CE).

``logit`` suite: the MaskedKD / reference-runner objective,
0.5 * CE + 0.5 * KL with T = 1, applied in every distillation stage.
"""
import torch
from torch.nn import functional as F

EPS = 1e-8


def kernel_probabilities(features, kernel):
    """Row-normalised p_{j|i} over a batch, diagonal excluded (paper Eq. 5/6)."""
    f = features.float()
    if kernel == "cosine":
        f = f / (f.norm(dim=1, keepdim=True) + 1e-6)
        similarity = (f @ f.t() + 1.0) / 2.0
    elif kernel == "student_t":
        squared = (f * f).sum(dim=1)
        distance = (squared[:, None] + squared[None, :] - 2.0 * (f @ f.t())).clamp_min(0.0)
        similarity = 1.0 / (1.0 + (distance + 1e-6).sqrt())
    else:
        raise ValueError(kernel)
    eye = torch.eye(f.shape[0], dtype=torch.bool, device=f.device)
    similarity = similarity.masked_fill(eye, 0.0)
    return similarity / similarity.sum(dim=1, keepdim=True).clamp_min(EPS)


def jeffreys(p_teacher, p_student):
    """sum_{j != i} (p_t - p_s)(log p_t - log p_s), averaged over anchors i (paper Eq. 8)."""
    term = (p_teacher - p_student) * (torch.log(p_teacher + EPS) - torch.log(p_student + EPS))
    eye = torch.eye(term.shape[0], dtype=torch.bool, device=term.device)
    return term.masked_fill(eye, 0.0).sum(dim=1).mean()


def pkt(teacher_features, student_features):
    """Hybrid cosine + Student-t probability matching for one layer pair (paper Eq. 7)."""
    if teacher_features.shape[0] < 3:
        return student_features.new_zeros((), dtype=torch.float32)
    return sum(jeffreys(kernel_probabilities(teacher_features.detach(), k), kernel_probabilities(student_features, k))
               for k in ("cosine", "student_t"))


def critical_period_weights(taps, epoch_index, alpha_init, gamma):
    """alpha for each intermediate tap at epoch k = epoch_index (0-based)."""
    return {tap: alpha_init * gamma ** epoch_index for tap in taps}


class Objective:
    """Stage objective; ``stage`` is "assistant" (T -> A) or "student" (-> S)."""

    def __init__(self, cfg, stage, num_classes):
        loss = cfg["loss"]
        self.kind = loss["type"]
        self.num_classes = num_classes
        self.label_smoothing = loss["label_smoothing"]
        self.temperature = loss.get("temperature", 1.0)
        self.w_ce = loss["ce_weight"]
        self.w_logit = loss["logit_weight"]
        self.w_flow = loss["flow_weight"]
        stage_cfg = loss.get("stages", {}).get(stage, {"taps": [], "final": False})
        self.taps = list(stage_cfg["taps"])
        self.final = bool(stage_cfg["final"])
        self.alpha_init = loss.get("alpha_init", 100.0)
        self.gamma = loss.get("gamma", 0.7)
        self.feature_pool = loss.get("feature_pool", "cls") if self.uses_features else None

    @property
    def uses_features(self):
        return self.w_flow > 0 and (bool(self.taps) or self.final)

    @property
    def needs_intermediate(self):
        return self.w_flow > 0 and bool(self.taps)

    def weights(self, epoch_index):
        return critical_period_weights(self.taps, epoch_index, self.alpha_init, self.gamma)

    def supervised(self, logits, labels):
        return F.cross_entropy(logits.float(), labels, label_smoothing=self.label_smoothing)

    def __call__(self, student, teacher, labels, epoch_index):
        """Returns (total, parts) where parts holds detached per-term values."""
        ce = self.supervised(student.logits, labels)
        total = self.w_ce * ce
        parts = {"ce": ce.detach()}
        if teacher is None:
            return total, parts
        if self.w_logit > 0:
            t = self.temperature
            kd = F.kl_div(F.log_softmax(student.logits.float() / t, dim=1),
                          F.softmax(teacher.logits.float() / t, dim=1), reduction="batchmean") * t * t
            total = total + self.w_logit * kd
            parts["logit_kd"] = kd.detach()
        if self.uses_features:
            flow = student.logits.new_zeros((), dtype=torch.float32)
            for tap, alpha in self.weights(epoch_index).items():
                term = pkt(teacher.features[tap - 1], student.features[tap - 1])
                parts[f"pkt_block{tap}"] = term.detach()
                flow = flow + alpha * term
            if self.final:
                term = pkt(teacher.embedding, student.embedding)
                parts["pkt_final"] = term.detach()
                flow = flow + term
            total = total + self.w_flow * flow
            parts["flow"] = flow.detach()
        return total, parts
