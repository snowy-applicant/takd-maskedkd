"""Accuracy and MaskedKD diagnostics (all values are fractions in [0, 1])."""
import numpy as np
import torch
from torch.nn import functional as F

from .data import NUM_PATCHES, eval_batches


def autocast(device, precision="bf16"):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype,
                          enabled=device.type == "cuda" and precision in ("bf16", "fp16"))


def accuracy_report(predictions, labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes)
    correct = torch.bincount(labels[predictions == labels], minlength=num_classes)
    if bool((counts == 0).any()):
        raise ValueError("Evaluation split is missing a class")
    per_class = correct.double() / counts.double()
    return {"n": int(counts.sum()), "correct": int(correct.sum()),
            "overall_accuracy": float(correct.sum() / counts.sum()),
            "macro_accuracy": float(per_class.mean()),
            "class_counts": counts.tolist(), "class_correct": correct.tolist(),
            "class_accuracy": per_class.tolist()}


@torch.inference_mode()
def predict(model, split, normalizer, batch_size, device, precision):
    model.eval()
    outputs = []
    for images, _, _, _ in eval_batches(split, batch_size, normalizer):
        with autocast(device, precision):
            outputs.append(model(images).logits.float())
    return torch.cat(outputs)


def evaluate(model, split, normalizer, batch_size, device, precision, num_classes):
    logits = predict(model, split, normalizer, batch_size, device, precision)
    return accuracy_report(logits.argmax(1), split.labels, num_classes)


def pack_selection(selected):
    """(N, 196) bool -> list of hex strings (row-major 14x14, 1 = the teacher sees the patch)."""
    packed = np.packbits(selected.cpu().numpy(), axis=1)
    return [row.tobytes().hex() for row in packed]


def selection_overlap(current, previous, keep):
    if not previous or len(current) != len(previous):
        return None
    shared = sum((int.from_bytes(bytes.fromhex(a), "big") & int.from_bytes(bytes.fromhex(b), "big")).bit_count()
                 for a, b in zip(current, previous))
    return shared / (len(current) * keep)


@torch.inference_mode()
def mask_probe(trainee, teacher, split, normalizer, batch_size, device, precision, keep, record=False):
    """How much does masking change the frozen teacher, judged on a fixed split?

    The trainee picks the patches (its last-block CLS attention, top-k) exactly as
    during training; the same teacher is then run on all patches and on the kept ones.
    """
    trainee.eval()
    teacher.eval()
    rows = {"label": [], "student": [], "teacher_full": [], "teacher_masked": []}
    kl_sum, fg_recall, fg_precision, fg_images = 0.0, 0.0, 0.0, 0
    selections = []
    for images, labels, foreground, _ in eval_batches(split, batch_size, normalizer):
        with autocast(device, precision):
            out = trainee(images, need_cls_attention=True)
            indices = out.cls_attention.float().topk(keep, dim=1).indices
            full = teacher(images).logits.float()
            masked = teacher(images, keep=indices).logits.float()
        kl_sum += F.kl_div(F.log_softmax(masked, 1), F.softmax(full, 1), reduction="none").sum(1).sum().item()
        selected = torch.zeros(len(images), NUM_PATCHES, dtype=torch.bool, device=images.device)
        selected.scatter_(1, indices, True)
        selections.append(selected)
        fg_selected = (foreground & selected).sum(1).float()
        fg_total = foreground.sum(1).float()
        valid = fg_total > 0
        fg_recall += (fg_selected[valid] / fg_total[valid]).sum().item()
        fg_precision += (fg_selected[valid] / keep).sum().item()
        fg_images += int(valid.sum())
        for key, value in (("label", labels), ("student", out.logits.float().argmax(1)),
                           ("teacher_full", full.argmax(1)), ("teacher_masked", masked.argmax(1))):
            rows[key].append(value)
    r = {key: torch.cat(value) for key, value in rows.items()}
    n = r["label"].numel()
    full_ok, masked_ok = r["teacher_full"].eq(r["label"]), r["teacher_masked"].eq(r["label"])
    result = {
        "n": n,
        "student_accuracy": r["student"].eq(r["label"]).float().mean().item(),
        "teacher_full_accuracy": full_ok.float().mean().item(),
        "teacher_masked_accuracy": masked_ok.float().mean().item(),
        "full_correct_masked_wrong": (full_ok & ~masked_ok).float().mean().item(),
        "full_wrong_masked_correct": (~full_ok & masked_ok).float().mean().item(),
        "full_masked_disagreement": r["teacher_full"].ne(r["teacher_masked"]).float().mean().item(),
        "full_student_disagreement": r["teacher_full"].ne(r["student"]).float().mean().item(),
        "masked_student_disagreement": r["teacher_masked"].ne(r["student"]).float().mean().item(),
        "kl_full_to_masked": kl_sum / n,
        "foreground_recall": fg_recall / fg_images if fg_images else None,
        "foreground_precision": fg_precision / fg_images if fg_images else None,
        "foreground_images": fg_images,
        "selection_hex": pack_selection(torch.cat(selections)),
    }
    if record:
        result["predictions"] = {key: value.tolist() for key, value in r.items()}
    return result
