"""Resumable training of one run: teacher fine-tuning, T -> A, or -> S."""
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import flops as flop_model
from .data import NUM_PATCHES, Normalizer, load_cache, sha256, synthetic_dataset, train_batches
from .losses import Objective
from .metrics import autocast, evaluate, mask_probe, selection_overlap
from .models import (ARCHS, IMAGENET_URLS, build, imagenet_file, imagenet_state, load_backbone,
                     parameter_count, widen_state)
from .plan import Run

ROOT = Path(__file__).resolve().parents[1]
TRAINING_CODE = ("data.py", "models.py", "losses.py", "flops.py", "metrics.py", "engine.py", "plan.py")


# --------------------------------------------------------------------------- provenance

def code_hash():
    digest = hashlib.sha256()
    for name in TRAINING_CODE:
        path = ROOT / "takd" / name
        digest.update(name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def source_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "takd", "configs"], cwd=ROOT,
                               capture_output=True, text=True)
    except OSError:
        return "unknown"
    if head.returncode:
        return "unknown"
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def cpu_state(model):
    return {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- run layout

def run_directory(output_root, suite, run_name):
    # The fine-tuned teacher is shared by every suite: it is trained with CE only.
    return Path(output_root) / ("teacher" if run_name == "teacher" else Path(suite) / run_name)


def keep_patches(cfg):
    return int(round(NUM_PATCHES * (1.0 - cfg["mask_ratio"])))


def role_settings(cfg, role):
    """Everything a run of this role depends on (goes into its signature)."""
    shared = {key: cfg[key] for key in ("batch_size", "warmup_epochs", "learning_rate", "min_learning_rate",
                                        "weight_decay", "drop_path", "gradient_clip", "precision")}
    settings = {"arch": cfg["archs"][role], "init": cfg["init"][role], "epochs": cfg["epochs"][role],
                "label_smoothing": cfg["loss"]["label_smoothing"], **shared}
    if settings["init"] == "widen":
        settings.update(widen_from=cfg["widen_from"], widen_noise=cfg["widen_noise"])
    if role != "teacher":
        settings.update(loss=cfg["loss"], keep_patches=keep_patches(cfg), probe_epochs=cfg["probe_epochs"],
                        eval_batch_size=cfg["eval_batch_size"])
    return settings


def run_signature(run, cfg, teacher_sha, data_sha, debug):
    """Identity of a run: same signature <=> same inputs, config, frozen teacher and training code."""
    keep = keep_patches(cfg) if run.masked else NUM_PATCHES
    return hashlib.sha256(json.dumps(
        {"run": run.name, "role": run.role, "seed": run.seed, "masked": run.masked, "keep": keep,
         "teacher": run.teacher, "teacher_sha256": teacher_sha, "settings": role_settings(cfg, run.role),
         "data": data_sha, "code": code_hash(), "debug": debug},
        sort_keys=True).encode()).hexdigest()


def learning_rate(cfg, epoch, total):
    """Reference-runner schedule: linear warm-up per epoch, then cosine to the floor."""
    base = cfg["learning_rate"]
    warmup = min(cfg["warmup_epochs"], max(total - 1, 0))
    if epoch <= warmup:
        return base * epoch / max(1, warmup)
    fraction = (epoch - warmup - 1) / max(1, total - warmup - 1)
    floor = min(base, cfg["min_learning_rate"])
    return floor + (base - floor) * (1 + math.cos(math.pi * fraction)) / 2


def optimizer_for(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        skip = parameter.ndim == 1 or name.endswith(".bias") or name in ("pos_embed", "cls_token")
        (no_decay if skip else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}], lr=cfg["learning_rate"])


# --------------------------------------------------------------------------- models

def initial_backbone(cfg, role, seed, debug):
    """(state or None, provenance string) for the trainee's backbone before training."""
    arch, init = cfg["archs"][role], cfg["init"][role]
    if init == "scratch":
        return None, "scratch"
    if init == "imagenet":
        if arch not in IMAGENET_URLS:
            raise ValueError(f"No ImageNet weights for {arch}; use init 'scratch' or 'widen'")
        state = imagenet_state(arch)
        return state, f"imagenet:{sha256(imagenet_file(arch))}"
    if init == "widen":
        source = cfg["widen_from"]
        if source in IMAGENET_URLS and not debug:
            state, origin = imagenet_state(source), f"imagenet:{sha256(imagenet_file(source))}"
        else:  # debug stand-ins have no pretrained weights; widen a seeded random init instead
            torch.manual_seed(seed + 1)
            state = {k: v for k, v in build(source, 1).state_dict().items() if not k.startswith("head.")}
            origin = f"random:{source}:seed{seed + 1}"
        generator = torch.Generator().manual_seed(seed * 7919 + 17)
        widened = widen_state(state, ARCHS[source], ARCHS[arch], cfg["widen_noise"], generator)
        return widened, f"widen({origin}, x{ARCHS[arch]['dim'] // ARCHS[source]['dim']}, noise={cfg['widen_noise']})"
    raise ValueError(f"Unknown init {init}")


def load_frozen(path, device):
    checkpoint = load_checkpoint(path)
    model = build(checkpoint["arch"], checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.requires_grad_(False).eval(), checkpoint


# --------------------------------------------------------------------------- one step

class Stepper:
    """Forward/backward of one micro-batch, shared by training, VRAM probing and FLOP counting."""

    def __init__(self, trainee, teacher, objective, keep, precision, device):
        self.trainee, self.teacher, self.objective = trainee, teacher, objective
        self.keep, self.precision, self.device = keep, precision, device
        self.masked = teacher is not None and keep < NUM_PATCHES
        self.timed = device.type == "cuda" and teacher is not None
        self.events = []

    def __call__(self, images, labels, epoch_index, weight=1.0):
        pool = self.objective.feature_pool if self.teacher is not None else None
        with autocast(self.device, self.precision):
            out = self.trainee(images, need_cls_attention=self.masked, feature_pool=pool)
            teacher_out = None
            if self.teacher is not None:
                keep = out.cls_attention.detach().topk(self.keep, dim=1).indices if self.masked else None
                if self.timed:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                with torch.no_grad():
                    teacher_out = self.teacher(images, keep=keep, feature_pool=pool)
                if self.timed:
                    end.record()
                    self.events.append((start, end))
        # Outside autocast: inside it every matmul (the PKT kernels included) is re-cast to bf16
        # whatever the input dtype, which would leave ~3 significant digits in the B x B similarities.
        loss, parts = self.objective(out, teacher_out, labels, epoch_index)
        (loss * weight).backward()
        return loss.detach(), parts, out.logits.detach()

    def teacher_seconds(self):
        """Sum of the recorded teacher-forward intervals; call after a device synchronize."""
        total = sum(start.elapsed_time(end) for start, end in self.events) / 1000.0
        self.events.clear()
        return total


def probe_micro_batch(stepper, batch_size, device, num_classes, allow_split=True):
    """Largest micro-batch (batch_size / 2^k) whose step fits in free VRAM.

    CE and logit KD are per-sample means, so accumulating micro-batches reproduces the
    full-batch objective exactly. PKT is estimated from within-batch similarities, so for
    feature distillation a smaller micro-batch would silently change the loss: refuse instead.
    """
    micro = batch_size
    while True:
        try:
            images = torch.randn(micro, 3, 224, 224, device=device)
            labels = torch.arange(micro, device=device) % num_classes
            stepper(images, labels, 0)
            # Also allocate AdamW's moment buffers (a throwaway optimizer with lr=0 leaves the
            # weights untouched) so the probe's peak includes everything training will hold.
            scratch = torch.optim.AdamW(stepper.trainee.parameters(), lr=0.0, weight_decay=0.0)
            scratch.step()
            del scratch
            stepper.trainee.zero_grad(set_to_none=True)
            stepper.events.clear()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return micro
        except torch.OutOfMemoryError:
            stepper.trainee.zero_grad(set_to_none=True)
            stepper.events.clear()
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            if not allow_split:
                free = torch.cuda.mem_get_info(device)[0] / 1024**3 if device.type == "cuda" else 0.0
                raise RuntimeError(
                    f"A batch of {batch_size} does not fit in free VRAM ({free:.1f} GiB free now). The PKT "
                    f"(information-flow) loss needs whole batches, so micro-batching would change the objective. "
                    f"Free the GPU (nvidia-smi: other processes?) and rerun the same command to resume, or set "
                    f"\"allow_pkt_micro_batch\": true in the config to accept a different objective.") from None
            micro //= 2
            print(f"OOM: retrying with micro-batch {micro} (gradient accumulation keeps batch {batch_size})",
                  flush=True)


def measured_flops(stepper, device, num_classes, samples=4):
    """torch FlopCounterMode count of one real step, per sample (cross-check only)."""
    try:
        from contextlib import nullcontext

        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.utils.flop_counter import FlopCounterMode
        images = torch.randn(samples, 3, 224, 224, device=device)
        labels = torch.arange(samples, device=device) % num_classes
        counter = FlopCounterMode(display=False)
        # CPU's fused SDPA kernel has no FLOP formula; the math backend decomposes into counted bmm.
        backend = sdpa_kernel(SDPBackend.MATH) if device.type == "cpu" else nullcontext()
        with counter, backend:
            stepper(images, labels, 0)
        stepper.trainee.zero_grad(set_to_none=True)
        stepper.events.clear()
        return counter.get_total_flops() / samples
    except Exception as error:  # noqa: BLE001 - diagnostics must never stop a run
        print(f"FlopCounterMode cross-check unavailable: {error!r}", flush=True)
        stepper.trainee.zero_grad(set_to_none=True)
        stepper.events.clear()
        return None


# --------------------------------------------------------------------------- training

def train_epoch(stepper, optimizer, data, normalizer, cfg, seed, epoch, micro, device):
    stepper.trainee.train()
    seed_all(seed * 100003 + epoch)  # drop-path masks; batch order has its own generator
    sums = defaultdict(lambda: torch.zeros((), device=device))
    samples = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for images, labels in train_batches(data.train, cfg["batch_size"], normalizer, seed, epoch):
        if len(labels) < cfg["batch_size"]:
            continue  # drop the ragged tail: the batch-level PKT estimate needs full batches
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(labels), micro):
            x, y = images[start:start + micro], labels[start:start + micro]
            loss, parts, logits = stepper(x, y, epoch - 1, weight=len(y) / len(labels))
            sums["loss"] += loss * len(y)
            for key, value in parts.items():
                sums[key] += value * len(y)
            sums["correct"] += (logits.argmax(1) == y).sum()
        torch.nn.utils.clip_grad_norm_(stepper.trainee.parameters(), cfg["gradient_clip"])
        optimizer.step()
        samples += len(labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    teacher_seconds = stepper.teacher_seconds() if stepper.timed else None
    values = {key: value.item() for key, value in sums.items()}
    if not math.isfinite(values["loss"]):
        raise FloatingPointError(f"Non-finite training loss in epoch {epoch}")
    result = {key: value / samples for key, value in values.items() if key != "correct"}
    result.update(accuracy=values["correct"] / samples, samples=samples, seconds=seconds,
                  teacher_seconds=teacher_seconds, samples_per_second=samples / seconds)
    return result


def train_run(run: Run, cfg, suite, output_root, device_name="cuda", debug=False, data_cache=None,
              allow_code_change=False):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the CUDA build (bash scripts/setup.sh) "
                           "or use --device cpu for smoke tests")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    directory = run_directory(output_root, suite, run.name)
    directory.mkdir(parents=True, exist_ok=True)

    data = synthetic_dataset(device) if debug else load_cache(data_cache, device)
    num_classes = len(data.classes)
    role = run.role
    teacher, teacher_ckpt, teacher_sha = None, None, ""
    if run.teacher:
        teacher_file = run_directory(output_root, suite, run.teacher) / "best.pt"
        if not teacher_file.is_file():
            raise FileNotFoundError(f"Frozen teacher missing: {teacher_file} (run {run.teacher} first)")
        teacher_sha = sha256(teacher_file)
        teacher, teacher_ckpt = load_frozen(teacher_file, device)
    settings = role_settings(cfg, role)
    keep = keep_patches(cfg) if run.masked else NUM_PATCHES
    signature = run_signature(run, cfg, teacher_sha, data.manifest_sha256, debug)

    result_path = directory / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing["signature"] == signature or allow_code_change:
            print(f"REUSE {run.name}", flush=True)
            return existing
        raise ValueError(f"Completed run {directory} was produced by different code/config/teacher. "
                         f"Move it aside to retrain, or pass --allow-code-change to keep it.")

    resume_path = directory / "resume.pt"
    resume = load_checkpoint(resume_path) if resume_path.is_file() else None
    if resume and resume["signature"] != signature:
        raise ValueError(f"Resume checkpoint in {directory} has a different signature; move it aside")

    seed_all(run.seed)
    trainee = build(settings["arch"], num_classes, drop_path=cfg["drop_path"])
    state, init_provenance = initial_backbone(cfg, role, run.seed, debug)
    if state is not None and resume is None:
        load_backbone(trainee, state)
    trainee = trainee.to(device)
    objective = Objective(cfg, role if role != "teacher" else "assistant", num_classes)
    if role == "teacher":  # CE only, full weight, whatever the suite's distillation loss is
        objective.w_ce, objective.w_logit, objective.w_flow = 1.0, 0.0, 0.0
        objective.feature_pool = None
    if teacher is not None and objective.needs_intermediate:
        depth = min(trainee.depth, teacher.depth)
        if max(objective.taps) > depth or min(objective.taps) < 1:
            raise ValueError(f"Feature taps {objective.taps} must lie in blocks 1..{depth}")
    stepper = Stepper(trainee, teacher, objective, keep, cfg["precision"], device)
    optimizer = optimizer_for(trainee, cfg)
    normalizer = Normalizer(device)

    analytic = flop_model.step_flops_per_sample(settings["arch"], teacher_ckpt["arch"] if teacher_ckpt else None,
                                                num_classes, keep)
    measured = measured_flops(stepper, device, num_classes)
    allow_split = not (teacher is not None and objective.uses_features) or cfg.get("allow_pkt_micro_batch", False)
    micro = probe_micro_batch(stepper, cfg["batch_size"], device, num_classes, allow_split)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history, probes, best_epoch, best_score, start_epoch, best_weights = [], [], 0, -1.0, 0, None
    if resume:
        trainee.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        history, probes = resume["history"], resume["probes"]
        best_epoch, best_score, start_epoch = resume["best_epoch"], resume["best_score"], resume["epoch"]
        best_weights = resume["best_model"]
        # best.pt/history.json may already hold epochs after the last durable resume
        # checkpoint; those epochs are retrained, so restore the matching best.pt.
        save_checkpoint(directory / "best.pt", {"model": best_weights, "arch": settings["arch"],
                                                "num_classes": num_classes, "epoch": best_epoch,
                                                "signature": signature})
        write_json(directory / "history.json", history)
        print(f"RESUME {run.name} after epoch {start_epoch}", flush=True)

    other = os.environ.get("TAKD_GPU_OTHER_PROCESSES", "")
    other_processes = int(other) if other.isdigit() else None  # seen by the runner when this session began
    total_epochs = settings["epochs"]
    probe_epochs = {e for e in cfg["probe_epochs"] if e <= total_epochs} if run.masked else set()
    eval_batch = cfg["eval_batch_size"]
    precision = cfg["precision"]
    if 0 in probe_epochs and not probes:
        started = time.perf_counter()
        probe = mask_probe(trainee, teacher, data.val, normalizer, eval_batch, device, precision, keep)
        probes.append({"epoch": 0, **probe, "selection_overlap_initial": 1.0,
                       "selection_overlap_previous": 1.0, "seconds": time.perf_counter() - started})

    header = (f"{run.name}: trainee={settings['arch']} ({parameter_count(trainee) / 1e6:.1f}M, init={init_provenance})"
              f" teacher={teacher_ckpt['arch'] if teacher_ckpt else '-'} keep={keep}/{NUM_PATCHES}"
              f" micro_batch={micro}/{cfg['batch_size']} epochs={total_epochs}"
              f" analytic={analytic['total'] / 1e9:.2f} GFLOP/sample"
              f" measured={'n/a' if measured is None else f'{measured / 1e9:.2f}'} GFLOP/sample")
    print(header, flush=True)

    for epoch in range(start_epoch + 1, total_epochs + 1):
        begun = time.perf_counter()
        lr = learning_rate(cfg, epoch, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        training = train_epoch(stepper, optimizer, data, normalizer, cfg, run.seed, epoch, micro, device)
        eval_started = time.perf_counter()
        validation = evaluate(trainee, data.val, normalizer, eval_batch, device, precision, num_classes)
        eval_seconds = time.perf_counter() - eval_started
        score = validation["macro_accuracy"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_weights = cpu_state(trainee)
            save_checkpoint(directory / "best.pt", {"model": best_weights, "arch": settings["arch"],
                                                    "num_classes": num_classes, "epoch": epoch,
                                                    "signature": signature})
        probe_seconds = 0.0
        if epoch in probe_epochs:
            started = time.perf_counter()
            probe = mask_probe(trainee, teacher, data.val, normalizer, eval_batch, device, precision, keep)
            probe_seconds = time.perf_counter() - started
            probes.append({"epoch": epoch, **probe,
                           "selection_overlap_initial": selection_overlap(probe["selection_hex"],
                                                                          probes[0]["selection_hex"], keep)
                           if probes else None,
                           "selection_overlap_previous": selection_overlap(probe["selection_hex"],
                                                                           probes[-1]["selection_hex"], keep)
                           if probes else None,
                           "seconds": probe_seconds})
        cumulative = (history[-1]["cumulative_train_flops"] if history else 0.0) + training["samples"] * analytic["total"]
        row = {"epoch": epoch, "lr": lr, "train": training, "validation": validation,
               "micro_batch": micro, "gpu_other_processes": other_processes,
               "train_flops": training["samples"] * analytic["total"], "cumulative_train_flops": cumulative,
               "eval_seconds": eval_seconds, "probe_seconds": probe_seconds}
        # train + eval (+ best.pt write); the MaskedKD diagnostics run only in masked runs and are
        # kept out so wall-time ratios compare like with like (probe_seconds is recorded apart).
        row["epoch_seconds"] = time.perf_counter() - begun - probe_seconds
        history.append(row)
        if epoch % cfg["checkpoint_every"] == 0 or epoch == total_epochs:
            started = time.perf_counter()
            save_checkpoint(resume_path, {"signature": signature, "epoch": epoch, "model": cpu_state(trainee),
                                          "optimizer": optimizer.state_dict(), "history": history,
                                          "probes": probes, "best_epoch": best_epoch, "best_score": best_score,
                                          "best_model": best_weights})
            row["checkpoint_seconds"] = time.perf_counter() - started
        write_json(directory / "history.json", history)
        if probes:
            write_json(directory / "probes.json", probes)
        teacher_part = (f" teacher_fwd={training['teacher_seconds']:.1f}s"
                        if training["teacher_seconds"] is not None else "")
        print(f"{run.name} {epoch}/{total_epochs} lr={lr:.2e} loss={training['loss']:.4f} "
              f"val_macro={score:.4f} best={best_score:.4f}@{best_epoch} "
              f"train={training['seconds']:.1f}s{teacher_part} ({training['samples_per_second']:.0f} img/s)",
              flush=True)

    # Test is evaluated only after the validation-selected checkpoint is fixed.
    last_state = cpu_state(trainee)
    last_test = evaluate(trainee, data.test, normalizer, eval_batch, device, precision, num_classes)
    trainee.load_state_dict(best_weights)
    best_test = evaluate(trainee, data.test, normalizer, eval_batch, device, precision, num_classes)
    test_mask = None
    if run.masked:
        test_mask = mask_probe(trainee, teacher, data.test, normalizer, eval_batch, device, precision, keep,
                               record=True)
        test_mask["sample_ids"] = list(data.test.ids)
        write_json(directory / "test_mask.json", test_mask)
    if cfg.get("save_last"):
        save_checkpoint(directory / "last.pt", {"model": last_state, "arch": settings["arch"],
                                                "num_classes": num_classes, "epoch": total_epochs})

    train_seconds = sum(row["train"]["seconds"] for row in history)
    teacher_seconds = (sum(row["train"]["teacher_seconds"] for row in history)
                       if history and history[0]["train"]["teacher_seconds"] is not None else None)
    samples = sum(row["train"]["samples"] for row in history)
    train_flops = samples * analytic["total"]
    eval_flops = total_epochs * len(data.val) * flop_model.forward_flops(settings["arch"], num_classes)
    result = {
        "signature": signature, "suite": suite, "run": run.name, "role": role, "seed": run.seed,
        "masked": run.masked, "keep_patches": keep, "teacher_run": run.teacher,
        "arch": settings["arch"], "teacher_arch": teacher_ckpt["arch"] if teacher_ckpt else "",
        "init": settings["init"], "init_provenance": init_provenance,
        "parameters": parameter_count(trainee), "epochs": total_epochs, "batch_size": cfg["batch_size"],
        # smallest micro-batch used in any session (a resumed run may have been split only in part)
        "micro_batch": min([row.get("micro_batch", micro) for row in history] + [micro]),
        "train_samples_per_epoch": len(data.train), "class_names": data.classes,
        "data_sha256": data.manifest_sha256, "teacher_sha256": teacher_sha,
        "code_sha256": code_hash(), "code_commit": source_commit(), "settings": settings,
        "selection_metric": "macro_accuracy", "best_epoch": best_epoch, "best_validation": best_score,
        "best_test": best_test, "last_test": last_test,
        "test_mask": {k: v for k, v in test_mask.items() if k not in ("selection_hex", "predictions", "sample_ids")}
        if test_mask else None,
        "flops_per_sample": analytic, "flops_per_sample_measured": measured,
        "train_samples": samples, "train_flops": train_flops, "eval_flops": eval_flops,
        "train_seconds": train_seconds, "teacher_forward_seconds": teacher_seconds,
        "eval_seconds": sum(row["eval_seconds"] for row in history),
        "probe_seconds": sum(row["probe_seconds"] for row in history) + (probes[0]["seconds"] if probes and probes[0]["epoch"] == 0 else 0.0),
        "wall_seconds": sum(row["epoch_seconds"] + row.get("checkpoint_seconds", 0.0) for row in history),
        "samples_per_second": samples / train_seconds if train_seconds else None,
        "achieved_tflops": train_flops / train_seconds / 1e12 if train_seconds else None,
        "peak_vram_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0,
        "checkpoint_sha256": sha256(directory / "best.pt"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"},
        # most other GPU processes seen at the start of any session of this run
        "gpu_other_processes": max((row["gpu_other_processes"] for row in history
                                    if row.get("gpu_other_processes") is not None), default=None),
        "debug": debug,
    }
    write_json(result_path, result)
    if not cfg.get("keep_resume") and resume_path.exists():
        resume_path.unlink()  # history.json/result.json keep everything needed; saves ~GBs per run
    return result
