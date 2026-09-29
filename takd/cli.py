"""Command line entry point.

    .venv/bin/python -m takd.cli prepare                 # COCO single + GPU cache + DeiT weights
    .venv/bin/python -m takd.cli plan   --suite hkd      # schedule and analytic FLOPs per group
    .venv/bin/python -m takd.cli run    --suite all      # train everything (resumable, sequential)
    .venv/bin/python -m takd.cli status --suite all
    .venv/bin/python -m takd.cli report --suite all      # results/<suite>/*.csv, SUMMARY.md, figures
    .venv/bin/python -m takd.cli smoke                   # tiny synthetic end-to-end check (CPU or GPU)

The scripts/*.sh wrappers set the repo-local environment first; prefer them on the server.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("hkd", "logit")
DEFAULT_DATA = ROOT / "data" / "coco_single"
DEFAULT_CACHE = ROOT / "data" / "cache" / "coco_single_224.pt"
DEFAULT_OUTPUT = ROOT / "outputs"
SMOKE_OUTPUT = ROOT / "outputs" / "smoke"
# Manifest SHA-256 of the benchmark's published COCO single runs (results/coco/runs.csv upstream).
REFERENCE_DATA_SHA256 = "072f811cf4f7f95f889b0c8551c93853d21ca54aae47028e3185b1b7f4027fb9"

DEBUG_OVERRIDES = {
    "seeds": [0, 1],
    "archs": {"teacher": "debug_base", "assistant": "debug_small_w2", "student": "debug_small"},
    "init": {"teacher": "scratch", "assistant": "widen", "student": "scratch"},
    "widen_from": "debug_small",
    "epochs": {"teacher": 3, "assistant": 3, "student": 4},
    "batch_size": 16, "eval_batch_size": 32, "warmup_epochs": 1, "probe_epochs": [0, 2, 4],
    "checkpoint_every": 1, "learning_rate": 1e-3,
}
REQUIRED = ("seeds", "mask_ratio", "archs", "init", "epochs", "batch_size", "eval_batch_size", "warmup_epochs",
            "learning_rate", "min_learning_rate", "weight_decay", "drop_path", "gradient_clip", "precision",
            "loss", "probe_epochs", "checkpoint_every")


def load_config(suite, debug=False):
    cfg = json.loads((ROOT / "configs" / f"{suite}.json").read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in cfg]
    if missing:
        raise ValueError(f"configs/{suite}.json lacks {missing}")
    if debug:
        for key, value in DEBUG_OVERRIDES.items():
            cfg[key] = copy.deepcopy(value)
        for stage in cfg["loss"].get("stages", {}).values():  # debug models have 2 blocks
            stage["taps"] = [1] if stage["taps"] else []
    if not 0.0 <= cfg["mask_ratio"] < 1.0:
        raise ValueError("mask_ratio must be in [0, 1)")
    return cfg


def suites_of(name):
    return list(SUITES) if name == "all" else [name]


def data_sha_of(cache, debug):
    if debug:
        return "synthetic"
    import torch
    if not Path(cache).is_file():
        raise SystemExit(f"Data cache missing: {cache}\nRun `bash scripts/prepare_data.sh` first.")
    return torch.load(cache, map_location="cpu", weights_only=True, mmap=True)["manifest_sha256"]


# --------------------------------------------------------------------------- prepare

def cmd_prepare(args):
    import torch

    from .data import build_cache
    from .models import imagenet_file, imagenet_state

    data_root = Path(args.data_root).resolve()
    if not (data_root / "manifest.json").is_file():
        print(f"Preparing COCO single in {data_root} (downloads COCO 2017 annotations + eligible images)", flush=True)
        subprocess.run([sys.executable, str(ROOT / "datasets" / "coco_single" / "prepare.py"),
                        "--output", str(data_root), "--workers", str(args.workers)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "check_assets.py"), "coco", str(data_root)]
                   + (["--full"] if args.full_check else []), cwd=ROOT, check=True)
    cache = build_cache(data_root, args.data_cache, workers=args.workers)
    header = torch.load(cache, map_location="cpu", weights_only=True, mmap=True)
    sha = header["manifest_sha256"]
    same = "matches" if sha == REFERENCE_DATA_SHA256 else "DIFFERS from"
    print(f"Manifest SHA-256 {sha} ({same} the kshs-aimlab-benchmarks COCO single reference)", flush=True)
    for arch in ("deit_small", "deit_base"):
        imagenet_state(arch)
        print(f"ImageNet weights ready: {imagenet_file(arch)}", flush=True)


# --------------------------------------------------------------------------- plan

def cmd_plan(args):
    from .data import NUM_PATCHES
    from .engine import keep_patches
    from .flops import step_flops_per_sample
    from .plan import BASELINE, GROUPS, group_runs, schedule

    for suite in suites_of(args.suite):
        cfg = load_config(suite, args.debug)
        seeds = args.seeds or cfg["seeds"]
        n_train = 3210 if args.debug is False else 160
        per_epoch = (n_train // cfg["batch_size"]) * cfg["batch_size"]
        order = schedule(seeds, args.groups or tuple(GROUPS))
        print(f"== suite {suite}: {len(order)} runs, seeds {seeds}, mask ratio {cfg['mask_ratio']} "
              f"(teacher sees {keep_patches(cfg)}/{NUM_PATCHES} patches when masked)")
        for index, run in enumerate(order, 1):
            print(f"  {index:2d}. {run.name:28s} role={run.role:9s} masked={str(run.masked):5s} "
                  f"teacher={run.teacher or '-'}")
        arch_of = {"teacher": cfg["archs"]["teacher"], "assistant": cfg["archs"]["assistant"]}

        def cost(run):
            teacher_arch = arch_of["teacher"] if run.teacher == "teacher" else arch_of["assistant"]
            keep = keep_patches(cfg) if run.masked else NUM_PATCHES
            per_sample = step_flops_per_sample(cfg["archs"][run.role], teacher_arch, 10, keep)["total"]
            return per_sample * per_epoch * cfg["epochs"][run.role]

        base = sum(cost(r) for r in group_runs(BASELINE, seeds[0]))
        print(f"  analytic training FLOPs per group (one seed, {per_epoch} samples/epoch):")
        for group in GROUPS:
            total = sum(cost(r) for r in group_runs(group, seeds[0]))
            print(f"    {group}: {total / 1e15:8.2f} PFLOP   ratio to {BASELINE}: {total / base:.3f}")


# --------------------------------------------------------------------------- run

def gpu_snapshot():
    """(text, other compute processes) from nvidia-smi, or ("", None) when unavailable."""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return "", None
    others = [line for line in apps.stdout.splitlines() if line.strip()]
    return gpu.stdout.strip(), len(others)


def tail(path, lines=40):
    try:
        return "\n".join(deque(Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), maxlen=lines))
    except OSError:
        return f"(log unavailable: {path})"


def run_worker(command, log, extra_env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, encoding="utf-8", errors="replace",
                                   env={**os.environ, "PYTHONUNBUFFERED": "1", **(extra_env or {})})
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise


def completed(run, cfg, suite, output_root, data_sha, debug, allow_code_change):
    from .data import sha256
    from .engine import run_directory, run_signature

    result_path = run_directory(output_root, suite, run.name) / "result.json"
    if not result_path.is_file():
        return False
    result = json.loads(result_path.read_text(encoding="utf-8"))
    teacher_sha = sha256(run_directory(output_root, suite, run.teacher) / "best.pt") if run.teacher else ""
    if result["signature"] == run_signature(run, cfg, teacher_sha, data_sha, debug):
        return True
    if allow_code_change:
        print(f"WARNING {run.name}: kept although code/config changed since it ran (--allow-code-change)")
        return True
    raise SystemExit(f"{result_path.parent} was produced by different code, config or teacher.\n"
                     f"Move that directory aside to retrain it, or pass --allow-code-change to keep it.")


def safe_export(output_root, suite, seeds, results_root):
    """Reports are a by-product; a failure there must never stop the training queue."""
    from .report import export
    try:
        export(output_root, suite, seeds, results_root=results_root)
    except Exception as error:  # noqa: BLE001
        print(f"WARNING: report export failed ({error!r}); training continues. "
              f"Rebuild later with: bash scripts/report.sh", flush=True)


def cmd_run(args):
    from .engine import run_directory
    from .plan import GROUPS, schedule

    output_root = Path(args.output_root or (SMOKE_OUTPUT if args.debug else DEFAULT_OUTPUT)).resolve()
    results_root = (output_root / "results") if args.debug else None
    data_sha = data_sha_of(args.data_cache, args.debug)
    started = time.time()
    for suite in suites_of(args.suite):
        cfg = load_config(suite, args.debug)
        seeds = args.seeds or cfg["seeds"]
        report_seeds = sorted(set(cfg["seeds"]) | set(seeds))  # a --seeds subset must not shrink results/
        order = schedule(seeds, args.groups or tuple(GROUPS))
        print(f"== suite {suite}: {len(order)} runs -> {output_root}", flush=True)
        for index, run in enumerate(order, 1):
            if completed(run, cfg, suite, output_root, data_sha, args.debug, args.allow_code_change):
                print(f"[{suite} {index}/{len(order)}] DONE {run.name}", flush=True)
                continue
            snapshot, others = gpu_snapshot()
            if others:
                print(f"WARNING: {others} other process(es) are using the GPU; training times will be inflated. "
                      f"Measured times are only comparable between runs made under the same load.", flush=True)
            print(f"[{suite} {index}/{len(order)}] START {run.name}  GPU: {snapshot or 'n/a'}", flush=True)
            directory = run_directory(output_root, suite, run.name)
            command = [sys.executable, "-m", "takd.cli", "_worker", "--suite", suite, "--run", run.name,
                       "--output-root", str(output_root), "--device", args.device,
                       "--data-cache", str(args.data_cache)]
            command += ["--debug"] * args.debug + ["--allow-code-change"] * args.allow_code_change
            code = run_worker(command, directory / "train.log",
                              {"TAKD_GPU_OTHER_PROCESSES": "" if others is None else str(others)})
            if code:
                raise SystemExit(f"{run.name} failed with exit code {code}. Last lines of {directory / 'train.log'}:\n"
                                 f"{tail(directory / 'train.log')}\nFix the cause and rerun the same command to resume.")
            safe_export(output_root, suite, report_seeds, results_root)
        safe_export(output_root, suite, report_seeds, results_root)
    print(f"All requested runs complete ({(time.time() - started) / 3600:.2f} h this session).", flush=True)


def cmd_worker(args):
    from .engine import train_run
    from .plan import run_by_name

    cfg = load_config(args.suite, args.debug)
    run = run_by_name(args.run)
    result = train_run(run, cfg, args.suite, Path(args.output_root), args.device, args.debug,
                       Path(args.data_cache), args.allow_code_change)
    print(json.dumps({"run": run.name, "best_epoch": result["best_epoch"],
                      "best_validation": result["best_validation"],
                      "test_macro": result["best_test"]["macro_accuracy"],
                      "train_hours": result["train_seconds"] / 3600}), flush=True)


def cmd_status(args):
    from .engine import run_directory
    from .plan import GROUPS, schedule

    output_root = Path(args.output_root or (SMOKE_OUTPUT if args.debug else DEFAULT_OUTPUT)).resolve()
    for suite in suites_of(args.suite):
        cfg = load_config(suite, args.debug)
        order = schedule(args.seeds or cfg["seeds"], args.groups or tuple(GROUPS))
        done = 0
        print(f"== suite {suite}")
        for run in order:
            directory = run_directory(output_root, suite, run.name)
            history = directory / "history.json"
            epochs = len(json.loads(history.read_text(encoding="utf-8"))) if history.is_file() else 0
            complete = (directory / "result.json").is_file()
            done += complete
            state = "complete" if complete else ("running/interrupted" if epochs else "pending")
            print(f"  {run.name:28s} {epochs:3d}/{cfg['epochs'][run.role]:<3d} {state}")
        print(f"  {done}/{len(order)} runs complete")


def cmd_report(args):
    from .report import export

    output_root = Path(args.output_root or (SMOKE_OUTPUT if args.debug else DEFAULT_OUTPUT)).resolve()
    for suite in suites_of(args.suite):
        cfg = load_config(suite, args.debug)
        export(output_root, suite, sorted(set(cfg["seeds"]) | set(args.seeds or [])),
               target_fraction=args.target_fraction,
               results_root=(output_root / "results") if args.debug else None)


def cmd_smoke(args):
    import torch

    args.debug = True
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root = args.output_root or str(SMOKE_OUTPUT)
    args.suite, args.seeds, args.groups, args.allow_code_change = "all", None, None, False
    cmd_run(args)
    print(f"Smoke test passed. Tables: {Path(args.output_root) / 'results'}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "plan", "run", "status", "report", "smoke", "_worker"):
        q = sub.add_parser(name)
        q.add_argument("--data-cache", default=str(DEFAULT_CACHE))
        if name == "prepare":
            q.add_argument("--data-root", default=str(DEFAULT_DATA))
            q.add_argument("--workers", type=int, default=8)
            q.add_argument("--full-check", action="store_true", help="hash every image while validating")
            continue
        if name in ("plan", "run", "status", "report"):
            q.add_argument("--suite", choices=SUITES + ("all",), default="all")
            q.add_argument("--seeds", type=int, nargs="+")
            q.add_argument("--groups", nargs="+", choices=tuple("ABCDEF"))
            q.add_argument("--debug", action="store_true", help="tiny models on synthetic data")
        if name in ("run", "status", "report", "smoke", "_worker"):
            q.add_argument("--output-root")
        if name in ("run", "_worker"):
            q.add_argument("--allow-code-change", action="store_true",
                           help="reuse finished runs even if training code/config changed since")
        if name in ("run", "smoke", "_worker"):
            q.add_argument("--device", choices=("cuda", "cpu"), default=None if name == "smoke" else "cuda")
        if name == "report":
            q.add_argument("--target-fraction", type=float, default=0.99,
                           help="to-target metrics use this fraction of F's best validation accuracy")
        if name == "_worker":
            q.add_argument("--suite", choices=SUITES, required=True)
            q.add_argument("--run", required=True)
            q.add_argument("--debug", action="store_true")
    return p


def main():
    args = parser().parse_args()
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    handlers = {"prepare": cmd_prepare, "plan": cmd_plan, "run": cmd_run, "status": cmd_status,
                "report": cmd_report, "smoke": cmd_smoke, "_worker": cmd_worker}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
