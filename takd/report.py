"""Tables for the plan: FLOPs, training time and accuracy of A-E relative to F.

Per (group, seed) the cost is stage 1 (the T -> A run the group uses, if any)
plus stage 2 (its student run). Ratios X/F are formed within a seed and then
averaged over seeds (mean and sample standard deviation). The shared teacher's
fine-tuning is identical for every group and is reported on its own.
"""
import csv
import json
import os
import statistics
from pathlib import Path

from .engine import ROOT, run_directory
from .plan import BASELINE, GROUPS, group_runs, teacher_run

RUN_FIELDS = (
    "suite", "run", "role", "seed", "masked", "keep_patches", "arch", "teacher_arch", "teacher_run", "init",
    "epochs", "batch_size", "micro_batch", "best_epoch", "val_macro", "test_macro", "test_overall",
    "last_test_macro", "last_test_overall", "flops_per_sample", "teacher_flops_per_sample",
    "flops_per_sample_measured", "train_flops", "train_seconds", "teacher_forward_seconds", "wall_seconds",
    "samples_per_second", "achieved_tflops", "peak_vram_gib", "gpu_other_processes", "gpu", "torch",
    "data_sha256", "teacher_sha256",
    "checkpoint_sha256", "code_sha256", "code_commit", "init_provenance")
GROUP_FIELDS = (
    "suite", "group", "seed", "assistant_masked", "student_masked", "stage1_run", "stage2_run",
    "stage1_flops", "stage2_flops", "total_flops", "stage1_train_seconds", "stage2_train_seconds",
    "total_train_seconds", "total_wall_seconds", "teacher_forward_seconds", "achieved_tflops",
    "val_macro", "test_macro", "test_overall", "last_test_macro", "last_test_overall",
    "target_val_macro", "flops_to_target", "seconds_to_target")
RATIO_METRICS = ("total_flops", "total_train_seconds", "total_wall_seconds", "achieved_tflops",
                 "test_macro", "test_overall", "last_test_macro", "flops_to_target", "seconds_to_target")
CENSORED = ("flops_to_target", "seconds_to_target")
SUMMARY_METRICS = ("total_flops", "total_train_seconds", "total_wall_seconds", "achieved_tflops",
                   "val_macro", "test_macro", "test_overall", "last_test_macro", "flops_to_target",
                   "seconds_to_target")
EPOCH_FIELDS = ("suite", "run", "role", "seed", "epoch", "lr", "train_loss", "train_ce", "train_flow",
                "train_logit_kd", "train_accuracy", "val_macro", "val_overall", "train_seconds",
                "teacher_forward_seconds", "samples_per_second", "cumulative_train_flops")
CLASS_FIELDS = ("suite", "run", "seed", "class_index", "class_name", "correct", "n", "accuracy")
PROBE_FIELDS = ("suite", "run", "seed", "epoch", "n", "student_accuracy", "teacher_full_accuracy",
                "teacher_masked_accuracy", "full_correct_masked_wrong", "full_wrong_masked_correct",
                "full_masked_disagreement", "full_student_disagreement", "masked_student_disagreement",
                "kl_full_to_masked", "foreground_recall", "foreground_precision", "foreground_images",
                "selection_overlap_initial", "selection_overlap_previous")


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except json.JSONDecodeError:  # being rewritten by a running worker; treat as absent this time
        return None


def run_row(result, suite):
    fps = result["flops_per_sample"]
    return {
        "suite": suite, "run": result["run"], "role": result["role"], "seed": result["seed"],
        "masked": result["masked"], "keep_patches": result["keep_patches"], "arch": result["arch"],
        "teacher_arch": result["teacher_arch"], "teacher_run": result["teacher_run"], "init": result["init"],
        "epochs": result["epochs"], "batch_size": result["batch_size"], "micro_batch": result["micro_batch"],
        "best_epoch": result["best_epoch"], "val_macro": result["best_validation"],
        "test_macro": result["best_test"]["macro_accuracy"], "test_overall": result["best_test"]["overall_accuracy"],
        "last_test_macro": result["last_test"]["macro_accuracy"],
        "last_test_overall": result["last_test"]["overall_accuracy"],
        "flops_per_sample": fps["total"], "teacher_flops_per_sample": fps["teacher"],
        "flops_per_sample_measured": result["flops_per_sample_measured"], "train_flops": result["train_flops"],
        "train_seconds": result["train_seconds"], "teacher_forward_seconds": result["teacher_forward_seconds"],
        "wall_seconds": result["wall_seconds"], "samples_per_second": result["samples_per_second"],
        "achieved_tflops": result["achieved_tflops"], "peak_vram_gib": result["peak_vram_gib"],
        "gpu_other_processes": result.get("gpu_other_processes"), "gpu": result["environment"]["gpu"], "torch": result["environment"]["torch"],
        "data_sha256": result["data_sha256"], "teacher_sha256": result["teacher_sha256"],
        "checkpoint_sha256": result["checkpoint_sha256"], "code_sha256": result["code_sha256"],
        "code_commit": result["code_commit"], "init_provenance": result["init_provenance"],
    }


def reach(history, target, flops_offset, seconds_offset):
    """Cumulative (FLOPs, train seconds) when validation macro accuracy first reaches target."""
    seconds = seconds_offset
    for row in history:
        seconds += row["train"]["seconds"]
        if row["validation"]["macro_accuracy"] >= target:
            return flops_offset + row["cumulative_train_flops"], seconds
    return None, None


def group_row(suite, group, seed, results, histories, target):
    stage_runs = group_runs(group, seed)
    if any(run.name not in results for run in stage_runs):
        return None
    student = results[stage_runs[-1].name]
    stage1 = results[stage_runs[0].name] if len(stage_runs) == 2 else None
    s1_flops = stage1["train_flops"] if stage1 else 0.0
    s1_seconds = stage1["train_seconds"] if stage1 else 0.0
    teacher_seconds = [r["teacher_forward_seconds"] for r in (stage1, student) if r]
    total_flops = s1_flops + student["train_flops"]
    total_seconds = s1_seconds + student["train_seconds"]
    flops_to, seconds_to = (reach(histories[student["run"]], target, s1_flops, s1_seconds)
                            if target is not None else (None, None))
    return {
        "suite": suite, "group": group, "seed": seed, "assistant_masked": GROUPS[group]["assistant_masked"],
        "student_masked": GROUPS[group]["student_masked"], "stage1_run": stage1["run"] if stage1 else "",
        "stage2_run": student["run"], "stage1_flops": s1_flops, "stage2_flops": student["train_flops"],
        "total_flops": total_flops, "stage1_train_seconds": s1_seconds,
        "stage2_train_seconds": student["train_seconds"], "total_train_seconds": total_seconds,
        "total_wall_seconds": (stage1["wall_seconds"] if stage1 else 0.0) + student["wall_seconds"],
        "teacher_forward_seconds": sum(teacher_seconds) if all(v is not None for v in teacher_seconds) else None,
        "achieved_tflops": total_flops / total_seconds / 1e12 if total_seconds else None,
        "val_macro": student["best_validation"], "test_macro": student["best_test"]["macro_accuracy"],
        "test_overall": student["best_test"]["overall_accuracy"],
        "last_test_macro": student["last_test"]["macro_accuracy"],
        "last_test_overall": student["last_test"]["overall_accuracy"],
        "target_val_macro": target, "flops_to_target": flops_to, "seconds_to_target": seconds_to,
    }


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else None), len(values)


def export(output_root, suite, seeds, target_fraction=0.99, results_root=None):
    output_root = Path(output_root)
    destination = Path(results_root or ROOT / "results") / suite
    runs, histories = {}, {}
    names = {teacher_run().name} | {run.name for seed in seeds for g in GROUPS for run in group_runs(g, seed)}
    for name in sorted(names):
        directory = run_directory(output_root, suite, name)
        result = read_json(directory / "result.json")
        if result is None:
            continue
        runs[name] = result
        histories[name] = read_json(directory / "history.json") or []

    run_rows = [run_row(r, suite) for r in runs.values()]
    epoch_rows, class_rows, probe_rows, test_mask_rows = [], [], [], []
    for name, result in runs.items():
        for row in histories[name]:
            train = row["train"]
            epoch_rows.append({
                "suite": suite, "run": name, "role": result["role"], "seed": result["seed"], "epoch": row["epoch"],
                "lr": row["lr"], "train_loss": train["loss"], "train_ce": train.get("ce"),
                "train_flow": train.get("flow"), "train_logit_kd": train.get("logit_kd"),
                "train_accuracy": train["accuracy"], "val_macro": row["validation"]["macro_accuracy"],
                "val_overall": row["validation"]["overall_accuracy"], "train_seconds": train["seconds"],
                "teacher_forward_seconds": train["teacher_seconds"], "samples_per_second": train["samples_per_second"],
                "cumulative_train_flops": row["cumulative_train_flops"]})
        test = result["best_test"]
        for index, (correct, n, accuracy) in enumerate(zip(test["class_correct"], test["class_counts"],
                                                           test["class_accuracy"])):
            class_rows.append({"suite": suite, "run": name, "seed": result["seed"], "class_index": index,
                               "class_name": result["class_names"][index], "correct": correct, "n": n,
                               "accuracy": accuracy})
        for probe in read_json(run_directory(output_root, suite, name) / "probes.json") or []:
            probe_rows.append({"suite": suite, "run": name, "seed": result["seed"], **probe})
        if result.get("test_mask"):
            test_mask_rows.append({"suite": suite, "run": name, "seed": result["seed"], "epoch": "best",
                                   **result["test_mask"]})

    group_rows = []
    for seed in seeds:
        baseline = runs.get(group_runs(BASELINE, seed)[-1].name)
        target = baseline["best_validation"] * target_fraction if baseline else None
        for group in GROUPS:
            row = group_row(suite, group, seed, runs, histories, target)
            if row:
                group_rows.append(row)

    by_key = {(row["group"], row["seed"]): row for row in group_rows}
    summary_rows, ratio_rows = [], []
    for group in GROUPS:
        rows = [by_key[(group, s)] for s in seeds if (group, s) in by_key]
        if not rows:
            continue
        summary = {"suite": suite, "group": group, "seeds": "|".join(str(r["seed"]) for r in rows)}
        for metric in SUMMARY_METRICS:
            summary[f"{metric}_mean"], summary[f"{metric}_sd"], count = mean_sd([r[metric] for r in rows])
            if metric in CENSORED:
                summary[f"{metric}_n"] = count  # seeds that reached the target (others never did)
        summary_rows.append(summary)
        paired = [(r, by_key[(BASELINE, r["seed"])]) for r in rows if (BASELINE, r["seed"]) in by_key]
        if not paired:
            continue
        ratio = {"suite": suite, "group": group, "seeds": "|".join(str(r["seed"]) for r, _ in paired)}
        for metric in RATIO_METRICS:
            values = [r[metric] / f[metric] if r[metric] is not None and f[metric] else None for r, f in paired]
            ratio[f"{metric}_ratio_mean"], ratio[f"{metric}_ratio_sd"], ratio[f"{metric}_n"] = mean_sd(values)
        deltas = [100.0 * (r["test_macro"] - f["test_macro"]) for r, f in paired]
        ratio["delta_test_macro_pp_mean"], ratio["delta_test_macro_pp_sd"], _ = mean_sd(deltas)
        efficiency = [(r["test_macro"] / f["test_macro"]) / (r["total_flops"] / f["total_flops"])
                      for r, f in paired if f["test_macro"] and f["total_flops"]]
        ratio["accuracy_per_flop_ratio_mean"], ratio["accuracy_per_flop_ratio_sd"], _ = mean_sd(efficiency)
        ratio_rows.append(ratio)

    summary_fields = (("suite", "group", "seeds")
                      + tuple(f"{m}_{s}" for m in SUMMARY_METRICS for s in ("mean", "sd"))
                      + tuple(f"{m}_n" for m in CENSORED))
    ratio_fields = (("suite", "group", "seeds")
                    + tuple(f"{m}_{s}" for m in RATIO_METRICS for s in ("ratio_mean", "ratio_sd", "n"))
                    + ("delta_test_macro_pp_mean", "delta_test_macro_pp_sd",
                       "accuracy_per_flop_ratio_mean", "accuracy_per_flop_ratio_sd"))
    write_csv(destination / "runs.csv", RUN_FIELDS, run_rows)
    write_csv(destination / "groups.csv", GROUP_FIELDS, group_rows)
    write_csv(destination / "summary.csv", summary_fields, summary_rows)
    write_csv(destination / "ratios.csv", ratio_fields, ratio_rows)
    write_csv(destination / "epochs.csv", EPOCH_FIELDS, epoch_rows)
    write_csv(destination / "per_class.csv", CLASS_FIELDS, class_rows)
    write_csv(destination / "mask_validation.csv", PROBE_FIELDS, probe_rows)
    write_csv(destination / "mask_test.csv", PROBE_FIELDS, test_mask_rows)
    (destination / "SUMMARY.md").write_text(markdown(suite, runs, summary_rows, ratio_rows, seeds, target_fraction),
                                            encoding="utf-8")
    try:
        from .figures import draw
        draw(destination, suite, runs, histories, group_rows, ratio_rows, seeds)
    except ImportError as error:
        print(f"Figures skipped ({error})", flush=True)
    print(f"Exported {len(runs)} completed runs, {len(group_rows)} group rows -> {destination}", flush=True)
    return destination


def _fmt(mean, sd, scale=1.0, digits=3, unit=""):
    if mean is None:
        return "–"
    text = f"{mean * scale:.{digits}f}"
    if sd is not None:
        text += f" ± {sd * scale:.{digits}f}"
    return text + unit


def _fmt_target(row, key, digits=3):
    """Censored metric: seeds that never reached the target are counted, not silently dropped."""
    total = len(row["seeds"].split("|")) if row["seeds"] else 0
    reached = row.get(f"{key}_n") or 0
    if reached == 0:
        return f"미도달 (0/{total})"
    value = _fmt(row[f"{key}_ratio_mean"] if f"{key}_ratio_mean" in row else row[f"{key}_mean"],
                 row[f"{key}_ratio_sd"] if f"{key}_ratio_sd" in row else row[f"{key}_sd"], digits=digits)
    return value if reached == total else f"{value} ({reached}/{total} 도달)"


def markdown(suite, runs, summary_rows, ratio_rows, seeds, target_fraction):
    ratios = {row["group"]: row for row in ratio_rows}
    summaries = {row["group"]: row for row in summary_rows}
    labels = {"A": "T→A mask, A→S full", "B": "T→A full, A→S mask", "C": "T→A mask, A→S mask",
              "D": "T→A full, A→S full (TAKD)", "E": "T→S mask (direct)", "F": "T→S full (direct, 기준)"}
    lines = [f"# 결과 요약 — suite `{suite}`", "",
             f"요청한 seed: {', '.join(map(str, seeds))}. 각 비율은 같은 seed의 F로 나눈 뒤, 두 run이 모두 끝난 seed들로",
             "평균 ± 표본표준편차를 냈다(사용한 seed는 표의 `seed` 열).",
             "연산량(FLOPs)은 해석적 모델 FLOPs(곱-덧셈 = 2 FLOPs, 행렬곱만), 학습 시간은 최적화 루프만(평가·진단·체크포인트 제외),",
             "달성 FLOP/s(초당 연산 속도) = 학습 FLOPs / 학습 시간. 정확도는 validation으로 고른 체크포인트의 test macro accuracy.",
             f"도달 FLOPs는 validation macro accuracy가 같은 seed F 최고값의 {target_fraction:.0%}에 처음 닿을 때까지의 누적 학습 FLOPs",
             "(1단계 T→A 포함). 끝까지 닿지 못한 seed는 평균에서 빠지므로 `(k/n 도달)`로 표시했다.", "",
             "## F 대비 비율", "",
             "| 군 | 설정 | seed | FLOPs 비 | 학습시간 비 | FLOP/s 비 | 정확도 비 | Δ정확도 (pp) | 정확도/FLOPs 비 | 도달 FLOPs 비 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for group in GROUPS:
        r = ratios.get(group)
        if not r:
            lines.append(f"| {group} | {labels[group]} | – | 미완료 | | | | | | |")
            continue
        lines.append("| " + " | ".join([
            group, labels[group], r["seeds"].replace("|", ","),
            _fmt(r["total_flops_ratio_mean"], r["total_flops_ratio_sd"]),
            _fmt(r["total_train_seconds_ratio_mean"], r["total_train_seconds_ratio_sd"]),
            _fmt(r["achieved_tflops_ratio_mean"], r["achieved_tflops_ratio_sd"]),
            _fmt(r["test_macro_ratio_mean"], r["test_macro_ratio_sd"], digits=4),
            _fmt(r["delta_test_macro_pp_mean"], r["delta_test_macro_pp_sd"], digits=2),
            _fmt(r["accuracy_per_flop_ratio_mean"], r["accuracy_per_flop_ratio_sd"]),
            _fmt_target(r, "flops_to_target"),
        ]) + " |")
    lines += ["", "## 절대값", "",
              "| 군 | seed | 학습 FLOPs (PFLOP) | 학습 시간 (h) | 달성 TFLOP/s | test macro acc | test overall acc | val macro acc |",
              "|---|---|---|---|---|---|---|---|"]
    for group in GROUPS:
        s = summaries.get(group)
        if not s:
            continue
        lines.append("| " + " | ".join([
            group, s["seeds"].replace("|", ","), _fmt(s["total_flops_mean"], s["total_flops_sd"], 1e-15, 2),
            _fmt(s["total_train_seconds_mean"], s["total_train_seconds_sd"], 1 / 3600, 3),
            _fmt(s["achieved_tflops_mean"], s["achieved_tflops_sd"], 1, 1),
            _fmt(s["test_macro_mean"], s["test_macro_sd"], digits=4),
            _fmt(s["test_overall_mean"], s["test_overall_sd"], digits=4),
            _fmt(s["val_macro_mean"], s["val_macro_sd"], digits=4)]) + " |")
    teacher = runs.get("teacher")
    assistants = [r for name, r in sorted(runs.items()) if r["role"] == "assistant"]
    lines += ["", "## Teacher / Teacher assistant", "", "| run | test macro acc | 학습 FLOPs (PFLOP) | 학습 시간 (h) |",
              "|---|---|---|---|"]
    for result in ([teacher] if teacher else []) + assistants:
        lines.append(f"| {result['run']} | {result['best_test']['macro_accuracy']:.4f} | "
                     f"{result['train_flops'] / 1e15:.2f} | {result['train_seconds'] / 3600:.3f} |")
    contended = [r["run"] for r in runs.values() if r.get("gpu_other_processes")]
    if contended:
        lines += ["", f"**주의:** 다음 run은 시작 시 다른 GPU 프로세스가 있어 학습 시간이 부풀려졌을 수 있다: "
                      f"{', '.join(sorted(contended))}. FLOPs와 정확도는 영향받지 않는다."]
    micro = [r["run"] for r in runs.values() if r.get("micro_batch") and r["micro_batch"] != r["batch_size"]]
    if micro:
        lines += ["", f"**주의:** VRAM 부족으로 micro-batch를 쓴 run: {', '.join(sorted(micro))}."]
    lines += ["", "Teacher fine-tuning은 모든 군이 공유하므로 비율 계산에서 제외했다.", ""]
    return "\n".join(lines)
