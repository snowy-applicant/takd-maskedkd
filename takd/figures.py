"""Static figures for results/<suite>/ (matplotlib, headless)."""
from collections import defaultdict

from .plan import BASELINE, GROUPS, group_runs

# Validated categorical order (dataviz reference palette, light surface); fixed per group.
SERIES = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4", "F": "#008300"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw(destination, suite, runs, histories, group_rows, ratio_rows, seeds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Learning efficiency: validation macro accuracy vs cumulative training FLOPs
    #    (stage-1 T->A FLOPs included as an offset), averaged over completed seeds.
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    drawn = False
    for group in GROUPS:
        curves = []
        for seed in seeds:
            stages = group_runs(group, seed)
            if any(run.name not in runs for run in stages):
                continue
            offset = runs[stages[0].name]["train_flops"] if len(stages) == 2 else 0.0
            history = histories[stages[-1].name]
            curves.append([(offset + row["cumulative_train_flops"], row["validation"]["macro_accuracy"])
                           for row in history])
        if not curves:
            continue
        length = min(len(c) for c in curves)
        xs = [curves[0][i][0] / 1e15 for i in range(length)]
        ys = [sum(c[i][1] for c in curves) / len(curves) for i in range(length)]
        ax.plot(xs, ys, color=SERIES[group], linewidth=2, label=f"{group} (n={len(curves)})")
        drawn = True
    if drawn:
        ax.set_xlabel("Cumulative training PFLOP (T→A stage included)", color=INK, fontsize=10)
        ax.set_ylabel("Validation macro accuracy", color=INK, fontsize=10)
        ax.set_title(f"{suite}: accuracy vs training compute (mean over seeds)", color=INK, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(destination / "fig_accuracy_vs_flops.png", dpi=150)
    plt.close(fig)

    # 2) Ratios to F, one panel per metric (small multiples, shared baseline at 1).
    rows = [r for r in ratio_rows if r["group"] != BASELINE]
    if not rows:
        return
    panels = (("total_flops", "FLOPs / F"), ("total_train_seconds", "Training time / F"),
              ("achieved_tflops", "Achieved FLOP/s / F"), ("test_macro", "Test macro acc / F"))
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.6), facecolor=SURFACE)
    for ax, (metric, title) in zip(axes, panels):
        _style(ax)
        groups = [r["group"] for r in rows]
        means = [r[f"{metric}_ratio_mean"] or 0.0 for r in rows]
        sds = [r[f"{metric}_ratio_sd"] or 0.0 for r in rows]
        ax.bar(groups, means, yerr=sds, color=SERIES["A"], width=0.6, error_kw={"ecolor": MUTED, "elinewidth": 1})
        ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        if metric == "test_macro":
            low = min(m - s for m, s in zip(means, sds))
            high = max(m + s for m, s in zip(means, sds))
            pad = max(0.005, (high - low) * 0.3)
            ax.set_ylim(min(low, 1.0) - pad, max(high, 1.0) + pad)
    fig.suptitle(f"{suite}: groups A–E relative to F (mean ± sd over seeds)", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(destination / "fig_ratios.png", dpi=150)
    plt.close(fig)

    # 3) Where the time goes: teacher forward vs the rest, per group (first complete seed).
    split = defaultdict(dict)
    for row in group_rows:
        if row["teacher_forward_seconds"] is not None and "teacher" not in split[row["group"]]:
            split[row["group"]] = {"teacher": row["teacher_forward_seconds"] / 3600,
                                   "rest": (row["total_train_seconds"] - row["teacher_forward_seconds"]) / 3600,
                                   "seed": row["seed"]}
    if split:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        _style(ax)
        groups = [g for g in GROUPS if g in split]
        teacher = [split[g]["teacher"] for g in groups]
        rest = [split[g]["rest"] for g in groups]
        ax.bar(groups, rest, color=SERIES["A"], width=0.6, label="trainee forward+backward+update",
               edgecolor=SURFACE, linewidth=2)
        ax.bar(groups, teacher, bottom=rest, color=SERIES["B"], width=0.6, label="frozen-teacher forward",
               edgecolor=SURFACE, linewidth=2)
        ax.set_ylabel("Training hours (both stages)", color=INK, fontsize=10)
        ax.set_title(f"{suite}: training time split (seed {split[groups[0]]['seed']})", color=INK, fontsize=11,
                     loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(destination / "fig_time_split.png", dpi=150)
        plt.close(fig)
