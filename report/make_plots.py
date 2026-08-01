"""
Draws the figures for the report from the exported run history and the
saved predictions. The homework asks for matplotlib figures made from the
logs rather than screenshots of the experiment tracker.

Get the history first with export_wandb.py, then run from the repository
root:

    python3 report/make_plots.py
"""

import csv
import importlib
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
compute_eer = importlib.import_module("src.metrics.eer").compute_eer

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
PROTOCOL = (
    ROOT.parent / "deep-learning-research" / "hw" / "ASVspoof2019.LA.cm.eval.trl.txt"
)

HISTORY = REPORT_DIR / "wandb_4-fft-zero-pad-final.csv"
PREDICTIONS = ROOT / "avistratov.csv"


STEPS_PER_EPOCH = 3172
ATTACKS = [f"A{i:02d}" for i in range(7, 20)]


def read_protocol(path):
    labels, attacks = {}, {}
    for line in Path(path).read_text().splitlines():
        _, key, _, attack, label = line.split()
        labels[key] = 1 if label == "bonafide" else 0
        attacks[key] = attack
    return labels, attacks


def read_scores(path):
    scores = {}
    with open(path) as f:
        for row in csv.reader(f):
            if len(row) == 2:
                scores[row[0]] = float(row[1])
    return scores


def plot_training(history, out):
    """
    Training loss against the epoch, and the two per-epoch curves next to it.

    The training loss is logged every 50 steps, the evaluation metrics once
    per epoch, so the two live on different grids and the step count is
    converted to a fractional epoch.
    """
    train = history[history["loss_train"].notna()]
    val = history[history["EER_val"].notna()]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))


    train_epoch = train["_step"] / STEPS_PER_EPOCH
    left.plot(train_epoch, train["loss_train"], lw=0.5, alpha=0.25, color="tab:blue")

    binned = train.groupby(np.ceil(train_epoch).astype(int))["loss_train"].mean()
    left.plot(
        binned.index, binned.values, marker="o", ms=3, color="tab:blue", label="train"
    )

    left.plot(
        val["_step"] / STEPS_PER_EPOCH,
        val["loss_val"],
        marker="s",
        ms=3,
        color="tab:orange",
        label="dev",
    )
    left.set_yscale("log")
    left.set_xlabel("epoch")
    left.set_ylabel("cross entropy")
    left.set_title("Loss")
    left.legend()
    left.grid(alpha=0.3)

    right.plot(
        val["_step"] / STEPS_PER_EPOCH,
        val["EER_val"],
        marker="o",
        ms=3,
        color="tab:red",
    )
    right.set_xlabel("epoch")
    right.set_ylabel("EER, %")
    right.set_title("Development EER")
    right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def plot_per_attack(scores, labels, attacks, out):
    keys = list(labels)
    score = np.array([scores[k] for k in keys])
    label = np.array([labels[k] for k in keys])
    attack = np.array([attacks[k] for k in keys])
    bonafide = score[label == 1]

    values = []
    for name in ATTACKS:
        eer, _ = compute_eer(bonafide, score[attack == name])
        values.append(eer * 100)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["tab:red" if v > 5 else "tab:blue" for v in values]
    ax.bar(ATTACKS, values, color=colors)
    for x, v in enumerate(values):
        ax.text(x, v + 0.3, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylabel("EER, %")
    ax.set_xlabel("attack")
    ax.set_title("Equal error rate per attack, evaluation set")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)
    return dict(zip(ATTACKS, values))


def read_log_curve(path):
    """
    Per-epoch metrics from a trainer log. Used for the runs that report the
    evaluation EER every epoch, where the tracker export is not at hand.

    Returns:
        rows (list[dict]): one entry per epoch with the metrics it logged.
    """
    rows, current = [], {}
    pattern = re.compile(r"(epoch|val_EER|test_EER)\s+:\s+([\d.eE+-]+)")
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        found = pattern.search(line)
        if not found:
            continue
        key, value = found.group(1), float(found.group(2))
        if key == "epoch":
            if current:
                rows.append(current)
            current = {"epoch": int(value)}
        else:
            current[key] = value
    if current:
        rows.append(current)
    return rows


def plot_eval_curve(path, out):
    """
    Evaluation EER after every epoch, from the second run of the same
    configuration. Read from the trainer log rather than the tracker export,
    because this run was done locally.
    """
    rows = read_log_curve(path)

    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, [r["test_EER"] for r in rows], marker="o", ms=4, label="evaluation")
    ax.plot(epochs, [r["val_EER"] for r in rows], marker="s", ms=4, label="development")
    ax.set_xlabel("epoch")
    ax.set_ylabel("EER, %")
    ax.set_title("EER after every epoch, second run")
    ax.set_xticks(range(2, 21, 2))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)
    return rows


def plot_seed_spread(runs, out):
    """
    Evaluation EER per epoch for the LFCC runs: three seeds of the same
    configuration, plus the one trained with the weighted loss.

    The three seeds measure how much of a difference the seed alone makes,
    which is what the weighted run has to be read against.

    Args:
        runs (dict[str, Path]): label of the run to its trainer log.
        out (Path): where to write the figure.
    Returns:
        summary (dict): label to its per-epoch evaluation EER.
    """
    seed_color = "tab:blue"
    fig, ax = plt.subplots(figsize=(8, 4.5))

    summary = {}
    for label, path in runs.items():
        if Path(path).suffix == ".csv":
            frame = pd.read_csv(path)
            frame = frame[frame["EER_test"].notna()].sort_values("epoch_test")
            epochs = frame["epoch_test"].astype(int).tolist()
            values = frame["EER_test"].tolist()
        else:
            rows = [r for r in read_log_curve(path) if "test_EER" in r]
            epochs = [r["epoch"] for r in rows]
            values = [r["test_EER"] for r in rows]
        if not values:
            print(f"  no evaluation EER in {path}, skipped")
            continue
        summary[label] = values

        weighted = "weighted" in label
        ax.plot(
            epochs,
            values,
            marker="s" if weighted else "o",
            ms=4,
            lw=2 if weighted else 1.2,
            color="tab:red" if weighted else seed_color,
            alpha=1.0 if weighted else 0.55,
            label=label,
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("EER, %")
    ax.set_title("Evaluation EER per epoch, LFCC front end")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)

    for label, values in summary.items():
        best = min(values)
        print(
            f"  {label:16s} last {values[-1]:6.2f}   best {best:6.2f} "
            f"at epoch {values.index(best) + 1}"
        )
    seeds = [v[-1] for k, v in summary.items() if "weighted" not in k]
    if len(seeds) > 1:
        print(
            f"  spread over {len(seeds)} seeds at the last epoch: "
            f"{min(seeds):.2f} to {max(seeds):.2f}"
        )
    return summary


def main():


    saved = ROOT / "saved"
    lfcc_runs = {}
    for name in ("lfcc_seed1", "lfcc_seed2", "lfcc_seed3", "lfcc_weighted"):
        for candidate in (REPORT_DIR / f"wandb_{name}.csv", saved / name / "info.log"):
            if candidate.exists():
                lfcc_runs[name] = candidate
                break
    if lfcc_runs:
        plot_seed_spread(lfcc_runs, REPORT_DIR / "fig_seed_spread.png")
    else:
        print("no LFCC run logs under saved/, skipping the seed spread figure")

    local_log = ROOT / "saved" / "5-fft-zero-pad-eval-tracked" / "info.log"
    if local_log.exists():
        rows = plot_eval_curve(local_log, REPORT_DIR / "fig_eval_curve.png")
        best = min(rows, key=lambda r: r["test_EER"])
        print(
            f"  second run: best {best['test_EER']:.2f}% at epoch {best['epoch']}, "
            f"last {rows[-1]['test_EER']:.2f}%"
        )

    if not HISTORY.exists():
        raise SystemExit(f"{HISTORY} is missing, run export_wandb.py first")

    history = pd.read_csv(HISTORY)
    plot_training(history, REPORT_DIR / "fig_training.png")

    labels, attacks = read_protocol(PROTOCOL)
    scores = read_scores(PREDICTIONS)
    per_attack = plot_per_attack(
        scores, labels, attacks, REPORT_DIR / "fig_per_attack.png"
    )

    keys = list(labels)
    score = np.array([scores[k] for k in keys])
    label = np.array([labels[k] for k in keys])
    overall, _ = compute_eer(score[label == 1], score[label == 0])

    print(f"\nepochs in the history: {int(history['EER_val'].notna().sum())}")
    print(f"overall eval EER {overall * 100:.4f}%")
    print("worst attacks:", sorted(per_attack.items(), key=lambda kv: -kv[1])[:3])


if __name__ == "__main__":
    main()
