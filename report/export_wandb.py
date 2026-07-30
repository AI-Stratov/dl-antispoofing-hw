"""
Downloads the history of a run into a csv, for the report plots.

The export button in the runs table gives one row per run with the final
values only, which is not what a curve needs. The api history() method is
no good either: it samples the history down to about 500 rows, which drops
most of the per-epoch metrics. scan_history returns every logged row.

Usage:

    python3 report/export_wandb.py aistratov/asvspoof-lcnn 4-fft-zero-pad-final
"""

import sys

import pandas as pd

import wandb


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(1)

    project, run_name = sys.argv[1], sys.argv[2]

    api = wandb.Api()
    runs = [r for r in api.runs(project) if r.name == run_name]
    if not runs:
        available = ", ".join(r.name for r in api.runs(project))
        raise SystemExit(f"no run named {run_name}. available: {available}")

    run = runs[0]
    history = pd.DataFrame(list(run.scan_history()))

    out = f"report/wandb_{run_name}.csv"
    history.to_csv(out, index=False)

    print(f"{len(history)} rows -> {out}")
    print("columns:", ", ".join(c for c in history.columns if not c.startswith("_")))
    if "EER_val" in history:
        print("epochs with an EER:", int(history["EER_val"].notna().sum()))


if __name__ == "__main__":
    main()
