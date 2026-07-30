"""
Writes a description and tags onto each run, so the project page says what
every run actually was without opening its config.

The early runs do not log the evaluation EER during training, it comes from
a separate inference pass, so the numbers are filled in by hand here.

Usage:

    python3 report/annotate_wandb.py aistratov/asvspoof-lcnn
"""

import sys

import wandb

RUNS = {
    "1-fft-repeat-pad": {
        "notes": (
            "FFT front-end, 863x600. Short trials padded by repeating the "
            "waveform. 20 epochs. Eval EER 12.43%, which is above the bar. "
            "Perfect on six attacks, near useless on four: memorised the "
            "training attacks."
        ),
        "tags": ["fft", "repeat-pad", "failed"],
    },
    "2-lfcc-repeat-pad": {
        "notes": (
            "LFCC front-end, 60x487, and 10 epochs instead of 20. Two "
            "variables changed at once, so the run says nothing about "
            "either. Eval EER 14.20%."
        ),
        "tags": ["lfcc", "repeat-pad", "failed"],
    },
    "3-lfcc-zero-pad": {
        "notes": (
            "Same as run 2 with zero padding instead of repetition, one "
            "variable. Eval EER 13.46%, apparently no change, but ten of "
            "the thirteen attacks improved by a factor of three or four. "
            "Excluding A17, A18 and A19 the EER is 3.28% against 9.42% for "
            "run 1. This is what pointed at the padding."
        ),
        "tags": ["lfcc", "zero-pad", "ablation"],
    },
    "4-fft-zero-pad-final": {
        "notes": (
            "FFT front-end with zero padding, 20 epochs. Single-variable "
            "change against run 1. Eval EER 4.56% at the last epoch, which "
            "is the submitted model. Selecting on the dev EER would have "
            "picked epoch 19 at 6.50% instead."
        ),
        "tags": ["fft", "zero-pad", "final", "submitted"],
    },
    "onebatchtest": {
        "notes": (
            "Sanity check before the long runs: trained and evaluated on the "
            "same eight utterances, four per class. The loss reaches 0.004, "
            "so the pipeline can at least fit what it is shown."
        ),
        "tags": ["sanity"],
    },
    "5-fft-zero-pad-eval-tracked": {
        "notes": (
            "Run 4 repeated on different hardware with the evaluation EER "
            "measured every epoch. Ended at 9.12% where run 4 ended at "
            "4.56%; the best epoch was the seventh at 6.36%, not the last. "
            "Note that measuring the evaluation set every epoch creates an "
            "extra dataloader, which shifts the random stream, so this is "
            "not the same trajectory as run 4 despite the same seed."
        ),
        "tags": ["fft", "zero-pad", "variance"],
    },
    "lfcc_seed1": {
        "notes": (
            "LFCC configuration, seed 1, evaluation EER every epoch. Ends at "
            "8.75%. One of three runs that differ only by the seed, used to "
            "measure the spread rather than guess it."
        ),
        "tags": ["lfcc", "zero-pad", "seed-spread"],
    },
    "lfcc_seed2": {
        "notes": (
            "Same configuration as lfcc_seed1 with seed 2. Ends at 9.41%, "
            "best epoch is the seventh at 8.28%."
        ),
        "tags": ["lfcc", "zero-pad", "seed-spread"],
    },
    "lfcc_seed3": {
        "notes": (
            "Same configuration as lfcc_seed1 with seed 3. Ends at 9.34%, "
            "after a spike to 45% at the fourth epoch."
        ),
        "tags": ["lfcc", "zero-pad", "seed-spread"],
    },
    "lfcc_weighted": {
        "notes": (
            "Class-weighted cross entropy, weights [0.557, 4.919] for "
            "[spoof, bonafide], the balanced inverse frequency of the "
            "training partition. Seed 1, everything else as in lfcc_seed1. "
            "Ends at 10.66%, which falls inside the range of the unweighted "
            "runs, so the comparison is inconclusive."
        ),
        "tags": ["lfcc", "class-weighting", "ablation"],
    },
}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)

    project = sys.argv[1]
    api = wandb.Api()

    for run in api.runs(project):
        entry = RUNS.get(run.name)
        if entry is None:
            print(f"skipped {run.name}, not in the table")
            continue

        run.notes = entry["notes"]
        run.tags = entry["tags"]
        run.update()
        print(f"annotated {run.name}")


if __name__ == "__main__":
    main()
