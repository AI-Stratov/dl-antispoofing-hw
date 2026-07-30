import numpy as np
import torch

from src.datasets.asvspoof import BONAFIDE, SPOOF


def compute_det_curve(target_scores, nontarget_scores):
    """
    Taken from calculate_eer.py of the homework repository, which the task
    description asks to use for the metric.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )


    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]


    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )


    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))

    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )

    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    """
    Returns equal error rate (EER) and the corresponding threshold.

    Taken from calculate_eer.py of the homework repository.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def bonafide_score(logits):
    """
    Difference of the two logits: the log likelihood ratio the challenge
    asks for, larger meaning more bonafide. Not the log probability of the
    bonafide class, which orders the trials the same way in exact arithmetic
    but underflows to 0.0 in float32 and ties thousands of them.

    Args:
        logits (Tensor): model outputs of shape (B, 2).
    Returns:
        scores (Tensor): detection scores of shape (B,).
    """
    return logits[:, BONAFIDE] - logits[:, SPOOF]


class EER:
    """
    Equal Error Rate over a whole partition.

    Not a BaseMetric: its threshold is chosen over all the scores at once,
    so the EER of a partition is not the average of the EERs of its batches.
    The trainer collects the scores and calls result() after the epoch.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = []
        self.labels = []

    def update(self, logits, labels):
        """
        Store the scores of one batch.

        Args:
            logits (Tensor): model outputs of shape (B, 2).
            labels (Tensor): ground-truth labels of shape (B,).
        """
        self.scores.append(bonafide_score(logits).detach().cpu())
        self.labels.append(labels.detach().cpu())

    def result(self):
        """
        Returns:
            eer (float): equal error rate in percent, on the 0-100 scale
                used by the grading script.
        """
        scores = torch.cat(self.scores).numpy()
        labels = torch.cat(self.labels).numpy()

        bonafide = scores[labels == BONAFIDE]
        spoof = scores[labels == SPOOF]
        if len(bonafide) == 0 or len(spoof) == 0:

            return float("nan")

        eer, _ = compute_eer(bonafide, spoof)


        return float(eer) * 100
