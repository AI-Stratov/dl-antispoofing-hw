import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy over the two classes, rather than the angular margin
    softmax of STC: the comparative study measures both on this database and
    finds the gain smaller than the spread between seeds. See the report.
    """

    def __init__(self, weight=None):
        """
        Args:
            weight (list[float] | None): per-class weight, indexed by label,
                so [spoof, bonafide]. The partition holds about nine spoof
                trials per bonafide one; None leaves them unweighted, as in
                both papers.
        """
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output predictions of shape (B, 2).
            labels (Tensor): ground-truth labels of shape (B,).
        Returns:
            losses (dict): dict containing the calculated loss.
        """
        return {"loss": self.loss(logits, labels)}
