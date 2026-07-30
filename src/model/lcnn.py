import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    """
    Max-Feature-Map activation, the MFM 2/1 operation of LightCNN
    (https://arxiv.org/abs/1511.02683). Splits the input in two halves along
    dim and takes the element-wise maximum, so dim halves. No parameters.
    """

    def __init__(self, dim=1):
        """
        Args:
            dim (int): dimension to split, channels for conv layers and
                features for fully-connected ones.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x):
        first, second = x.chunk(2, dim=self.dim)
        return torch.max(first, second)


class LCNN(nn.Module):
    """
    Light CNN of the STC anti-spoofing system, following its Table 1
    (https://arxiv.org/abs/1904.05576). The comments name the layers as that
    table does. Convolutions are padded to keep the resolution, which the
    output sizes in the table imply.

    The layer sizes are reproduced as printed; two of the paper's parameter
    counts contradict them and are treated as misprints, see the report. The
    dropout is not in the table, the homework asks for it.
    """

    def __init__(self, n_class=2, input_freq=863, input_time=600, dropout=0.75):
        """
        Args:
            n_class (int): number of classes, two for bonafide vs spoof.
            input_freq (int): number of frequency bins of the front-end.
            input_time (int): number of frames of the front-end.
            dropout (float): dropout rate before the final batch norm.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            MaxFeatureMap(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            MaxFeatureMap(),
            nn.MaxPool2d(2),
        )

        n_flatten = 32 * self._pooled_size(input_freq) * self._pooled_size(input_time)

        self.classifier = nn.Sequential(
            nn.Linear(n_flatten, 160),
            MaxFeatureMap(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, n_class),
        )

        self._init_weights()

    @staticmethod
    def _pooled_size(size):
        """
        Size of a dimension after the four 2x2 poolings of the network.
        """
        for _ in range(4):
            size = (size - 2) // 2 + 1
        return size

    def _init_weights(self):
        """
        Kaiming initialization, as in the paper.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data_object, **batch):
        """
        Args:
            data_object (Tensor): spectrograms of shape (B, 1, freq, time).
        Returns:
            output (dict): output dict containing logits.
        """
        x = self.features(data_object)
        x = x.flatten(start_dim=1)
        return {"logits": self.classifier(x)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
