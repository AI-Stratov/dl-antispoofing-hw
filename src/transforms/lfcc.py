import torch
import torchaudio
from torch import nn


class LFCC(nn.Module):
    """
    60-dimensional LFCC front-end of the ASVspoof2019 baseline, as described
    in the comparative study (https://arxiv.org/abs/2103.11326): 20 ms
    frames, 10 ms shift, 20 linear triangle filters, delta and delta-delta
    on top of the 20 static coefficients, the first of which is replaced
    with the log frame energy. No normalization.

    Used for quick experiments only: the 60 x 487 input trains sixteen times
    faster than the FFT front-end but scores worse. See the report.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=320,
        hop_length=160,
        n_filter=20,
        n_lfcc=20,
        eps=1e-8,
    ):
        """
        Args:
            sample_rate (int): sampling rate of the waveforms.
            n_fft (int): size of the FFT.
            win_length (int): frame length in samples, 20 ms at 16 kHz.
            hop_length (int): frame shift in samples, 10 ms at 16 kHz.
            n_filter (int): number of linear triangle filters.
            n_lfcc (int): number of cepstral coefficients per frame.
            eps (float): constant added before taking the logarithm.
        """
        super().__init__()

        self.eps = eps

        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
            },
        )

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): batch of waveforms of shape (B, n_samples).
        Returns:
            x (Tensor): batch of features of shape (B, 1, 60, n_frames).
        """
        features = self.lfcc(x)

        energy = torch.log(self.spectrogram(x).sum(dim=1) + self.eps)
        features[:, 0, :] = energy

        delta = torchaudio.functional.compute_deltas(features)
        delta2 = torchaudio.functional.compute_deltas(delta)

        return torch.cat([features, delta, delta2], dim=1).unsqueeze(1)
