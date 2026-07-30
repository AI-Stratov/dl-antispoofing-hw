import torch
from torch import nn


class LogPowerSpectrogram(nn.Module):
    """
    Log power spectrum, the front-end of the LCNN.

    The defaults are the FFT system of STC (https://arxiv.org/abs/1904.05576):
    a 1724-point transform and a Blackman window, giving 863 bins at a step
    of 130 samples, or 0.0081 s at 16 kHz. Not normalized, following the
    comparative study.
    """

    def __init__(self, n_fft=1724, hop_length=130, eps=1e-8):
        """
        Args:
            n_fft (int): size of the FFT.
            hop_length (int): step between two consecutive frames, in samples.
            eps (float): constant added before taking the logarithm.
        """
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps


        self.register_buffer("window", torch.blackman_window(n_fft))

    def forward(self, x):
        """
        Args:
            x (Tensor): batch of waveforms of shape (B, n_samples).
        Returns:
            x (Tensor): batch of spectrograms of shape (B, 1, n_freq, n_frames).
        """
        spectrum = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spectrum.real**2 + spectrum.imag**2


        return torch.log(power + self.eps).unsqueeze(1)
