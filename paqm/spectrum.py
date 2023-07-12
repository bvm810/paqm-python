import torch
import math
from .utils import hertz_to_bark, frame_signal, bark_to_hertz


class SpectrumAnalyzer:
    def __init__(
        self,
        fs: float,
        frame_duration: float = 0.04,
        window: str = "hann",
        overlap: float = 0.5,
        nfft: int = 2048,
        bark_binwidth: float = 0.2,
    ) -> None:
        self.fs = fs
        self.frame_size = math.floor(fs * frame_duration)
        self.overlap_size = math.floor(overlap * self.frame_size)
        self.window = self._get_window(window, self.frame_size)
        self.nfft = nfft
        self.bark_binwidth = bark_binwidth

    def _get_window(self, window_name: str, size: int) -> torch.Tensor:
        if window_name == "hann":
            return torch.hann_window(size, periodic=False)
        if window_name == "rect":
            return torch.ones(size)
        raise NotImplementedError

    def stft(self, signal: torch.Tensor) -> torch.Tensor:
        # output stft with size (batch, n_channels, nfft // 2 + 1, n_frames)
        hop = self.frame_size - self.overlap_size
        frames = frame_signal(signal, self.window, hop)
        stft = torch.fft.rfft(input=frames, n=self.nfft, dim=-1)
        return torch.movedim(stft, -2, -1)

    def power_spectrum(self, signal: torch.Tensor) -> torch.Tensor:
        spectrum = self.stft(signal)
        return torch.abs(spectrum) ** 2

    def _get_bark_filterbank(self) -> torch.Tensor:
        freqs = self.freq_axis_in_hertz
        bark_axis = self.freq_axis_in_barks
        lower_limits = bark_to_hertz(bark_axis - self.bark_binwidth / 2)
        upper_limits = bark_to_hertz(bark_axis + self.bark_binwidth / 2)
        upper_limits[-1] = self.fs / 2 + 0.1  # extend last bark bin to fs / 2
        filterbank = torch.logical_and(
            input=(freqs >= lower_limits.unsqueeze(1)),
            other=(freqs < upper_limits.unsqueeze(1)),
        ).to(dtype=torch.int8)
        filterbank = filterbank / torch.sum(filterbank, dim=1).unsqueeze(1)
        delta_freqs = (upper_limits - lower_limits).unsqueeze(1)
        weights = delta_freqs / self.bark_binwidth
        filterbank = filterbank * weights
        return filterbank

    def bark_spectrum(self, signal: torch.Tensor) -> torch.Tensor:
        # output bark spectrum with shape (batch, channels, barks, frames)
        filterbank = self._get_bark_filterbank()
        filterbank = filterbank.view(1, 1, filterbank.shape[0], -1)
        power_spectrum = self.power_spectrum(signal)
        return filterbank.to(power_spectrum.dtype) @ power_spectrum

    @property
    def freq_axis_in_hertz(self):
        bins = torch.arange(self.nfft // 2 + 1)
        return bins * (self.fs / self.nfft)

    @property
    def freq_axis_in_barks(self):
        last_bin = hertz_to_bark(self.freq_axis_in_hertz[-1]) - self.bark_binwidth
        barks = torch.arange(start=0, end=last_bin, step=self.bark_binwidth)
        return barks + self.bark_binwidth / 2

    @property
    def overlap_duration(self):
        return self.overlap_size / self.fs
