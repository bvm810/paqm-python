import scipy
import torch
from .utils import hertz_to_bark, bark_to_hertz
from typing import List, Tuple


ENERGY_TIME_DECAY_CONSTANT = [
    (25.4, 300 * 1e-3),
    (63.7, 100 * 1e-3),
    (184.5, 30 * 1e-3),
    (562.2, 10 * 1e-3),
    (3272.8, 3 * 1e-3),
]

FREQ_SLOPES_CONSTANTS = (31, 22)


class Masker:
    def __init__(
        self,
        time_compression: float = 0.6,
        freq_compression: float = 0.8,
        tau_curve: List[Tuple[float, float]] = ENERGY_TIME_DECAY_CONSTANT,
        freq_spreading_constants: Tuple[float, float] = FREQ_SLOPES_CONSTANTS,
    ) -> None:
        self.time_compression = time_compression
        self.freq_compression = freq_compression
        self._tau_curve_frequencies = torch.Tensor([pair[0] for pair in tau_curve])
        self._tau_curve_values = torch.Tensor([pair[1] for pair in tau_curve])
        self._freq_spreading_constants = freq_spreading_constants

    def _get_time_decay(
        self, frequencies: torch.Tensor, overlap: float
    ) -> torch.Tensor:
        # returns time decay multipliers per bark bin, in the same order as frequencies
        tau_frequencies = self.tau_curve_frequencies_in_hertz
        tau_values = self.tau_curve_values
        # interpolation needs to be done in hertz because bark conversion is non linear
        frequencies = bark_to_hertz(frequencies)
        interpolator = scipy.interpolate.PchipInterpolator(tau_frequencies, tau_values)
        time_decay_constant = torch.from_numpy(
            interpolator(frequencies.detach().cpu().numpy())
        ).to(dtype=frequencies.dtype)
        time_decay_constant[frequencies > tau_frequencies[-1]] = tau_values[-1]
        time_decay = torch.exp(-overlap / time_decay_constant)
        return time_decay.to(frequencies.device)

    def time_domain_spreading(
        self, spectrum: torch.Tensor, bark_axis: torch.Tensor, overlap: float
    ) -> torch.Tensor:
        time_decay = self._get_time_decay(bark_axis, overlap)
        spread_spectrum = torch.zeros_like(spectrum)
        alpha = self.time_compression
        spread_spectrum[..., 0] = spectrum[..., 0]
        for i in range(1, spectrum.shape[-1]):
            previous = (spread_spectrum[..., i - 1] * time_decay) ** alpha
            current = spectrum[..., i] ** alpha
            spread_spectrum[..., i] = (previous + current) ** (1 / alpha)
        return spread_spectrum

    def _ascending_slopes(self, db_spectrum: torch.Tensor) -> torch.Tensor:
        slope = self._freq_spreading_constants[0]
        ascending_slopes = slope * torch.ones_like(db_spectrum)
        return ascending_slopes.unsqueeze(-1)

    def _descending_slopes(
        self, bark_axis: torch.Tensor, db_spectrum: torch.Tensor
    ) -> torch.Tensor:
        hertz_frequencies = bark_to_hertz(bark_axis).unsqueeze(1)
        base_slope = self._freq_spreading_constants[1]
        descending_slopes = base_slope + 230 / hertz_frequencies - 0.2 * db_spectrum
        return descending_slopes.unsqueeze(-1)

    def _get_freq_spreading_masks(
        self, bark_axis: torch.Tensor, db_spectrum: torch.Tensor
    ) -> torch.Tensor:
        freqs = bark_axis.view(1, 1, -1)
        center_freq = bark_axis.view(-1, 1, 1)
        up_slopes = self._ascending_slopes(db_spectrum)
        down_slopes = self._descending_slopes(bark_axis, db_spectrum)
        db_spectrum = db_spectrum.unsqueeze(-1)
        # ascending --> S1 * (f - fo + L/S1) for f < fo
        up_mask = up_slopes * (freqs - center_freq + db_spectrum / up_slopes)
        up_mask = (freqs < center_freq) * up_mask
        # descending --> -S2 * (f - fo - L/S2) for f >= fo
        down_mask = -down_slopes * (freqs - center_freq - db_spectrum / down_slopes)
        down_mask = (freqs >= center_freq) * down_mask
        return up_mask + down_mask

    def frequency_domain_spreading(
        self, power_spectrum: torch.Tensor, bark_axis: torch.Tensor
    ) -> torch.Tensor:
        db_spectrum = 10 * torch.log10(power_spectrum)
        masks = self._get_freq_spreading_masks(bark_axis, db_spectrum)
        masks = torch.pow(10, ((masks / 10) * (self.freq_compression / 2)))
        excitation = torch.sum(masks, dim=-3)
        excitation = excitation ** (2 / self.freq_compression)
        return excitation.movedim(-1, -2)

    @property
    def tau_curve_values(self) -> torch.Tensor:
        return self._tau_curve_values

    @property
    def tau_curve_frequencies_in_hertz(self) -> torch.Tensor:
        return self._tau_curve_frequencies

    @property
    def tau_curve_frequencies_in_barks(self) -> torch.Tensor:
        return hertz_to_bark(self._tau_curve_frequencies)
