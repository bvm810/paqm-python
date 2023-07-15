import scipy
import torch
from paqm.utils import hertz_to_bark, bark_to_hertz
from typing import List, Tuple


ENERGY_TIME_DECAY_CONSTANT = [
    (25.4, 300 * 1e-3),
    (63.7, 100 * 1e-3),
    (184.5, 30 * 1e-3),
    (562.2, 10 * 1e-3),
    (3272.8, 3 * 1e-3),
]

FREQ_SPREADING_SLOPES = (31, 22)


class Masker:
    def __init__(
        self,
        time_compression: float,
        freq_compression: float,
        tau_curve: List[Tuple[float, float]] = ENERGY_TIME_DECAY_CONSTANT,
        freq_spreading_slopes: Tuple[float, float] = FREQ_SPREADING_SLOPES,
    ) -> None:
        self.time_compression = time_compression
        self.freq_compression = freq_compression
        self._tau_curve_frequencies = torch.Tensor([pair[0] for pair in tau_curve])
        self._tau_curve_values = torch.Tensor([pair[1] for pair in tau_curve])
        self._freq_spreading_slopes = freq_spreading_slopes

    def _get_time_decay(
        self, frequencies: torch.Tensor, overlap: float
    ) -> torch.Tensor:
        # returns time decay multipliers per bark bin, in the same order as frequencies
        tau_frequencies = self.tau_curve_frequencies_in_hertz
        tau_values = self.tau_curve_values
        # interpolation needs to be done in hertz because bark conversion is non linear
        frequencies = bark_to_hertz(frequencies)
        interpolator = scipy.interpolate.PchipInterpolator(tau_frequencies, tau_values)
        time_decay_constant = interpolator(frequencies)
        time_decay_constant[frequencies > tau_frequencies[-1]] = tau_values[-1]
        time_decay = torch.exp(-overlap / torch.from_numpy(time_decay_constant))
        return time_decay

    def time_domain_spreading(
        self, spectrum: torch.Tensor, bark_axis: torch.Tensor, overlap: float
    ) -> torch.Tensor:
        time_decay = self._get_time_decay(bark_axis, overlap)
        spread_spectrum = torch.zeros_like(spectrum)
        spread_spectrum[:, 0] = spectrum[:, 0]
        for i in range(1, spectrum.shape[-1]):
            previous = (spread_spectrum[:, i - 1] * time_decay) ** self.time_compression
            current = spectrum[:, i] ** self.time_compression
            spread_spectrum[:, i] = (previous + current) ** (1 / self.time_compression)
        return spread_spectrum

    def frequency_domain_spreading(
        self, signal: torch.Tensor, bark_axis: torch.Tensor
    ) -> torch.Tensor:
        # TODO freq domain spreading
        pass

    @property
    def tau_curve_values(self) -> torch.Tensor:
        return self._tau_curve_values

    @property
    def tau_curve_frequencies_in_hertz(self) -> torch.Tensor:
        return self._tau_curve_frequencies

    @property
    def tau_curve_frequencies_in_barks(self) -> torch.Tensor:
        return hertz_to_bark(self._tau_curve_frequencies)

    @property
    def ascending_slope(self) -> float:
        return self._freq_spreading_slopes[0]

    @property
    def descending_slope(self) -> float:
        return self._freq_spreading_slopes[1]
