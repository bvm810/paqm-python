import scipy
import torch
from .utils import hertz_to_bark
from typing import List, Tuple


DEFAULT_TRANSFER_FUNCTION = [
    (100, 0),
    (200, 0),
    (500, 0),
    (1000, 0),
    (1720, 1.27),
    (2000, 2.68),
    (3150, 6.73),
    (5000, 3.04),
    (6400, -1.64),
    (10000, -7.5),
    (12000, -14.9),
]


class OuterToInnerTransfer:
    def __init__(
        self, transfer_function: List[Tuple[float, float]] = DEFAULT_TRANSFER_FUNCTION
    ) -> None:
        self._transfer_frequencies = torch.Tensor(
            [pair[0] for pair in transfer_function]
        )
        self._transfer_log_magnitudes = torch.Tensor(
            [-1 * pair[1] for pair in transfer_function]
        )

    def transfer_signal_with_freqs(
        self, signal_magnitudes: torch.Tensor, signal_frequencies: torch.Tensor
    ) -> torch.Tensor:
        gains = self.transfer_function_gains(signal_frequencies)
        gains = gains.unsqueeze(1).to(dtype=signal_magnitudes.dtype)
        gains = gains.to(signal_magnitudes.device)
        return gains * signal_magnitudes

    def transfer_signal_with_gains(
        self, signal_magnitudes: torch.Tensor, gains: torch.Tensor
    ) -> torch.Tensor:
        gains = gains.unsqueeze(1).to(dtype=signal_magnitudes.dtype)
        return gains * signal_magnitudes

    def transfer_function_gains(self, bark_freqs: torch.Tensor) -> torch.Tensor:
        interpolator = scipy.interpolate.CubicSpline(
            x=self.transfer_frequencies_in_barks, y=self.transfer_log_magnitudes
        )
        interpolated_log_mags = interpolator(bark_freqs.detach().cpu().numpy())
        gains = torch.pow(10, (torch.from_numpy(interpolated_log_mags) / 10))
        return gains.to(bark_freqs.device)

    @property
    def transfer_log_magnitudes(self) -> torch.Tensor:
        return self._transfer_log_magnitudes

    @property
    def transfer_frequencies_in_hertz(self) -> torch.Tensor:
        return self._transfer_frequencies

    @property
    def transfer_frequencies_in_barks(self) -> torch.Tensor:
        return hertz_to_bark(self._transfer_frequencies)
