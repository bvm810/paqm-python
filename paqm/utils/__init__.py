import torch
import math
from typing import List


def bark_to_hertz(bark_freqs: torch.Tensor) -> torch.Tensor:
    return 600 * torch.sinh(bark_freqs / 6)


def hertz_to_bark(hertz_freqs: torch.Tensor) -> torch.Tensor:
    return 6 * torch.asinh(hertz_freqs / 600)


def pad_signal(signal: torch.Tensor, frame_size: int, hop: int) -> torch.Tensor:
    remainder = (signal.shape[-1] - frame_size) % hop
    pad_size = hop - remainder if remainder != 0 else 0
    pad = torch.zeros((signal.shape[0], signal.shape[1], math.ceil(pad_size / 2)))
    return torch.cat((pad, signal, pad), dim=-1)  # (batch, channels, samples)


def frame_signal(signal: torch.Tensor, window: torch.Tensor, hop: int) -> torch.Tensor:
    block_size = window.shape[-1]
    padded_signal = pad_signal(signal, block_size, hop)
    blocks = padded_signal.unfold(-1, block_size, hop)
    windowed_blocks = blocks * window.unsqueeze(0)
    return windowed_blocks  # (batch, channels, blocks, block_size)


# TODO 31/07/2023 -- In order to be able to batch process audio files, they should be first padded individually
# with zeros, and then padded with NaN before collating them as a batch. The simplest way to adapt the code would be
# to remove the pad_signal function call from frame_signal, and assume it already receives a batch correctly padded with
# zeros and NaNs. Insert a collate_fn here to make padding easy for the user.
def collate(signals: List[torch.Tensor]) -> torch.Tensor:
    pass
