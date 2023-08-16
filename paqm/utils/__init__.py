import torch
import math
from typing import List, Tuple
import torch.nn.functional as F


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
    blocks = signal.unfold(-1, block_size, hop)
    windowed_blocks = blocks * window.unsqueeze(0)
    return windowed_blocks  # (batch, channels, blocks, block_size)


def collate(data: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    inputs, refs = zip(*data)
    max_len = max(t.shape[-1] for t in inputs)
    collated_in = torch.zeros((len(data), inputs[0].shape[0], max_len))
    collated_refs = torch.zeros(collated_in.shape)
    for i in range(len(data)):
        pad_len = max_len - inputs[i].shape[-1]
        collated_in[i, ...] = F.pad(inputs[i], (0, pad_len), "constant", float("nan"))
        collated_refs[i, ...] = F.pad(refs[i], (0, pad_len), "constant", float("nan"))
    return collated_in, collated_refs
