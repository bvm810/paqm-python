import torch
import math


def bark_to_hertz(bark_freqs: torch.Tensor) -> torch.Tensor:
    return 600 * torch.sinh(bark_freqs / 6)


def hertz_to_bark(hertz_freqs: torch.Tensor) -> torch.Tensor:
    return 6 * torch.asinh(hertz_freqs / 600)


def pad_signal(signal: torch.Tensor, frame_size: int, hop: int) -> torch.Tensor:
    pad_size = (signal.shape[-1] - frame_size) % hop
    pad = torch.zeros((math.ceil(pad_size / 2)))
    return torch.cat((pad, signal, pad), dim=-1)  # (batch, channels, samples)


def frame_signal(signal: torch.Tensor, window: torch.Tensor, hop: int) -> torch.Tensor:
    block_size = window.shape[-1]
    padded_signal = pad_signal(signal, block_size, hop)
    blocks = padded_signal.unfold(-1, block_size, hop)
    windowed_blocks = blocks * window.unsqueeze(0)
    return windowed_blocks  # (batch, channels, blocks, block_size)
