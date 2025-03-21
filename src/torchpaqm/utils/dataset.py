import torch
import math
import torchaudio
from typing import List
from torch.utils.data import Dataset
from ..utils import pad_signal


class PAQMDataset(Dataset):
    def __init__(
        self,
        inputs: List[str],
        references: List[str],
        fs: float = 44100,
        frame_duration: float = 0.04,
        overlap: float = 0.5,
    ):
        if len(inputs) != len(references):
            raise ValueError(
                f"Input path list has {len(inputs)} entries, but refs path list has {len(references)} elements"
            )
        self.inputs = inputs
        self.references = references
        self.fs = fs
        self.frame_size = math.floor(fs * frame_duration)
        self.overlap_size = math.floor(overlap * self.frame_size)

    def __len__(self):
        return len(self.inputs)

    def _validate_index(self, idx):
        input, fs = torchaudio.load(self.inputs[idx])
        if fs != self.fs:
            raise ValueError(
                f"File {self.inputs[idx]} has sample rate {fs}. Should be {self.fs}"
            )
        ref, fs = torchaudio.load(self.references[idx])
        if fs != self.fs:
            raise ValueError(
                f"File {self.references[idx]} has sample rate {fs}. Should be {self.fs}"
            )
        if input.shape[0] != ref.shape[0]:
            raise ValueError(
                f"{self.inputs[idx]} and {self.references[idx]} have different number of channels"
            )
        if input.shape[1] != ref.shape[1]:
            raise ValueError(
                f"{self.inputs[idx]} and {self.references[idx]} have different number of samples"
            )
        return input, ref

    def __getitem__(self, idx):
        input, ref = self._validate_index(idx)
        hop = self.frame_size - self.overlap_size
        padded_signals = pad_signal(torch.stack((input, ref)), self.frame_size, hop)
        return padded_signals[0, ...], padded_signals[1, ...]
