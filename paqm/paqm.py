import torch
from paqm.spectrum import SpectrumAnalyzer
from paqm.transfer import OuterToInnerTransfer
from paqm.masker import Masker
from paqm.loudness import LoudnessCompressor


SCALING_LOWER_LIMITS = torch.Tensor([0, 2, 22])
SCALING_UPPER_LIMITS = torch.Tensor([2, 22, float("inf")])
MOS_CONVERSION_COEFS = torch.Tensor([3.0495, 3.4806, 0.4769])


class PAQM:
    def __init__(
        self,
        audio: torch.Tensor,
        reference: torch.Tensor,
        analyzer: SpectrumAnalyzer = SpectrumAnalyzer(fs=44100),
        transfer: OuterToInnerTransfer = OuterToInnerTransfer(),
        masker: Masker = Masker(),
        compressor: LoudnessCompressor = LoudnessCompressor(),
    ) -> None:
        if audio.shape != reference.shape:
            raise ValueError("Input and reference tensors must have the same shape")
        self.audio = audio
        self.reference = reference
        self.analyzer = analyzer
        self.transfer = transfer
        self.masker = masker
        self.compressor = compressor
        self._scores = self._get_scores(self.audio, self.reference)

    def _get_internal_representation(self, audio: torch.Tensor) -> torch.Tensor:
        bark_stft = self.analyzer.bark_spectrum(audio)
        bark_axis = self.analyzer.freq_axis_in_barks
        inner_ear_spectrum = self.transfer.transfer_signal_with_freqs(
            bark_stft, bark_axis
        )
        time_spread_spectrum = self.masker.time_domain_spreading(
            inner_ear_spectrum, bark_axis, self.analyzer.overlap_duration
        )
        freq_spread_spectrum = self.masker.frequency_domain_spreading(
            time_spread_spectrum, bark_axis
        )
        internal_representation = self.compressor.compress(
            freq_spread_spectrum, bark_axis, self.transfer
        )
        return internal_representation

    def _scaling(self, input: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        freqs = self.analyzer.freq_axis_in_barks
        intervals = torch.logical_and(
            input=(freqs > SCALING_LOWER_LIMITS.unsqueeze(1)),
            other=(freqs <= SCALING_UPPER_LIMITS.unsqueeze(1)),
        ).to(dtype=input.dtype)
        input_energy = intervals @ input
        reference_energy = intervals @ reference
        input_energy[input_energy == 0] = 1
        factors = reference_energy / input_energy
        intervals = intervals.view(-1, 1, 1, freqs.shape[-1], 1)
        factors = torch.movedim(factors, -2, 0).unsqueeze(-2)
        scaled_input = ((intervals * input) * factors).sum(dim=0)
        return scaled_input

    def _get_scores(self, audio: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        input_internal_repr = self._get_internal_representation(audio)
        ref_internal_repr = self._get_internal_representation(reference)
        scaled_input = self._scaling(input_internal_repr, ref_internal_repr)
        return torch.abs(scaled_input - ref_internal_repr)

    @property
    def full_scores(self):
        return self._scores  # (batch, channels, bark bins, frames)

    @property
    def frame_scores(self):
        return torch.sum(self._scores, dim=-2)  # (batch, channels, frames)

    @property
    def score(self):
        return torch.nanmean(self.frame_scores, dim=-1)  # (batch, channels)

    @property
    def mean_opinion_score(self):
        log_score = torch.log10(self.score)
        poly = torch.stack(
            (torch.ones(log_score.shape), log_score, log_score**2), dim=-1
        )
        mos = 0.999 + 4 / (1 + torch.exp(poly @ MOS_CONVERSION_COEFS))
        return mos
