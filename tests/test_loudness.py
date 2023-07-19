import os
import scipy
import torch
import matplotlib.pyplot as plt
from paqm.loudness import LoudnessCompressor, PHON_HEARING_THRESHOLD
from paqm.utils import bark_to_hertz

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
MATLAB_FIXTURES = scipy.io.loadmat(os.path.join(FIXTURES_PATH, "cmpLoudness.mat"))
compressor = LoudnessCompressor(schwell_factor=0.5, compression_level=0.04)


def test_equal_loudness_contour():
    output = compressor.equal_loudness_contour(PHON_HEARING_THRESHOLD)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["hearing_threshold_spl"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    assert torch.allclose(output, expected_output)


def test_hearing_threshold_excitation():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    output = compressor.hearing_threshold_excitation(bark_frequencies)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["hearing_threshold_excitation"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    assert torch.allclose(output, expected_output)


def test_internal_representation():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    input = torch.from_numpy(MATLAB_FIXTURES["excitation"])
    input = input.to(dtype=torch.float32)
    output = compressor.internal_representation(input, bark_frequencies)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["compressed_loudness"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    assert torch.allclose(expected_output, output, atol=1e-06, rtol=1e-02)


def test_batch_internal_representation():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    input = torch.from_numpy(MATLAB_FIXTURES["excitation"])
    input = input.to(dtype=torch.float32)
    input = input.view(1, 1, input.shape[0], -1)
    channels = torch.cat((input, input), dim=1)
    batch = torch.cat([channels] * 7, dim=0)
    output = compressor.internal_representation(batch, bark_frequencies)
    assert output.shape == (7, 2, 128, 24)
