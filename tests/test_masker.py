import os
import scipy
import torch
import matplotlib.pyplot as plt
from paqm.masker import Masker

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
MATLAB_FIXTURES = scipy.io.loadmat(os.path.join(FIXTURES_PATH, "masker.mat"))
masker = Masker(time_compression=0.6, freq_compression=0.8)
overlap = 0.02


def test_time_decay_values():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    output = masker._get_time_decay(bark_frequencies, overlap)
    assert output.shape == (128,)
    output = output.to(dtype=torch.float32)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["energy_time_decay"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    assert torch.allclose(expected_output, output)


def test_time_spreading():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    input = torch.from_numpy(MATLAB_FIXTURES["transfered_bark_spectrum"])
    input = input.to(dtype=torch.float32)
    output = masker.time_domain_spreading(input, bark_frequencies, overlap)
    assert output.shape == (128, 24)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["time_spread_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    assert torch.allclose(expected_output, output, atol=1e-06, rtol=1e-02)
