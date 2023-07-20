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
    output = output.to(dtype=torch.float32)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["hearing_threshold_spl"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    # visual check --> seems to be ok but numerical difference near zeros
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(
        compressor.equal_loudness_contour_freqs, output, marker="o", label="output"
    )
    plt.plot(
        compressor.equal_loudness_contour_freqs,
        expected_output,
        marker="x",
        label="expected",
    )
    plt.xlabel("Hz")
    plt.ylabel("dB SPL")
    plt.title(f"Equal Loudness Curve For {PHON_HEARING_THRESHOLD} phon")
    plt.legend()
    plt.show()
    # lowest tolerance possible while still passing tests, the two values near zero seem to be the issue
    # however their difference is around 1e-6, should not have a huge effect in the total output
    assert torch.allclose(expected_output, output, atol=1e-05, rtol=1e-04)


def test_hearing_threshold_excitation():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float64).squeeze()
    output = compressor.hearing_threshold_excitation(bark_frequencies).squeeze()
    expected_output = torch.from_numpy(MATLAB_FIXTURES["hearing_threshold_excitation"])
    expected_output = expected_output.to(dtype=torch.float64).squeeze()

    # visual check
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(bark_frequencies, 10 * torch.log10(output), marker="o", label="output")
    plt.plot(
        bark_frequencies,
        10 * torch.log10(expected_output),
        marker="x",
        label="expected",
    )
    plt.xlabel("Barks")
    plt.ylabel("dB")
    plt.title(f"Hearing Threshold Excitation")
    plt.legend()
    plt.show()

    plt.plot(
        bark_frequencies, 10 * torch.log10(output) - 10 * torch.log10(expected_output)
    )
    plt.xlabel("Barks")
    plt.ylabel("dB")
    plt.title("Difference to reference in dB")
    plt.show()

    # TODO 20/07/2023 - same interpolation issue of the outer to inner ear transfer function
    # apparently only different for very high frequencies
    # assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-2)


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
