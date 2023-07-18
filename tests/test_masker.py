import os
import scipy
import torch
import matplotlib.pyplot as plt
from paqm.masker import Masker
from paqm.utils import bark_to_hertz

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


def test_batch_time_spreading():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    input = torch.from_numpy(MATLAB_FIXTURES["transfered_bark_spectrum"])
    input = input.to(dtype=torch.float32)
    input = input.view(1, 1, input.shape[0], -1)
    channels = torch.cat((input, input), dim=1)
    batch = torch.cat([channels] * 5, dim=0)
    output = masker.time_domain_spreading(batch, bark_frequencies, overlap)
    assert output.shape == (5, 2, 128, 24)


def test_ascending_freq_spreading_slopes():
    spectrum = torch.ones((3, 4))
    output = masker._ascending_slopes(spectrum)
    assert output.shape == (3, 4, 1)
    expected_output = 31 * spectrum.unsqueeze(-1)
    assert torch.allclose(expected_output, output)


def test_descending_freq_spreading_slopes():
    spectrum = torch.ones(3, 4)
    axis = torch.Tensor([4, 5, 6])
    output = masker._descending_slopes(axis, spectrum)
    assert output.shape == (3, 4, 1)
    expected_frame = 22 + 230 / bark_to_hertz(axis) - 0.2
    expected_frame = expected_frame.unsqueeze(1)
    assert expected_frame.shape == (3, 1)
    expected_output = torch.cat([expected_frame] * 4, dim=1)
    expected_output = expected_output.unsqueeze(-1)
    assert torch.allclose(expected_output, output)


def test_freq_spreading_masks():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    input = torch.from_numpy(MATLAB_FIXTURES["time_spread_spectrum"])
    input = input.to(dtype=torch.float32)
    db_input = 10 * torch.log10(input)
    output = masker._get_freq_spreading_masks(bark_frequencies, db_input)

    # visual check, could not think of automated way to test
    fig, axs = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
    for ax, bin in zip(axs.flatten(), [38, 39, 40, 41]):
        print(type(ax))
        ax.plot(
            bark_frequencies,
            output[bin, 12, :],
            marker="o",
        )
        ax.set_xlabel("Barks")
        ax.set_ylabel("dB")
        ax.set_title(f"12th Frame, {bin}th Bin Mask")
    plt.show()


def test_freq_spreading():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    input = torch.from_numpy(MATLAB_FIXTURES["time_spread_spectrum"])
    input = input.to(dtype=torch.float32)
    output = masker.frequency_domain_spreading(input, bark_frequencies)
    assert output.shape == (128, 24)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["excitation"])
    expected_output = expected_output.to(dtype=torch.float32)

    # visual check
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)
    obtained = ax1.imshow(output, aspect="auto", origin="lower")
    ax1.set_title("Frequency Spread Spectrum")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Bark bins")
    fig.colorbar(obtained, ax=ax1)
    expected = ax2.imshow(expected_output, aspect="auto", origin="lower")
    ax2.set_title("Expected Frequency Spread Spectrum")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Bark bins")
    fig.colorbar(expected, ax=ax2)
    plt.show()

    assert torch.allclose(expected_output, output, atol=1e-06, rtol=1e-02)


def test_batch_frequency_spreading():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32).squeeze()
    input = torch.from_numpy(MATLAB_FIXTURES["time_spread_spectrum"])
    input = input.to(dtype=torch.float32)
    input = input.view(1, 1, input.shape[0], -1)
    channels = torch.cat((input, input), dim=1)
    batch = torch.cat([channels] * 5, dim=0)
    output = masker.frequency_domain_spreading(batch, bark_frequencies)
    assert output.shape == (5, 2, 128, 24)
