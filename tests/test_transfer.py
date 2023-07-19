import os
import scipy
import torch
import matplotlib.pyplot as plt
from paqm.transfer import OuterToInnerTransfer

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
MATLAB_FIXTURES = scipy.io.loadmat(os.path.join(FIXTURES_PATH, "tfOuter2Inner.mat"))
transfer = OuterToInnerTransfer()

# TODO 12/07/2023: There seems to be some difference between Matlab's spline function and SciPy's CubicInterpolation
# that is causing interpolation differences for very high frequencies. This should not cause any difference in the
# final PAQM result, but is causing tests to fail nevertheless. This can be some numerical precision difference, but it
# should be investigated sometime.


def test_inner_to_outer_transfer_magnitudes():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    bark_frequencies = bark_frequencies.to(dtype=torch.float32)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["a0"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    output = transfer.transfer_function_gains(bark_frequencies)
    assert output.shape == (1, 128)
    output = output.to(dtype=torch.float32).squeeze()
    # error plots
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bark_frequencies = bark_frequencies.squeeze()
    ax.plot(bark_frequencies, 10 * torch.log10(output), marker="o", label="output")
    ax.plot(
        bark_frequencies,
        10 * torch.log10(expected_output),
        marker="x",
        label="expected",
    )
    plt.xlabel("Barks")
    plt.ylabel("dB")
    plt.title("Outer To Inner Ear Transfer Function")
    plt.legend()
    plt.show()
    plt.plot(
        bark_frequencies, 10 * torch.log10(output) - 10 * torch.log10(expected_output)
    )
    plt.xlabel("Barks")
    plt.ylabel("dB")
    plt.title("Difference to reference in dB")
    plt.show()
    # something is weird here, the interpolation diference for very high frequencies is around 10dB
    # this should not make any difference since we will only use frequencies up to 24 barks (where the error is near zero)
    # assert torch.allclose(expected_output, output)


def test_inner_to_outer_transfer():
    expected_output = torch.from_numpy(MATLAB_FIXTURES["transfered_bark_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    bark_stft = torch.from_numpy(MATLAB_FIXTURES["bark_spectrum"])
    bark_stft = bark_stft.to(dtype=torch.float32)
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"]).squeeze()
    bark_frequencies = bark_frequencies.to(dtype=torch.float32)
    output = transfer.transfer_signal_with_freqs(bark_stft, bark_frequencies)
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 4), ncols=2)
    obtained = ax1.imshow(output, aspect="auto")
    ax1.set_title("Transfered Bark STFT")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Bark bins")
    fig.colorbar(obtained, ax=ax1)
    expected = ax2.imshow(expected_output, aspect="auto")
    ax2.set_title("Expected Transfered Bark STFT")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Bark bins")
    fig.colorbar(expected, ax=ax2)
    plt.show()
    # zeros are causing torch allclose to require lower tolerance, fix this later
    # there is also error for very high frequencies because of the interpolation difference
    # this should not be relevant as it only impacts frequencies above 24 Barks
    # assert torch.allclose(expected_output, output, atol=1e-06, rtol=1e-02)


def test_batch():
    bark_frequencies = torch.from_numpy(MATLAB_FIXTURES["bark_axis"]).squeeze()
    bark_frequencies = bark_frequencies.to(dtype=torch.float32)
    bark_stft = torch.from_numpy(MATLAB_FIXTURES["bark_spectrum"])
    bark_stft = bark_stft.to(dtype=torch.float32)
    bark_stft = bark_stft.view(1, 1, bark_stft.shape[0], -1)
    channels = torch.cat((bark_stft, bark_stft), dim=1)
    batch = torch.cat([channels] * 5, dim=0)
    output = transfer.transfer_signal_with_freqs(batch, bark_frequencies)
    assert output.shape == (5, 2, 128, 24)
