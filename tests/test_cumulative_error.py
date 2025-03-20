import os
import scipy
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from src.paqm.paqm import PAQM
from src.paqm.transfer import OuterToInnerTransfer

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
CUMULATIVE = scipy.io.loadmat(
    os.path.join(FIXTURES_PATH, "matlab", "cumulative-test.mat")
)
AUDIO = torch.from_numpy(CUMULATIVE["x_pad"])
AUDIO = AUDIO.to(dtype=torch.float32).view(1, 1, AUDIO.shape[0])
evaluator = PAQM(AUDIO, AUDIO)


def test_cumulative_error():
    bark_stft, axis, overlap_duration = check_analyzer(AUDIO)
    inner_ear_spec = check_transfer(bark_stft, axis)
    time_spread_spec = check_time_spread(inner_ear_spec, axis, overlap_duration)
    freq_spread_spec = check_freq_spread(time_spread_spec, axis)
    internal_representation = check_compressor(
        freq_spread_spec, axis, evaluator.transfer
    )


def check_analyzer(audio: torch.Tensor):
    bark_stft = evaluator.analyzer.bark_spectrum(audio)
    axis = evaluator.analyzer.freq_axis_in_barks
    overlap_duration = evaluator.analyzer.overlap_duration
    output = bark_stft.squeeze()[:-1, :]
    expected_output = torch.from_numpy(CUMULATIVE["ply"])
    expected_output = expected_output.T.to(dtype=torch.float32)
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)
    return bark_stft, axis, overlap_duration


def check_transfer(bark_stft: torch.Tensor, axis: torch.Tensor):
    inner_ear_spec = evaluator.transfer.transfer_signal_with_freqs(bark_stft, axis)
    # only looking up to bin 115 because of interpolation error for high freqs
    output = inner_ear_spec.squeeze()[:115, :]
    expected_output = torch.from_numpy(CUMULATIVE["pyt"])
    expected_output = expected_output.T.to(dtype=torch.float32)[:115, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)
    return inner_ear_spec


def check_time_spread(inner_ear_spec: torch.Tensor, axis: torch.Tensor, overlap: float):
    time_spread = evaluator.masker.time_domain_spreading(inner_ear_spec, axis, overlap)
    output = time_spread.squeeze()[:115, :]
    expected_output = torch.from_numpy(CUMULATIVE["py"])
    expected_output = expected_output.T.to(dtype=torch.float32)[:115, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)
    return time_spread


def check_freq_spread(time_spread_spec: torch.Tensor, axis: torch.Tensor):
    freq_spread = evaluator.masker.frequency_domain_spreading(time_spread_spec, axis)
    output = freq_spread.squeeze()[:115, :]
    expected_output = torch.from_numpy(CUMULATIVE["Ey"])
    expected_output = expected_output.T.to(dtype=torch.float32)[:115, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)
    return freq_spread


def check_compressor(
    freq_spread_spec: torch.Tensor, axis: torch.Tensor, transfer: OuterToInnerTransfer
):
    internal_representation = evaluator.compressor.compress(
        freq_spread_spec, axis, transfer
    )
    output = internal_representation.squeeze()[:-1, :]
    expected_output = torch.from_numpy(CUMULATIVE["Ly"])
    expected_output = expected_output.T.to(dtype=torch.float32)[:, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    close = torch.isclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)

    # visual check
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(17, 4), ncols=3)
    obtained = ax1.imshow(output, aspect="auto", origin="lower")
    ax1.set_title("Internal Representation")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Bark bins")
    fig.colorbar(obtained, ax=ax1)
    expected = ax2.imshow(expected_output, aspect="auto", origin="lower")
    ax2.set_title("Expected Internal Representation")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Bark bins")
    cmap = ListedColormap(["red", "green"])
    ax3.imshow(close, aspect="auto", origin="lower", cmap=cmap)
    ax3.set_title("Bins Out of Tolerance")
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("Bark bins")
    fig.colorbar(expected, ax=ax2)
    plt.show()

    # 5% relative error tolerance because of interpolation error when transfering to inner ear
    # had to increase criteria to 0.1% of range because of clip in compression
    assert torch.allclose(output, expected_output, atol=(range * 1e-3), rtol=5e-2)
    return internal_representation
