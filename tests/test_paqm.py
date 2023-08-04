import os
import scipy
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from paqm.paqm import PAQM

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
WGN_PAQM_SAME = scipy.io.loadmat(os.path.join(FIXTURES_PATH, "paqm-same-wgn.mat"))
MP3_PAQM = scipy.io.loadmat(os.path.join(FIXTURES_PATH, "paqm-audio.mat"))
WGN = torch.from_numpy(WGN_PAQM_SAME["input"])
WGN = WGN.to(dtype=torch.float32).view(1, 1, WGN.shape[-1])
REFERENCE = torch.from_numpy(MP3_PAQM["reference"])
REFERENCE = REFERENCE.to(dtype=torch.float32).view(1, 1, REFERENCE.shape[0])
AUDIO = torch.from_numpy(MP3_PAQM["audio"])
AUDIO = AUDIO.to(dtype=torch.float32).view(1, 1, AUDIO.shape[0])
evaluator_same = PAQM(WGN, WGN)
evaluator_mp3 = PAQM(AUDIO, REFERENCE)


def test_internal_representation_with_wgn():
    # only looking at bins below 24 barks (which are the only ones used in the score)
    output = evaluator_same._get_internal_representation(WGN).squeeze()[:121, :]
    expected_output = torch.from_numpy(WGN_PAQM_SAME["Lx"])
    expected_output = expected_output.T.to(dtype=torch.float32)[:121, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    # 5% relative error tolerance because of interpolation error when transfering to inner ear
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)


def test_internal_representation_with_mp3():
    # only looking at bins below 24 barks (which are the only ones used in the score)
    output = evaluator_mp3._get_internal_representation(AUDIO).squeeze()[:121, :]
    expected_output = torch.from_numpy(MP3_PAQM["Ly"])
    expected_output = expected_output.T.to(dtype=torch.float32)[:121, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    # 5% relative error tolerance because of interpolation error when transfering to inner ear
    # had to increase criteria to 0.1% of range because of clip in compression
    assert torch.allclose(output, expected_output, atol=(range * 1e-3), rtol=5e-2)


def test_scaling_with_wgn():
    input = torch.from_numpy(WGN_PAQM_SAME["Ly"])
    input = input.T.to(dtype=torch.float32)
    # had to create last bin filled with zeros since MATLAB version has 1 bin less
    last_bin = torch.zeros((1, input.shape[-1]))
    input = torch.cat((input, last_bin), dim=0)
    input = input.view(1, 1, input.shape[0], input.shape[1])
    reference = torch.from_numpy(WGN_PAQM_SAME["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    reference = torch.cat((reference, last_bin), dim=0)
    reference = reference.view(1, 1, reference.shape[0], reference.shape[1])
    output = evaluator_same._scaling(input, reference)
    expected_output = torch.from_numpy(WGN_PAQM_SAME["Lys"])
    expected_output = expected_output.T.to(dtype=torch.float32)
    assert torch.allclose(output[..., :-1, :], expected_output, atol=1e-6, rtol=1e-2)


def test_scaling_with_mp3():
    input = torch.from_numpy(MP3_PAQM["Ly"])
    input = input.T.to(dtype=torch.float32)
    # had to create last bin filled with zeros since MATLAB version has 1 bin less
    last_bin = torch.zeros((1, input.shape[-1]))
    input = torch.cat((input, last_bin), dim=0)
    input = input.view(1, 1, input.shape[0], input.shape[1])
    reference = torch.from_numpy(MP3_PAQM["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    reference = torch.cat((reference, last_bin), dim=0)
    reference = reference.view(1, 1, reference.shape[0], reference.shape[1])
    output = evaluator_same._scaling(input, reference)
    output = output[..., :-1, :]
    expected_output = torch.from_numpy(MP3_PAQM["Lys"])
    expected_output = expected_output.T.to(dtype=torch.float32)
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)


def test_batch_scaling():
    input = torch.from_numpy(WGN_PAQM_SAME["Ly"])
    input = input.T.to(dtype=torch.float32)
    # had to create last bin filled with zeros since MATLAB version has 1 bin less
    last_bin = torch.zeros((1, input.shape[-1]))
    input = torch.cat((input, last_bin), dim=0)
    input = input.view(1, 1, input.shape[0], input.shape[1])
    stereo_input = torch.cat((input, input), dim=1)
    batch_input = torch.cat([stereo_input] * 5, dim=0)
    reference = torch.from_numpy(WGN_PAQM_SAME["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    reference = torch.cat((reference, last_bin), dim=0)
    reference = reference.view(1, 1, reference.shape[0], reference.shape[1])
    stereo_reference = torch.cat((reference, reference), dim=1)
    batch_reference = torch.cat([stereo_reference] * 5, dim=0)
    output = evaluator_same._scaling(batch_input, batch_reference)
    assert output.shape == (5, 2, 129, 184)
    expected_output = torch.from_numpy(WGN_PAQM_SAME["Lys"])
    expected_output = expected_output.T.to(dtype=torch.float32)
    assert torch.allclose(output[2, 1, :-1, :], expected_output, atol=1e-6, rtol=1e-2)


def test_full_scores_on_same_audio():
    output = evaluator_same.full_scores.squeeze()[:-1, :]
    input_scaled = torch.from_numpy(WGN_PAQM_SAME["Lys"])
    input_scaled = input_scaled.T.to(dtype=torch.float32)
    reference = torch.from_numpy(WGN_PAQM_SAME["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    expected_output = torch.abs(input_scaled - reference)
    assert torch.allclose(output, expected_output, atol=1e-6, rtol=1e-2)
    assert torch.allclose(output, torch.zeros(output.shape), atol=1e-6, rtol=1e-2)


def test_internal_representation_with_scaling():
    input = evaluator_mp3._get_internal_representation(AUDIO)
    expected_input = torch.from_numpy(MP3_PAQM["Ly"])
    expected_input = expected_input.T.to(dtype=torch.float32)
    range = torch.max(expected_input) - torch.min(expected_input)
    assert torch.allclose(
        input.squeeze()[:-1, :], expected_input, atol=(range * 1e-3), rtol=5e-2
    )
    reference = evaluator_mp3._get_internal_representation(REFERENCE)
    expected_ref = torch.from_numpy(MP3_PAQM["Lx"])
    expected_ref = expected_ref.T.to(dtype=torch.float32)
    range = torch.max(expected_ref) - torch.min(expected_ref)
    assert torch.allclose(
        reference.squeeze()[:-1, :], expected_ref, atol=(range * 1e-3), rtol=5e-2
    )
    output = evaluator_mp3._scaling(input, reference)
    expected_output = torch.from_numpy(MP3_PAQM["Lys"])
    expected_output = expected_output.T.to(dtype=torch.float32)
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(
        output.squeeze()[:-1, :], expected_output, atol=(range * 1e-3), rtol=5e-2
    )


def test_full_scores_on_compressed_audio():
    input_scaled = torch.from_numpy(MP3_PAQM["Lys"])
    input_scaled = input_scaled.T.to(dtype=torch.float32)
    reference = torch.from_numpy(MP3_PAQM["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    expected_output = torch.abs(input_scaled - reference)
    range = torch.max(expected_output) - torch.min(expected_output)
    output = evaluator_mp3.full_scores.squeeze()[:-1, :]
    close = torch.isclose(output, expected_output, atol=(range * 1e-3), rtol=5e-2)

    # visual check
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(17, 4), ncols=3)
    obtained = ax1.imshow(output, aspect="auto", origin="lower")
    ax1.set_title("Noise Disturbance")
    ax1.set_xlabel("Frames")
    ax1.set_ylabel("Bark bins")
    fig.colorbar(obtained, ax=ax1)
    expected = ax2.imshow(expected_output, aspect="auto", origin="lower")
    ax2.set_title("Expected Noise Disturbance")
    ax2.set_xlabel("Frames")
    ax2.set_ylabel("Bark bins")
    fig.colorbar(expected, ax=ax2)
    cmap = ListedColormap(["red", "green"])
    ax3.imshow(close, aspect="auto", origin="lower", cmap=cmap)
    ax3.set_title("Bins Out of Tolerance")
    ax3.set_xlabel("Frames")
    ax3.set_ylabel("Bark bins")
    plt.show()

    assert torch.allclose(output, expected_output, atol=(range * 1e-3), rtol=5e-2)


def test_frame_scores():
    pass


def test_average_score_with_mp3():
    output = evaluator_mp3.score
    expected_output = torch.from_numpy(MP3_PAQM["Ln"])
    expected_output = expected_output.to(dtype=torch.float32)[0, 0]
    print((output))
    print((expected_output))
    print(torch.log(output))
    print(torch.log(expected_output))
    assert torch.isclose(output, expected_output, atol=1e-5, rtol=1e-2)


def test_mean_opinion_score():
    pass
