import os
import scipy
import torch
import matplotlib.pyplot as plt
from paqm.paqm import PAQM
from paqm.spectrum import SpectrumAnalyzer

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
WGN_PAQM = scipy.io.loadmat(os.path.join(FIXTURES_PATH, "paqm-same-wgn.mat"))
REFERENCE = torch.from_numpy(WGN_PAQM["input"])
REFERENCE = REFERENCE.to(dtype=torch.float32).view(1, 1, REFERENCE.shape[-1])
AUDIO = torch.from_numpy(WGN_PAQM["input"])
AUDIO = AUDIO.to(dtype=torch.float32).view(1, 1, AUDIO.shape[-1])
evaluator = PAQM(AUDIO, REFERENCE)


def test_internal_representation():
    # only looking at bins below 24 barks (which are the only ones used in the score)
    output = evaluator._get_internal_representation(REFERENCE).squeeze()[:121, :]
    expected_output = torch.from_numpy(WGN_PAQM["Lx"])
    expected_output = expected_output.T.to(dtype=torch.float32)[:121, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    # 5% relative error tolerance because of interpolation error when transfering to inner ear
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)


def test_scaling():
    input = torch.from_numpy(WGN_PAQM["Ly"])
    input = input.T.to(dtype=torch.float32)
    # had to create last bin filled with zeros since MATLAB version has 1 bin less
    last_bin = torch.zeros((1, input.shape[-1]))
    input = torch.cat((input, last_bin), dim=0)
    input = input.view(1, 1, input.shape[0], input.shape[1])
    reference = torch.from_numpy(WGN_PAQM["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    reference = torch.cat((reference, last_bin), dim=0)
    reference = reference.view(1, 1, reference.shape[0], reference.shape[1])
    output = evaluator._scaling(input, reference)
    expected_output = torch.from_numpy(WGN_PAQM["Lys"])
    expected_output = expected_output.T.to(dtype=torch.float32)
    assert torch.allclose(output[..., :-1, :], expected_output, atol=1e-6, rtol=1e-2)


def test_batch_scaling():
    input = torch.from_numpy(WGN_PAQM["Ly"])
    input = input.T.to(dtype=torch.float32)
    # had to create last bin filled with zeros since MATLAB version has 1 bin less
    last_bin = torch.zeros((1, input.shape[-1]))
    input = torch.cat((input, last_bin), dim=0)
    input = input.view(1, 1, input.shape[0], input.shape[1])
    stereo_input = torch.cat((input, input), dim=1)
    batch_input = torch.cat([stereo_input] * 5, dim=0)
    reference = torch.from_numpy(WGN_PAQM["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    reference = torch.cat((reference, last_bin), dim=0)
    reference = reference.view(1, 1, reference.shape[0], reference.shape[1])
    stereo_reference = torch.cat((reference, reference), dim=1)
    batch_reference = torch.cat([stereo_reference] * 5, dim=0)
    output = evaluator._scaling(batch_input, batch_reference)
    assert output.shape == (5, 2, 129, 184)
    expected_output = torch.from_numpy(WGN_PAQM["Lys"])
    expected_output = expected_output.T.to(dtype=torch.float32)
    assert torch.allclose(output[2, 1, :-1, :], expected_output, atol=1e-6, rtol=1e-2)


def test_full_scores():
    output = evaluator.full_scores.squeeze()
    input_scaled = torch.from_numpy(WGN_PAQM["Lys"])
    input_scaled = input_scaled.T.to(dtype=torch.float32)
    reference = torch.from_numpy(WGN_PAQM["Lx"])
    reference = reference.T.to(dtype=torch.float32)
    expected_output = torch.abs(input_scaled - reference)
    # 5% relative error tolerance because of interpolation error when transfering to inner ear
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=5e-2)


def test_frame_scores():
    pass


def test_average_score():
    pass


def test_mean_opinion_score():
    pass
