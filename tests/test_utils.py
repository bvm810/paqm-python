import os
import scipy
import torch
from paqm.utils import bark_to_hertz, hertz_to_bark

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")


def test_bark_to_hertz():
    fixtures_path = os.path.join(FIXTURES_PATH, "bark2hertz.mat")
    fixtures = scipy.io.loadmat(fixtures_path)
    input = torch.from_numpy(fixtures["input"])
    expected_output = torch.from_numpy(fixtures["output"])
    output = bark_to_hertz(input)
    assert torch.allclose(expected_output, output)


def test_hertz_to_bark():
    fixtures_path = os.path.join(FIXTURES_PATH, "hertz2bark.mat")
    fixtures = scipy.io.loadmat(fixtures_path)
    input = torch.from_numpy(fixtures["input"])
    expected_output = torch.from_numpy(fixtures["output"])
    output = hertz_to_bark(input)
    assert torch.allclose(expected_output, output)
