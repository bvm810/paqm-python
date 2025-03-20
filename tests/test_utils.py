import os
import scipy
import torch
from torch.utils.data import DataLoader
from paqm.utils import bark_to_hertz, hertz_to_bark, collate
from paqm.utils.dataset import PAQMDataset

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")


def test_bark_to_hertz():
    fixtures_path = os.path.join(FIXTURES_PATH, "matlab", "bark2hertz.mat")
    fixtures = scipy.io.loadmat(fixtures_path)
    input = torch.from_numpy(fixtures["input"])
    expected_output = torch.from_numpy(fixtures["output"])
    output = bark_to_hertz(input)
    assert torch.allclose(expected_output, output)


def test_hertz_to_bark():
    fixtures_path = os.path.join(FIXTURES_PATH, "matlab", "hertz2bark.mat")
    fixtures = scipy.io.loadmat(fixtures_path)
    input = torch.from_numpy(fixtures["input"])
    expected_output = torch.from_numpy(fixtures["output"])
    output = hertz_to_bark(input)
    assert torch.allclose(expected_output, output)


def test_paqm_dataset():
    ref_folder = os.path.join(FIXTURES_PATH, "dataset-fixtures", "references")
    references = sorted([os.path.join(ref_folder, p) for p in os.listdir(ref_folder)])
    input_folder = os.path.join(FIXTURES_PATH, "dataset-fixtures", "inputs")
    inputs = sorted([os.path.join(input_folder, p) for p in os.listdir(input_folder)])
    dataset = PAQMDataset(inputs, references)
    assert len(dataset) == 3
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)
    input, ref = next(iter(loader))
    assert input.shape == ref.shape
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate)
    input, ref = next(iter(loader))
    assert input.shape == ref.shape
