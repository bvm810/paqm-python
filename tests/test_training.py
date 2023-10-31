import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from typing import Tuple, List
from paqm.paqm import PAQM
from paqm.utils import collate
from paqm.utils.dataset import PAQMDataset

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "fixtures", "training-fixtures")
EPOCHS = 10
torch.manual_seed(0)

class NeuralNetwork(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.net =  torch.nn.Sequential(
                    torch.nn.Linear(input_shape, 10),
                    torch.nn.Sigmoid(),
                    torch.nn.Linear(10, input_shape),
                )
        torch.nn.init.normal_(self.net[0].weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.net[2].weight, mean=0.0, std=1.0)
    def forward(self, x):
        x = self.net(x)
        return x

def _setup_network(input_shape: int) -> torch.nn.Module:
    net = NeuralNetwork(input_shape)

    return net


def _setup_data(root_folder: str) -> Tuple[List[str], List[str]]:
    ref_folder = os.path.join(root_folder, "references")
    references = [os.path.join(ref_folder, p) for p in os.listdir(ref_folder)]
    input_folder = os.path.join(root_folder, "inputs")
    inputs = [os.path.join(input_folder, p) for p in os.listdir(input_folder)]
    dataset = PAQMDataset(inputs, references)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return loader


def test_training():
    torch.autograd.set_detect_anomaly(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = _setup_data(DATA_FOLDER)
    input_size = next(iter(loader))[0].shape[-1]
    net = _setup_network(input_size).to(device)
    optimizer = Adam(net.parameters(), lr=1e-5)
    for _ in range(EPOCHS):
        previous_params = []
        for param in net.parameters():
            previous_params.append(param.clone())
        second_layer = list(net.parameters())[2]
        for input, ref in iter(loader):
            input, ref = input.to(device), ref.to(device)
            output = net(input)
            loss = PAQM(output, ref).score.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        after_params = []
        for param in net.parameters():
            after_params.append(param.clone())
    for i in range(len(after_params)):
        assert not torch.allclose(after_params[0], previous_params[0])