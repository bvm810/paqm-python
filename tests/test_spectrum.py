import os
import scipy
import torch
import matplotlib.pyplot as plt
from paqm.spectrum import SpectrumAnalyzer


FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
MATLAB_FIXTURES = os.path.join(FIXTURES_PATH, "matlab")
COSINE_PAQM = scipy.io.loadmat(os.path.join(MATLAB_FIXTURES, "barkSpectrum.mat"))
WGN_PAQM = scipy.io.loadmat(os.path.join(MATLAB_FIXTURES, "wgn-bark-spectrum.mat"))
analyzer = SpectrumAnalyzer(
    fs=44100,
    frame_duration=0.04,
    window="hann",
    overlap=0.5,
    nfft=2048,
    bark_binwidth=0.2,
)


def test_freq_axis_in_hertz():
    expected_output = torch.from_numpy(COSINE_PAQM["hertz_axis"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    output = analyzer.freq_axis_in_hertz
    assert torch.allclose(expected_output, output)


# for all tests ignoring last bark bin as it is not calculated in the MATLAB version
def test_freq_axis_in_barks():
    expected_output = torch.from_numpy(COSINE_PAQM["bark_axis"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    output = analyzer.freq_axis_in_barks[:-1]
    assert torch.allclose(expected_output, output)


def test_spectrogram_with_long_cosine():
    signal = torch.from_numpy(COSINE_PAQM["input"])
    signal = signal.to(dtype=torch.float64).T.unsqueeze(0)
    expected_output = torch.from_numpy(COSINE_PAQM["power_spectrogram"])
    expected_output = expected_output.to(dtype=torch.float64)
    output = analyzer.power_spectrum(signal)
    assert output.shape == (1, 1, 1025, 24)
    output = analyzer.power_spectrum(signal).squeeze()
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=1e-2)


def test_spectrogram_with_short_cosine():
    signal = torch.cos(2 * (torch.pi / 8) * torch.arange(32))
    signal = signal.view(1, 1, signal.shape[-1])
    simple_analyzer = SpectrumAnalyzer(
        fs=8, frame_duration=1.0, window="hann", overlap=0.0, nfft=8
    )
    expected_output = torch.load(os.path.join(FIXTURES_PATH, "stft-simple.pt"))
    output = simple_analyzer.power_spectrum(signal).squeeze()
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=1e-2)


def test_bark_filterbank():
    filterbank = analyzer._get_bark_filterbank()
    plt.imshow(filterbank, aspect="auto", origin="lower")
    plt.title("Bark filterbank")
    plt.xlabel("Hertz bins")
    plt.ylabel("Bark bins")
    plt.show()
    assert filterbank.shape == (129, 1025)
    non_zero_count = torch.count_nonzero(filterbank, dim=0)
    assert torch.equal(non_zero_count, torch.ones(1025, dtype=torch.int64))


# using 0.01% of dynamic range as absolute tolerance as very small amplitudes can influence allclose
def test_bark_spectrum_with_cosine():
    signal = torch.from_numpy(COSINE_PAQM["input"])
    signal = signal.to(dtype=torch.float32).T
    signal = signal.view(1, 1, signal.shape[-1])
    expected_output = torch.from_numpy(COSINE_PAQM["bark_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    output = analyzer.bark_spectrum(signal).squeeze()[:-1, :]
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=1e-2)


def test_spectrogram_with_wgn():
    input = torch.from_numpy(WGN_PAQM["x_pad"])
    input = input.to(dtype=torch.float32)
    input = input.view(1, 1, input.shape[-1])
    output = analyzer.power_spectrum(input).squeeze()
    expected_output = torch.from_numpy(WGN_PAQM["power_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=1e-2)


def test_bark_spectrum_with_wgn():
    input = torch.from_numpy(WGN_PAQM["x_pad"])
    input = input.to(dtype=torch.float32)
    input = input.view(1, 1, input.shape[-1])
    output = analyzer.bark_spectrum(input).squeeze()[:-1, :]
    expected_output = torch.from_numpy(WGN_PAQM["bark_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(output, expected_output, atol=(range * 1e-4), rtol=1e-2)


def test_batch():
    signal = torch.from_numpy(COSINE_PAQM["input"])
    signal = signal.to(dtype=torch.float32).T.unsqueeze(0)
    stereo_signal = torch.cat((signal, signal), dim=1)
    batch = torch.cat([stereo_signal] * 3, dim=0)
    output = analyzer.bark_spectrum(batch)
    assert output.shape == (3, 2, 129, 24)
    single_example = output[1, 0, :-1, :]
    expected_output = torch.from_numpy(COSINE_PAQM["bark_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    range = torch.max(expected_output) - torch.min(expected_output)
    assert torch.allclose(
        expected_output, single_example, atol=(range * 1e-4), rtol=1e-2
    )
