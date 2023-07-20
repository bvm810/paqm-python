import os
import scipy
import torch
import matplotlib.pyplot as plt
from paqm.spectrum import SpectrumAnalyzer

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")
MATLAB_FIXTURES = scipy.io.loadmat(os.path.join(FIXTURES_PATH, "barkSpectrum.mat"))
analyzer = SpectrumAnalyzer(
    fs=44100,
    frame_duration=0.04,
    window="hann",
    overlap=0.5,
    nfft=2048,
    bark_binwidth=0.2,
)


def test_freq_axis_in_hertz():
    expected_output = torch.from_numpy(MATLAB_FIXTURES["hertz_axis"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    output = analyzer.freq_axis_in_hertz
    assert torch.allclose(expected_output, output)


def test_freq_axis_in_barks():
    expected_output = torch.from_numpy(MATLAB_FIXTURES["bark_axis"])
    expected_output = expected_output.to(dtype=torch.float32).squeeze()
    output = analyzer.freq_axis_in_barks
    assert torch.allclose(expected_output, output)


def test_spectrogram_with_long_cosine():
    signal = torch.from_numpy(MATLAB_FIXTURES["input"])
    signal = signal.to(dtype=torch.float64).T.unsqueeze(0)
    expected_output = torch.from_numpy(MATLAB_FIXTURES["power_spectrogram"])
    expected_output = expected_output.to(dtype=torch.float64)
    output = analyzer.power_spectrum(signal)
    assert output.shape == (1, 1, 1025, 24)
    output = analyzer.power_spectrum(signal).squeeze()
    # zeros are causing torch allclose to require lower tolerance
    assert torch.allclose(expected_output, output, atol=1e-08, rtol=1e-03)


def test_spectrogram_with_short_cosine():
    signal = torch.cos(2 * (torch.pi / 8) * torch.arange(32))
    signal = signal.unsqueeze(0)
    simple_analyzer = SpectrumAnalyzer(
        fs=8, frame_duration=1.0, window="hann", overlap=0.0, nfft=8
    )
    expected_output = torch.load(os.path.join(FIXTURES_PATH, "stft-simple.pt"))
    output = simple_analyzer.power_spectrum(signal).squeeze()
    # zeros are causing torch allclose to require lower tolerance
    assert torch.allclose(expected_output, output, atol=1e-08, rtol=1e-03)


def test_bark_filterbank():
    filterbank = analyzer._get_bark_filterbank()
    plt.imshow(filterbank, aspect="auto", origin="lower")
    plt.show()
    assert filterbank.shape == (128, 1025)
    non_zero_count = torch.count_nonzero(filterbank, dim=0)
    assert torch.equal(non_zero_count, torch.ones(1025, dtype=torch.int64))


def test_bark_spectrum_with_cosine():
    signal = torch.from_numpy(MATLAB_FIXTURES["input"])
    signal = signal.to(dtype=torch.float32).T
    expected_output = torch.from_numpy(MATLAB_FIXTURES["bark_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    output = analyzer.bark_spectrum(signal).squeeze()
    # zeros are causing torch allclose to require lower tolerance
    assert torch.allclose(expected_output, output, atol=1e-06, rtol=1e-02)


def test_batch():
    signal = torch.from_numpy(MATLAB_FIXTURES["input"])
    signal = signal.to(dtype=torch.float32).T.unsqueeze(0)
    print(signal.shape)
    stereo_signal = torch.cat((signal, signal), dim=1)
    print(stereo_signal.shape)
    batch = torch.cat([stereo_signal] * 3, dim=0)
    output = analyzer.bark_spectrum(batch)
    assert output.shape == (3, 2, 128, 24)
    single_example = output[1, 0, :, :]
    expected_output = torch.from_numpy(MATLAB_FIXTURES["bark_spectrum"])
    expected_output = expected_output.to(dtype=torch.float32)
    # zeros are causing torch allclose to require lower tolerance
    assert torch.allclose(expected_output, single_example, atol=1e-06, rtol=1e-02)
