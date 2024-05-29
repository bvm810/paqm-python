# paqm-python

This repo contains a pytorch implementation of the perceptual audio quality measure (PAQM), published by Beerends et al. in [this paper](https://www.aes.org/e-lib/browse.cfm?elib=7019). I made it as a way to study objective audio quality assessment in the context of [my master's](https://github.com/bvm810/diffusion-audio-restoration).

## Dependencies

* pytorch >= 1.12.1
* torchaudio >= 0.12.1
* scipy >= 1.11.1

## Installation

I intend to transform this implementation into a package in the future, but haven't had the time yet because of my master's. Until then, just clone the repository using
```
git clone git@github.com:bvm810/paqm-python.git <destination-folder>
```

## Usage

The main PAQM class wraps the processing blocks described in the reference paper. The comparison between references and a test audio signals can be made by time-frequency bin, by frame, or with the whole signals. The average for the whole signal can be converted to a mean opinion score (MOS).

```
from paqm.paqm import PAQM

# test_signal.shape --> (batch, channels, sample)
# reference_signal.shape --> (batch, channels, sample)
evaluator = PAQM(test_signal, reference_signal)

# get MOS scores
mos = evaluator.mean_opinion_score # (batch, channel)

# get raw scores
scores = evaluator.score # (batch, channel)

# get frame scores
frame_scores = evaluator.frame_scores # (batch, channel, frame)

# get bin scores
full_scores = evaluator.full_scores # (batch, channels, bark bin, frame)
```

References and test signals should have the same size. This repo also has a pytorch dataset subclass that can be used to load multiple audio files into a batch. An utility collate function to pad signals with NaNs can be found in the ``utils`` subfolder.

The dataset class performs some basic validation on the input data. It checks if they all have a specific sampling rate (which is 44.1kHz by default, but can be passed as an optional parameter to PAQMDataset), and if test signals and references have the same shape.

```
from paqm.utils import collate
from paqm.utils.dataset import PAQMDataset

# test_signals --> list with absolute paths to test signal files
# references --> list with absolute paths to reference files
dataset = PAQMDataset(test_signals, references)
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)
```

The intermediate processing steps of PAQM are handled by subclasses which are given as optional arguments to the PAQM class. In order:

#### SpectrumAnalyzer 

Handles the Fourier representation of the signals, and conversion to the perceptual Bark scale.
```
from paqm.spectrum import SpectrumAnalyzer

analyzer = SpectrumAnalyzer(
    fs = <sampling-rate> # float in hertz,
    frame_duration = 0.04,
    window = "hann",
    overlap = 0.5,
    nfft = 2048 # int,
    bark_binwidth = 0.2
)
```
#### OuterToInnerTransfer

Handles the transfer function between the outer and inner ears. The default transfer function points were obtained from a previous MATLAB implementation, and can be seem in ``transfer.py``
```
from paqm.transfer import OuterToInnerTransfer

transfer = OuterToInnerTransfer(
    transfer_function = <tf-value-pairs> # list of tuples (f_in_hertz, log_magnitude)
)
```

#### Masker

Handles time domain and frequency domain spreading. Time spreading is done autoregressively, and frequency spreading follows the log curves of the reference paper. The default parameters were taken from the reference paper, and can be found in ``masker.py``
```
from paqm.masker import Masker

masker = Masker(
    time_compression = 0.6,
    freq_compression = 0.8,
    tau_curve = <tau-time-decay-curve-points>,
    freq_spreading_constants = <freq-spreading-slopes>,
)
```

#### LoudnessCompressor

Handles computing compressed loudness from the excitation representations. Outputs the final internal representation used in PAQM. The first two parameters are taken from the reference paper; the hearing threshold is the ISO 226 hearing threshold measured in [phons](https://en.wikipedia.org/wiki/Phon)
```
from paqm.loudness import LoudnessCompressor

compressor = LoudnessCompressor(
    schwell_factor = 0.5,
    compression_level = 0.04,
    hearing_threshold = <hearing-threshold-in-phons>
)
```

## Testing

After cloning the repo automated tests can be executed running the ``pytest``command from the repository folder. The fixtures for the test were extracted from a previous MATLAB implementation created in the Audio Processing Group of the Signals, Multimedia and Telecommunications Lab (SMT) of the Federal University of Rio de Janeiro (UFRJ). 

There is one test ``.py`` file per class in the repo, plus one test file for the utility functions, one test file to check PAQM's usage for neural network training, and one test file to see if there were considerable numerical error differences between this implementation and the MATLAB reference. 

## Contributing

Feel free to create an issue to discuss potential changes. Any additions to the code should update tests as appropriate. 

## Acknowledgements

This project would not be possible without the suggestions from Prof. Luiz Wagner Biscainho and Lucas Simões Maia, from SMT / UFRJ.

## License

[MIT](https://choosealicense.com/licenses/mit/)







