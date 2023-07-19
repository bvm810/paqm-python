import scipy
import torch

# Not entirely sure what those parameters are.
# f stands for frequency, but the other ones are not clear.
# They were extracted from Jeff Tacket's 2005 version of ISO 226 2003.
# There is a 2023 version, but it is not free
EQUAL_LOUDNESS_CONTOUR_PARAMS = [
    # (f, af, Lu, Tf)
    (20, 0.532, -31.6, 78.5),
    (25, 0.506, -27.2, 68.7),
    (31.5, 0.480, -23.0, 59.5),
    (40, 0.455, -19.1, 51.1),
    (50, 0.432, -15.9, 44.0),
    (63, 0.409, -13.0, 37.5),
    (80, 0.387, -10.3, 31.5),
    (100, 0.367, -8.1, 26.5),
    (125, 0.349, -6.2, 22.1),
    (160, 0.330, -4.5, 17.9),
    (200, 0.315, -3.1, 14.4),
    (250, 0.301, -2.0, 11.4),
    (315, 0.288, -1.1, 8.6),
    (400, 0.276, -0.4, 6.2),
    (500, 0.267, 0.0, 4.4),
    (630, 0.259, 0.3, 3.0),
    (800, 0.253, 0.5, 2.2),
    (1000, 0.250, 0.0, 2.4),
    (1250, 0.246, -2.7, 3.5),
    (1600, 0.244, -4.1, 1.7),
    (2000, 0.243, -1.0, -1.3),
    (2500, 0.243, 1.7, -4.2),
    (3150, 0.243, 2.5, -6.0),
    (4000, 0.242, 1.2, -5.4),
    (5000, 0.242, -2.1, -1.5),
    (6300, 0.245, -7.1, 6.0),
    (8000, 0.254, -11.2, 12.6),
    (10000, 0.271, -10.7, 13.9),
    (12500, 0.301, -3.1, 12.3),
]

PHON_HEARING_THRESHOLD = 3.539


class LoudnessCompressor:
    def __init__(
        self,
        schwell_factor: float = 0.5,
        compression_level: float = 0.04,
        hearing_threshold: float = PHON_HEARING_THRESHOLD,
    ) -> None:
        self.equal_loudness_contour_freqs = torch.Tensor(
            [p[0] for p in EQUAL_LOUDNESS_CONTOUR_PARAMS]
        )

    def equal_loudness_contour(self, loudness_level: float) -> torch.Tensor:
        # This method is not clear because I did not have access to the ISO 226 norm
        # It is directly translated from Jeff Tacket's 2005 version of ISO 226 2003.
        if (loudness_level < 0) or (loudness_level > 90):
            raise ValueError(
                "ISO 226 equal loudness contours are only defined between 0 phon and 90 phon"
            )
        af = torch.Tensor([p[1] for p in EQUAL_LOUDNESS_CONTOUR_PARAMS])
        Lu = torch.Tensor([p[2] for p in EQUAL_LOUDNESS_CONTOUR_PARAMS])
        Tf = torch.Tensor([p[3] for p in EQUAL_LOUDNESS_CONTOUR_PARAMS])
        Ln = loudness_level
        Af = (
            4.47e-3 * (10 ** (0.025 * Ln) - 1.15)
            + (0.4 * 10 ** (((Tf + Lu) / 10) - 9)) ** af
        )
        spl = ((10 / af) * torch.log10(Af)) - Lu + 94
        return spl

    def hearing_threshold_excitation(self, bark_freqs: torch.Tensor) -> torch.Tensor:
        pass

    def internal_representation(
        self, power_spectrum: torch.Tensor, bark_freqs: torch.Tensor
    ) -> torch.Tensor:
        pass
