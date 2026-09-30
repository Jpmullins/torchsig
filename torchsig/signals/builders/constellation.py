"""Constellation Signal Builder and Modulator Module"""

from __future__ import annotations

import numpy as np
import scipy.signal as sp

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.builders.constellation_maps import all_symbol_maps
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    estimate_filter_length,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
    srrc_taps,
)

__all__ = ["ConstellationSignalGenerator", "constellation_modulator", "constellation_modulator_baseband"]

# Pulse shapes accepted by the optional ``pulse_shape_name`` metadata key of ConstellationSignalGenerator.
_PULSE_SHAPE_NAMES = ("srrc", "rectangular")

# SRRC rolloff range ConstellationSignalGenerator draws from when alpha_rolloff_min/alpha_rolloff_max are not set.
_DEFAULT_ALPHA_ROLLOFF_RANGE = (0.1, 0.5)


def constellation_modulator_baseband(
    constellation_name: str,
    pulse_shape_name: str,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    alpha_rolloff: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Modulates constellation based signals (QAM/PSK/ASK/OOK) at complex baseband.

    Args:
        constellation_name: Name of the signal to modulate (e.g., 'qpsk').
        pulse_shape_name: Pulse shaping filter selection ('rectangular' or 'srrc').
        max_num_samples: Maximum number of samples to be produced.
        oversampling_rate_nominal: The amount of oversampling (sampling_rate/bandwidth).
        alpha_rolloff: The alpha-rolloff value for the SRRC filter (0 < alpha_rolloff < 1).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: IQ samples of the constellation-modulated complex baseband signal.

    Raises:
        ValueError: If pulse_shape_name is neither 'rectangular' or 'srrc'.
        ValueError: If alpha_rolloff is not defined when selecting 'srrc'.
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    # Input validation
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Get symbol map and normalize to unit power
    symbol_map = all_symbol_maps[constellation_name]
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map) ** 2))

    # Pulse shaping
    samples_per_symbol = oversampling_rate_nominal

    if pulse_shape_name == "rectangular":
        pulse_shape = np.ones(samples_per_symbol)
        pulse_shape_filter_span = 0
    elif pulse_shape_name == "srrc":
        if alpha_rolloff is None:
            raise ValueError("must define an alpha rolloff for SRRC filter")
        if not 0 < alpha_rolloff < 1:
            raise ValueError("alpha_rolloff must be between 0 and 1")

        attenuation_db = 120
        pulse_shape_filter_length = estimate_filter_length(alpha_rolloff, attenuation_db, 1)
        pulse_shape_filter_span = int(np.ceil((pulse_shape_filter_length - 1) / (2 * samples_per_symbol)))
        pulse_shape = srrc_taps(samples_per_symbol, pulse_shape_filter_span, alpha_rolloff)
    else:
        raise ValueError(f"pulse shape {pulse_shape_name} not supported")

    # Calculate number of symbols to generate
    subtract_off_symbols = 2 * pulse_shape_filter_span
    num_symbols = int(np.floor(max_num_samples / samples_per_symbol)) - subtract_off_symbols
    num_symbols = max(num_symbols, 1)

    # Generate symbols (handle OOK case where symbols might be zero)
    symbols = np.zeros(1)
    while np.equal(np.sum(np.abs(symbols)), 0):
        map_index = rng.integers(low=0, high=len(symbol_map), size=num_symbols)
        symbols = symbol_map[map_index]

    # Apply pulse shaping
    constellation_signal_baseband = sp.upfirdn(pulse_shape, symbols, up=samples_per_symbol, down=1)

    # Adjust signal length
    if len(constellation_signal_baseband) < max_num_samples:
        constellation_signal_baseband = pad_head_tail_to_length(constellation_signal_baseband, max_num_samples)
    elif len(constellation_signal_baseband) > max_num_samples:
        constellation_signal_baseband = slice_tail_to_length(constellation_signal_baseband, max_num_samples)

    return constellation_signal_baseband.astype(TorchSigComplexDataType)


def constellation_modulator(
    constellation_name: str,
    pulse_shape_name: str,
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    alpha_rolloff: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Modulator for constellation-based signals (QAM/PSK/ASK/OOK).

    Args:
        constellation_name: The modulation to create (e.g., 'qpsk').
        pulse_shape_name: Pulse shaping filter selection ('rectangular' or 'srrc').
        bandwidth: Desired 3 dB bandwidth of the signal (Hz).
        sample_rate: Sampling rate for the IQ signal (Hz).
        num_samples: Number of IQ samples to produce.
        alpha_rolloff: The alpha-rolloff value for the SRRC filter (0 < alpha_rolloff < 1).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: Constellation-modulated IQ samples at the appropriate bandwidth.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive.
        ValueError: If bandwidth exceeds sample_rate/2.
        ValueError: If num_samples is not positive.
        ValueError: If the number of samples produced is incorrect.
    """
    # Input validation
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Calculate resampling parameters
    oversampling_rate = sample_rate / bandwidth
    oversampling_rate_baseband = 4
    resample_rate_ideal = oversampling_rate / oversampling_rate_baseband

    # Determine baseband samples
    num_samples_baseband_init = int(np.floor(num_samples / resample_rate_ideal))
    num_samples_baseband = oversampling_rate_baseband if num_samples_baseband_init <= 0 else num_samples_baseband_init

    # Generate baseband signal
    constellation_signal_baseband = constellation_modulator_baseband(
        constellation_name,
        pulse_shape_name,
        num_samples_baseband,
        oversampling_rate_baseband,
        alpha_rolloff,
        rng,
    )

    # Apply resampling
    constellation_mod_correct_bw = multistage_polyphase_resampler(constellation_signal_baseband, resample_rate_ideal)

    # Adjust signal length
    constellation_mod_signal = (
        slice_head_tail_to_length(constellation_mod_correct_bw, num_samples) if len(constellation_mod_correct_bw) > num_samples else pad_head_tail_to_length(constellation_mod_correct_bw, num_samples)
    )

    # Validate output length
    if len(constellation_mod_signal) != num_samples:
        raise ValueError(f"constellation mod producing incorrect number of samples: {len(constellation_mod_signal)} but requested: {num_samples}")

    return constellation_mod_signal.astype(TorchSigComplexDataType)


class ConstellationSignalGenerator(BaseSignalGenerator):
    """Constellation Signal Generator.

    Implements multiple constellation based waveforms (QAM/PSK/ASK/OOK).

    Each signal is pulse shaped with either a square-root raised-cosine (SRRC)
    or a rectangular pulse, and the choice sets the occupied bandwidth: an SRRC
    pulse with rolloff alpha occupies (1 + alpha) times the requested bandwidth,
    while the sinc-shaped spectrum of the rectangular pulse spreads well beyond
    twice the requested bandwidth. By default the pulse shape and the SRRC
    rolloff are drawn for every signal. The optional ``pulse_shape_name``,
    ``alpha_rolloff_min`` and ``alpha_rolloff_max`` metadata keys pin them, so
    a caller can know the occupied bandwidth before generating a signal.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes Constellation Signal Generator.

        Args:
            **kwargs: Metadata parameters including:

                - constellation_name: Name of the constellation (e.g., 'qpsk')
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
                - pulse_shape_name: (optional) 'srrc' or 'rectangular'. Pins the
                  pulse shape. If absent or None, SRRC or rectangular is drawn per
                  signal with equal probability.
                - alpha_rolloff_min: (optional) Lower bound of the SRRC rolloff,
                  set together with alpha_rolloff_max, with
                  0 < alpha_rolloff_min <= alpha_rolloff_max < 1. The rolloff is
                  drawn uniformly from [alpha_rolloff_min, alpha_rolloff_max), or
                  is exactly that value when the two are equal. If both are absent
                  or None, the range is [0.1, 0.5). The pair has no effect on a
                  rectangular pulse but is validated whenever it is set.
                - alpha_rolloff_max: (optional) Upper bound of the SRRC rolloff;
                  see alpha_rolloff_min.

        Raises:
            ValueError: If required metadata fields are missing or invalid.
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "constellation_name",
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name(str(self["constellation_name"]))

    def generate(self) -> Signal:
        """Generates a constellation signal based on the configured parameters.

        Random values are drawn from ``self.random_generator`` in this order:
        the signal duration, the bandwidth, the pulse shape, the rolloff (SRRC
        only), then the symbols. The pulse-shape draw is skipped when
        ``pulse_shape_name`` is set, and the rolloff draw is skipped when
        ``alpha_rolloff_min == alpha_rolloff_max``, so a pinned value consumes
        no draw, as with ``bandwidth_min == bandwidth_max``. With none of the
        optional keys set, the draws and the output for a given seed are the
        same as in TorchSig 2.2.0. The optional keys are validated before any
        value is drawn.

        Returns:
            Signal: Generated constellation signal with metadata, including
            ``pulse_shape_name`` ('srrc' or 'rectangular') and ``alpha_rolloff``
            (the SRRC rolloff, or None for the rectangular pulse).

        Raises:
            ValueError: If required metadata fields are missing or invalid.
            ValueError: If pulse_shape_name is set to a value other than
                'srrc' or 'rectangular'.
            ValueError: If only one of alpha_rolloff_min and alpha_rolloff_max
                is set, or if they do not satisfy
                0 < alpha_rolloff_min <= alpha_rolloff_max < 1.
        """
        # Optional pulse-shape metadata, validated before any value is drawn. A key
        # that is absent or None leaves the choice to the draws below.
        pulse_shape_name = self["pulse_shape_name"] if hasattr(self, "pulse_shape_name") else None
        if pulse_shape_name is not None and pulse_shape_name not in _PULSE_SHAPE_NAMES:
            raise ValueError(f"unknown pulse_shape_name {pulse_shape_name!r}; expected one of {list(_PULSE_SHAPE_NAMES)}")
        alpha_rolloff_min = self["alpha_rolloff_min"] if hasattr(self, "alpha_rolloff_min") else None
        alpha_rolloff_max = self["alpha_rolloff_max"] if hasattr(self, "alpha_rolloff_max") else None
        if alpha_rolloff_min is None and alpha_rolloff_max is None:
            alpha_rolloff_min, alpha_rolloff_max = _DEFAULT_ALPHA_ROLLOFF_RANGE
        if alpha_rolloff_min is None or alpha_rolloff_max is None:
            raise ValueError("alpha_rolloff_min and alpha_rolloff_max must be set together")
        if not 0 < alpha_rolloff_min <= alpha_rolloff_max < 1:
            raise ValueError(f"alpha_rolloff_min and alpha_rolloff_max must satisfy 0 < alpha_rolloff_min <= alpha_rolloff_max < 1, got {alpha_rolloff_min} and {alpha_rolloff_max}")

        # Get parameters from metadata
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        constellation_name = self["constellation_name"]

        # Pulse shape: pinned by metadata, otherwise SRRC or rectangular with equal probability
        if pulse_shape_name is None:
            pulse_shape_name = "srrc" if self.random_generator.integers(0, 2) == 0 else "rectangular"

        # SRRC rolloff: a single-valued range is used as is, without a draw
        if pulse_shape_name == "rectangular":
            alpha_rolloff = None
        elif alpha_rolloff_min == alpha_rolloff_max:
            alpha_rolloff = float(alpha_rolloff_min)
        else:
            alpha_rolloff = self.random_generator.uniform(alpha_rolloff_min, alpha_rolloff_max)

        # Generate signal
        signal_data = constellation_modulator(
            constellation_name,
            pulse_shape_name,
            bandwidth,
            sample_rate,
            num_iq_samples_signal,
            alpha_rolloff,
            self.random_generator,
        )

        return Signal(
            data=signal_data,
            center_freq=0,
            bandwidth=bandwidth,
            pulse_shape_name=pulse_shape_name,
            alpha_rolloff=alpha_rolloff,
            pulse_shape_index=int(pulse_shape_name == "srrc"),
            alpha_rolloff_target=(float(alpha_rolloff) if alpha_rolloff is not None else 0.0),
        )
