"""Cellular Signal Builders."""

from __future__ import annotations

import numpy as np
import scipy.signal as sp

from torchsig.signals.builder import BaseSignalGenerator
from torchsig.signals.signal_types import Signal
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
)

# --------------------------------------------------------------------------- #
# GSM (Global System for Mobile)
# --------------------------------------------------------------------------- #
"""GSM is a 2G cellular standard using GMSK modulation with a TDMA frame
structure. The distinctive feature is a known 26-bit Training Sequence Code
(TSC) embedded in the middle of each burst, giving the waveform a periodic
autocorrelation peak that differentiates it from generic GMSK.

The builder models a base station (BTS) downlink carrier transmitting normal
bursts with random payload on every timeslot, as the BCCH carrier (C0) of a
cell does. A BTS is not required to ramp down and up between adjacent bursts
(3GPP TS 45.002 subclause 5.2.8), so the carrier stays on with constant
envelope and continuous phase across burst boundaries.

Physical layer (3GPP TS 45.002, 45.004):
    * Modulation    : GMSK, BT = 0.3, modulation index h = 0.5, with the
      differential encoding and sign convention of TS 45.004 subclause 2.3
      (alpha_i = 1 - 2 * (d_i XOR d_{i-1}); alpha_i = +1 raises the phase).
    * Symbol rate   : 270 833.333 Bd (13/48 MHz) nominal; like the other
      builders, the waveform is resampled to the drawn bandwidth, so the bit
      rate scales with it.
    * Channel BW    : 200 kHz
    * Timeslot      : 156.25 bits on average, as 157 bits on timeslots 0 and 4
      and 156 bits on the others (TS 45.002 subclause 5.2.2 integral option).

Normal burst layout (148 active bits, TS 45.002 subclause 5.2.3):
    3 tail | 57 data | 1 steal | 26 TSC | 1 steal | 57 data | 3 tail | 8 or 9 guard

Modeling choices:
    * One TSC from TSC Set 1, drawn once per signal (one cell) and used on
      every burst.
    * Data and stealing bits are random: no channel coding, interleaving or
      encryption.
    * The guard bits are modulated as the dummy bits d_i = 1 that define the
      modulator state before and after a burst (TS 45.004 subclause 2.2).
      osmo-trx and gr-gsm, which modulate each burst on its own, leave the
      guard unmodulated instead.
    * The first timeslot number is random.

Not modeled:
    * FCCH, SCH and dummy bursts, and the 51/26-multiframe structure.
    * Frequency hopping.
    * Handset (uplink) bursts, which ramp down and up in the guard period
      following the TS 45.005 power-time template.
"""

# Eight Training Sequence Codes of TSC Set 1 (3GPP TS 45.002 V15.1.0 Table 5.2.3a;
# identical to osmo-trx GSM/GSMCommon.cpp gTrainingSequence)
GSM_TSC: tuple[tuple[int, ...], ...] = (
    (0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1),
    (0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1),
    (0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0),
    (0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0),
    (0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1),
    (0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0),
    (1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1),
    (1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0),
)

_GSM_TAIL: np.ndarray = np.zeros(3, dtype=np.int8)
_GSM_DUMMY_BIT: int = 1  # TS 45.004 subclause 2.2: modulator state outside a burst


def gsm_timeslot_guard_bits(timeslot_number: int) -> int:
    """Number of guard bits after the burst in a timeslot.

    Timeslots 0 and 4 are 157 bits long and the others 156, giving the
    156.25-bit average timeslot (3GPP TS 45.002 subclause 5.2.2; osmo-trx
    Transceiver52M/Transceiver.cpp uses 8 + (TN % 4 == 0) guard bits).

    Args:
        timeslot_number: Timeslot number TN, 0-7.

    Returns:
        int: 9 for timeslots 0 and 4, 8 otherwise.
    """
    return 8 if timeslot_number % 4 else 9


def build_gsm_burst(tsc_idx: int, rng: np.random.Generator, num_guard_bits: int = 8) -> np.ndarray:
    """Builds the data bits of one GSM timeslot: a Normal Burst and its guard bits.

    Args:
        tsc_idx: Index 0-7 selecting the Training Sequence Code from TSC Set 1.
        rng: Random number generator.
        num_guard_bits: Guard bits after the 148 burst bits (8 or 9).

    Returns:
        np.ndarray: Data bits d_i in {0, 1}, length 148 + num_guard_bits. The
        guard bits are the dummy bits d_i = 1 of TS 45.004 subclause 2.2.
    """
    data_pre = rng.integers(0, 2, 57)
    data_post = rng.integers(0, 2, 57)
    steal_pre = rng.integers(0, 2, 1)
    steal_post = rng.integers(0, 2, 1)
    tsc = np.array(GSM_TSC[tsc_idx])
    guard = np.full(num_guard_bits, _GSM_DUMMY_BIT)

    bits = np.concatenate([_GSM_TAIL, data_pre, steal_pre, tsc, steal_post, data_post, _GSM_TAIL, guard])
    return bits.astype(np.int8)


def build_gsm_bit_stream(num_bits: int, rng: np.random.Generator) -> np.ndarray:
    """Builds the data bits of consecutive timeslots on one BTS carrier.

    One TSC is drawn for the whole stream, as a cell uses one training
    sequence. The stream starts at the beginning of a random timeslot and
    timeslots are 157 bits (TN 0 and 4) or 156 bits (other TN) long.

    Args:
        num_bits: Total data bits to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: Data bits d_i in {0, 1}, length num_bits.

    Raises:
        ValueError: If num_bits is not positive.
    """
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    tsc_idx = int(rng.integers(0, len(GSM_TSC)))
    timeslot_number = int(rng.integers(0, 8))
    bursts = []
    total = 0
    while total < num_bits:
        burst = build_gsm_burst(tsc_idx, rng, gsm_timeslot_guard_bits(timeslot_number))
        bursts.append(burst)
        total += len(burst)
        timeslot_number = (timeslot_number + 1) % 8
    return np.concatenate(bursts)[:num_bits]


def gsm_differential_encode(bits: np.ndarray) -> np.ndarray:
    """Maps GSM data bits to GMSK modulating values (3GPP TS 45.004 subclause 2.3).

    d_hat_i = d_i XOR d_{i-1} and alpha_i = 1 - 2 * d_hat_i, with d_{-1} = 1, the
    dummy bit that precedes a burst (subclause 2.2). gr-gsm's GSM GMSK
    modulator applies the same mapping (diff_decoder_bb, then
    chunks_to_symbols with table [1, -1]). osmo-trx reaches the same signal
    without explicit differential encoding by rotating 2 * d_i - 1 by +pi/2 per
    bit (sigProcLib.cpp, GMSKRotate), its linearized form.

    Args:
        bits: Data bits d_i in {0, 1}.

    Returns:
        np.ndarray: Modulating values alpha_i in {-1.0, +1.0}, same length as bits.
    """
    bits = np.asarray(bits, dtype=np.int8)
    previous = np.concatenate([[_GSM_DUMMY_BIT], bits[:-1]]).astype(np.int8)
    return 1.0 - 2.0 * np.bitwise_xor(bits, previous)


def _gaussian_pulse(samples_per_symbol: int, bt: float) -> np.ndarray:
    """Gaussian frequency pulse for GMSK (BT-product pulse shaping)."""
    m = 3
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / samples_per_symbol) ** 2)
    return p / np.sum(p)


def _gsm_frequency_pulse(samples_per_bit: int) -> np.ndarray:
    """GMSK frequency pulse g(t): BT = 0.3 Gaussian convolved with a one-bit rect (TS 45.004 subclause 2.4)."""
    return sp.convolve(_gaussian_pulse(samples_per_bit, bt=0.3), np.ones(samples_per_bit))


def gsm_gmsk_modulate(bits: np.ndarray, samples_per_bit: int) -> np.ndarray:
    """GMSK-modulates GSM data bits at complex baseband (3GPP TS 45.004).

    The bits are differentially encoded, filtered with a BT = 0.3 Gaussian
    frequency pulse and integrated to phase with h = 0.5, so alpha_i = +1
    advances the phase by pi/2 over the bit (positive frequency).

    Args:
        bits: Data bits d_i in {0, 1}.
        samples_per_bit: Samples per bit.

    Returns:
        np.ndarray: Unit-amplitude complex baseband, the full filter output of
        length (len(bits) - 1) * samples_per_bit + 7 * samples_per_bit.
        The frequency pulse of alpha_i peaks at sample (i + 3.5) * samples_per_bit - 0.5.
    """
    pulse_shape = _gsm_frequency_pulse(samples_per_bit)
    freq = sp.upfirdn(pulse_shape, gsm_differential_encode(bits), up=samples_per_bit, down=1)
    # h=0.5 MSK: scale so one symbol-period of rectangular freq = pi/2 phase
    phase = np.cumsum(freq) * (np.pi * 0.5 / samples_per_bit)
    return np.exp(1j * phase)


def gsm_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """GSM GMSK modulator at complex baseband.

    Args:
        max_num_samples: Maximum output samples.
        oversampling_rate_nominal: Samples per bit at baseband.
        rng: Random number generator.

    Returns:
        np.ndarray: Baseband GSM signal, exactly max_num_samples long.

    Raises:
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    sps = oversampling_rate_nominal
    max_minus_filter = max_num_samples - len(_gsm_frequency_pulse(sps)) + 1
    num_bits = max(1, int(np.floor(max_minus_filter / sps)))

    bits = build_gsm_bit_stream(num_bits, rng)
    modulated = gsm_gmsk_modulate(bits, sps)

    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated


def gsm_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """GSM GMSK modulator: builds burst stream and resamples to target bandwidth.

    Args:
        bandwidth: Desired signal bandwidth (Hz).
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.

    Returns:
        np.ndarray: GSM IQ at the requested bandwidth, length num_samples.

    Raises:
        ValueError: If bandwidth or sample_rate are not positive, bandwidth > sample_rate/2,
            or num_samples is not positive.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if rng is None:
        rng = np.random.default_rng()

    oversampling_rate_nominal = 4
    oversampling_rate = sample_rate / bandwidth
    resample_rate_ideal = oversampling_rate / oversampling_rate_nominal
    max_num_samples = max(oversampling_rate_nominal, int(np.floor(num_samples / resample_rate_ideal)))

    baseband = gsm_modulator_baseband(max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class GSMSignalGenerator(BaseSignalGenerator):
    """GSM Signal Generator.

    Builds a GSM base station downlink carrier: back-to-back GMSK Normal
    Bursts on every timeslot, with random data bits around one standard
    Training Sequence Code per signal. See the GSM notes above for what is
    and is not modeled.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the GSM Signal Generator.

        Args:
            **kwargs: Metadata parameters including:
                - sample_rate: Sampling rate (Hz)
                - bandwidth_min: Minimum bandwidth (Hz)
                - bandwidth_max: Maximum bandwidth (Hz)
                - signal_duration_in_samples_min: Minimum signal duration (samples)
                - signal_duration_in_samples_max: Maximum signal duration (samples)
        """
        super().__init__(**kwargs)
        self.required_metadata_fields = [
            "sample_rate",
            "bandwidth_min",
            "bandwidth_max",
            "signal_duration_in_samples_min",
            "signal_duration_in_samples_max",
        ]
        self.set_default_class_name("gsm")

    def generate(self) -> Signal:
        """Generates a GSM signal.

        Returns:
            Signal: Generated GSM signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        signal_data = gsm_modulator(bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
