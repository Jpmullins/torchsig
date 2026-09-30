"""Bluetooth Low Energy (BLE) Signal Builder.

BLE uses GFSK (BT=0.5, h=0.5) at 1 Msymbol/s. The distinctive structural
feature is a fixed 4-byte Access Address — 0x8E89BED6 for advertising — that
follows a short preamble, making BLE packets immediately identifiable by their
autocorrelation with the known access address sequence.

Physical layer (Bluetooth Core Spec 5.x, Vol 6, Part A):
    * Modulation   : GFSK, BT = 0.5, modulation index h = 0.5
    * Symbol rate  : 1 Msymbol/s
    * Occupied BW  : ~1 MHz

Advertising packet structure (Vol 6, Part B, Section 2.1):
    8-bit preamble (0xAA) | 32-bit access address | PDU (2–39 bytes) | 24-bit CRC

Symbol timing (selected by ``btle_modulator``):
    * Scaled (default): the symbol rate equals the requested bandwidth, one
      symbol per second per hertz, as in the other TorchSig builders. Packets
      are correct in symbols, but their duration scales with the bandwidth.
    * Standard (``pin_rate_to_standard=True``): the symbol rate is
      ``BTLE_SYMBOL_RATE_HZ`` whatever the bandwidth, so symbol and packet
      durations are those of the LE 1M PHY.

Toy simplifications (both timing modes):
    * Access address fixed to the advertising channel value (0x8E89BED6).
    * PDU type, payload, and CRC are random bits.
    * Fields are sent most significant bit first. The standard sends the least
      significant bit first (Vol 6, Part B, Section 1.2), so the on-air
      preamble and access address bit patterns differ from real packets.
    * Packets are concatenated without inter-frame gaps unless
      ``inter_frame_gap_symbols`` is set. A gap is a fixed number of silent
      symbols after every packet; neither the 150 microsecond inter frame
      space (T_IFS) nor the advertising interval and its random delay is
      modeled.
    * Only the LE 1M PHY is modeled, with the modulation index fixed at 0.5
      (the standard allows 0.45 to 0.55).
    * The symbol clock has no drift or jitter. In standard mode the realized
      rate differs from ``BTLE_SYMBOL_RATE_HZ`` only by the quantization of
      the fractional resampling stage: at most 100 ppm, and none at 10 MS/s,
      whereas the standard requires symbol timing better than 50 ppm.
"""

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

# Advertising channel access address (Bluetooth Core Spec 5.x, §2.1.2)
BTLE_ACCESS_ADDRESS: int = 0x8E89BED6

BTLE_SYMBOL_RATE_HZ: float = 1.0e6
"""LE 1M PHY symbol rate, in symbols per second.

Bluetooth Core Specification v5.4, Vol 6 (Low Energy Controller), Part A
(Physical Layer Specification), Section 1: "The symbol rate is 1 Msym/s."
Section 3.1 requires a symbol timing accuracy better than +/-50 ppm.
"""

_PREAMBLE_BITS: np.ndarray = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)  # 0xAA, MSB first

_AA_BITS: np.ndarray = np.unpackbits(np.array([0x8E, 0x89, 0xBE, 0xD6], dtype=np.uint8)).astype(np.float64)

_BLE_GFSK_BT: float = 0.5
_BLE_MOD_INDEX: float = 0.5


def _gaussian_pulse(samples_per_symbol: int, bt: float) -> np.ndarray:
    """Gaussian frequency pulse for GFSK."""
    m = 2
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / samples_per_symbol) ** 2)
    return p / np.sum(p)


def build_btle_bit_stream(num_bits: int, rng: np.random.Generator, *, inter_frame_gap_symbols: int = 0) -> np.ndarray:
    """Builds a stream of BLE advertising packets as bipolar symbols.

    Every packet is followed by ``inter_frame_gap_symbols`` zero symbols, so
    successive packets start ``packet length + inter_frame_gap_symbols``
    symbols apart. A zero symbol carries no frequency deviation, and
    ``btle_modulator_baseband`` switches the carrier off for it.

    Args:
        num_bits: Total symbols to produce, gap symbols included.
        rng: Random number generator.
        inter_frame_gap_symbols: Zero symbols inserted after every packet,
            before pulse shaping. Defaults to 0, which concatenates packets
            back to back.

    Returns:
        np.ndarray: Stream of {-1, +1} packet symbols and 0 gap symbols,
        length num_bits.

    Raises:
        TypeError: If inter_frame_gap_symbols is not an integer.
        ValueError: If num_bits is not positive or inter_frame_gap_symbols is
            negative.
    """
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")
    if isinstance(inter_frame_gap_symbols, bool) or not isinstance(inter_frame_gap_symbols, (int, np.integer)):
        raise TypeError(f"inter_frame_gap_symbols must be an integer, got {inter_frame_gap_symbols!r}")
    if inter_frame_gap_symbols < 0:
        raise ValueError(f"inter_frame_gap_symbols must be non-negative, got {inter_frame_gap_symbols}")

    gap = np.zeros(inter_frame_gap_symbols)
    frames = []
    num_symbols = 0
    while num_symbols < num_bits:
        # PDU: 2-byte header + random 0–37 byte payload
        pdu_payload_bytes = int(rng.integers(0, 38))
        pdu_bytes = 2 + pdu_payload_bytes
        pdu_bits = np.unpackbits(rng.integers(0, 256, pdu_bytes, dtype=np.uint8)).astype(np.float64)
        crc_bits = np.unpackbits(rng.integers(0, 256, 3, dtype=np.uint8)).astype(np.float64)
        packet = np.concatenate([_PREAMBLE_BITS, _AA_BITS, pdu_bits, crc_bits])
        frames.extend((2.0 * packet - 1.0, gap))  # {0,1} → {-1,+1}, then the gap
        num_symbols += len(packet) + inter_frame_gap_symbols

    return np.concatenate(frames)[:num_bits]


def btle_modulator_baseband(
    max_num_samples: int,
    oversampling_rate_nominal: int,
    rng: np.random.Generator | None = None,
    *,
    inter_frame_gap_symbols: int = 0,
) -> np.ndarray:
    """BLE GFSK modulator at complex baseband.

    Args:
        max_num_samples: Maximum output samples.
        oversampling_rate_nominal: Samples per bit at baseband.
        rng: Random number generator.
        inter_frame_gap_symbols: Silent symbols after every packet; see
            ``build_btle_bit_stream``. The envelope is the on/off symbol
            sequence shaped by the same pulse as the frequency, so the carrier
            ramps down and up over about one symbol at each gap and has unit
            amplitude inside a packet. Defaults to 0, which keeps the envelope
            constant.

    Returns:
        np.ndarray: Baseband BLE signal, exactly max_num_samples long.

    Raises:
        TypeError: If inter_frame_gap_symbols is not an integer.
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive,
            or inter_frame_gap_symbols is negative.
    """
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    if rng is None:
        rng = np.random.default_rng()

    sps = oversampling_rate_nominal
    rect = np.ones(sps)
    gauss = _gaussian_pulse(sps, _BLE_GFSK_BT)
    pulse_shape = sp.convolve(gauss, rect)

    max_minus_filter = max_num_samples - len(pulse_shape) + 1
    num_bits = max(1, int(np.floor(max_minus_filter / sps)))

    symbols = build_btle_bit_stream(num_bits, rng, inter_frame_gap_symbols=inter_frame_gap_symbols)
    freq = sp.upfirdn(pulse_shape, symbols, up=sps, down=1)
    phase = np.cumsum(freq) * (np.pi * _BLE_MOD_INDEX / np.sum(rect))
    modulated = np.exp(1j * phase)
    if inter_frame_gap_symbols > 0:
        # The carrier is off for gap symbols. Every polyphase branch of
        # pulse_shape sums to one, so the envelope is one inside a packet.
        modulated *= sp.upfirdn(pulse_shape, np.abs(symbols), up=sps, down=1)

    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)

    return modulated


def btle_modulator(
    bandwidth: float,
    sample_rate: float,
    num_samples: int,
    rng: np.random.Generator | None = None,
    *,
    pin_rate_to_standard: bool = False,
    inter_frame_gap_symbols: int = 0,
) -> np.ndarray:
    """BLE GFSK modulator: builds packet stream and resamples to the target symbol rate.

    By default the symbol rate equals ``bandwidth`` (one symbol per second per
    hertz), so the waveform scales with the requested bandwidth. With
    ``pin_rate_to_standard=True`` the symbol rate is ``BTLE_SYMBOL_RATE_HZ``
    and ``bandwidth`` does not affect the waveform.

    The standard rate is representable when it meets the rule this builder
    applies to ``bandwidth``: ``BTLE_SYMBOL_RATE_HZ <= sample_rate / 2``, that
    is, ``sample_rate >= 2 MHz``. The multistage polyphase resampler accepts
    any positive rate change, so it adds no further constraint.

    Args:
        bandwidth: Desired signal bandwidth (Hz). Validated in both modes; it
            sets the symbol rate only when pin_rate_to_standard is False.
        sample_rate: Capture sampling rate (Hz).
        num_samples: Number of IQ samples to produce.
        rng: Random number generator.
        pin_rate_to_standard: If True, generate at the standard LE 1M symbol
            rate, ``BTLE_SYMBOL_RATE_HZ``, instead of deriving the rate from
            ``bandwidth``. Defaults to False.
        inter_frame_gap_symbols: Silent symbols inserted after every packet;
            see ``btle_modulator_baseband``. Defaults to 0.

    Returns:
        np.ndarray: BLE IQ at the scaled or standard symbol rate, length num_samples.

    Raises:
        TypeError: If inter_frame_gap_symbols is not an integer.
        ValueError: If bandwidth or sample_rate are not positive, bandwidth > sample_rate/2,
            num_samples is not positive, inter_frame_gap_symbols is negative, or
            pin_rate_to_standard is True and BTLE_SYMBOL_RATE_HZ > sample_rate/2.
    """
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if bandwidth > sample_rate / 2:
        raise ValueError("bandwidth must be less than sample_rate/2")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if pin_rate_to_standard and sample_rate < 2 * BTLE_SYMBOL_RATE_HZ:
        raise ValueError(
            f"btle: pin_rate_to_standard=True requires the standard symbol rate of {BTLE_SYMBOL_RATE_HZ} Hz "
            f"to be at most sample_rate/2, but sample_rate is {sample_rate} Hz (need sample_rate >= {2 * BTLE_SYMBOL_RATE_HZ} Hz)"
        )

    if rng is None:
        rng = np.random.default_rng()

    symbol_rate = BTLE_SYMBOL_RATE_HZ if pin_rate_to_standard else bandwidth
    oversampling_rate_nominal = 4
    oversampling_rate = sample_rate / symbol_rate
    resample_rate_ideal = oversampling_rate / oversampling_rate_nominal
    max_num_samples = max(oversampling_rate_nominal, int(np.floor(num_samples / resample_rate_ideal)))

    baseband = btle_modulator_baseband(max_num_samples, oversampling_rate_nominal, rng, inter_frame_gap_symbols=inter_frame_gap_symbols)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal

    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


class BTLESignalGenerator(BaseSignalGenerator):
    """Bluetooth Low Energy Signal Generator.

    Builds structured BLE advertising packets: 0xAA preamble, 0x8E89BED6
    access address, random PDU, random CRC; GFSK modulated at BT=0.5, h=0.5.
    """

    def __init__(self, **kwargs: dict[str, str | float | int]) -> None:
        """Initializes the BLE Signal Generator.

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
        self.set_default_class_name("btle")

    def generate(self) -> Signal:
        """Generates a BLE signal.

        Returns:
            Signal: Generated BLE signal with metadata.
        """
        sample_rate = self["sample_rate"]
        num_iq_samples_signal = self.random_generator.integers(
            low=self["signal_duration_in_samples_min"],
            high=self["signal_duration_in_samples_max"] + 1,
        )
        bandwidth = self.random_generator.integers(low=self["bandwidth_min"], high=self["bandwidth_max"] + 1)
        signal_data = btle_modulator(bandwidth, sample_rate, num_iq_samples_signal, self.random_generator)
        return Signal(data=signal_data, center_freq=0, bandwidth=bandwidth)
