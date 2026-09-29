"""Unit tests for Bluetooth Low Energy (btle) signal builder."""

import re

import numpy as np
import pytest
import scipy.signal as sp
from test_builders_utils import estimate_symbol_rate

from torchsig.signals.builders.btle import (
    BTLE_ACCESS_ADDRESS,
    BTLE_SYMBOL_RATE_HZ,
    BTLESignalGenerator,
    btle_modulator,
    build_btle_bit_stream,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import (
    TorchSigComplexDataType,
    multistage_polyphase_resampler,
    pad_head_tail_to_length,
    slice_head_tail_to_length,
    slice_tail_to_length,
)
from torchsig.utils.signal_building import lookup_signal_generator_by_string

BTLE_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 1_000_000,
    "bandwidth_max": 2_000_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_btle_access_address():
    """BTLE advertising access address matches the spec value."""
    assert BTLE_ACCESS_ADDRESS == 0x8E89BED6


def test_btle_bit_stream_contains_access_address():
    """The bit stream begins with the 8-bit preamble then the 32-bit access address."""
    from torchsig.signals.builders.btle import _PREAMBLE_BITS

    rng = np.random.default_rng(0)
    stream = build_btle_bit_stream(100, rng)
    assert len(stream) == 100
    # Check preamble (bipolar): 0xAA = 10101010 → bipolar [-1,+1,-1,+1,...]
    preamble_bipolar = 2.0 * _PREAMBLE_BITS - 1.0
    np.testing.assert_array_equal(stream[:8], preamble_bipolar)


def test_btle_modulator_output():
    """The modulator returns finite complex IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = btle_modulator(1_000_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_btle_modulator_invalid_args():
    """Invalid bandwidth/sample-rate raise."""
    with pytest.raises(ValueError):
        btle_modulator(0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        btle_modulator(6_000_000, 10_000_000, 4096)


def test_btle_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = BTLESignalGenerator(metadata=BTLE_METADATA, seed=1)()
    assert signal.class_name == "btle"
    assert len(signal.data) == BTLE_METADATA["signal_duration_in_samples_min"]


def test_btle_generator_reproducible():
    """Same seed yields identical IQ."""
    a = BTLESignalGenerator(metadata=BTLE_METADATA, seed=2).generate()
    b = BTLESignalGenerator(metadata=BTLE_METADATA, seed=2).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_btle_registered_and_in_signal_lists():
    """'btle' resolves through lookup and belongs to the 'bluetooth' family."""
    assert isinstance(lookup_signal_generator_by_string("btle"), BTLESignalGenerator)
    assert CLASS_FAMILY_DICT["btle"] == "bluetooth"
    lists = TorchSigSignalLists()
    assert "btle" in lists.bluetooth_signals


# ---------------------------------------------------------------------------
# Timing modes: default output unchanged, standard symbol rate
# ---------------------------------------------------------------------------

# Golden reference for criterion 1: the default code path exactly as released in
# TorchSig 2.2.0 (commit 1197261), with its constants frozen. A stored golden array
# cannot be compared bit for bit across platforms, because libm, BLAS and FMA
# contraction round differently on x86-64 and aarch64, so the frozen implementation
# is run on the test platform and its output is the golden array.
_PREAMBLE_BITS_2_2_0 = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
_AA_BITS_2_2_0 = np.unpackbits(np.array([0x8E, 0x89, 0xBE, 0xD6], dtype=np.uint8)).astype(np.float64)


def _gaussian_pulse_2_2_0(samples_per_symbol, bt):
    m = 2
    n = np.arange(-m * samples_per_symbol, m * samples_per_symbol + 1)
    p = np.exp(-2 * np.pi**2 * bt**2 / np.log(2) * (n / samples_per_symbol) ** 2)
    return p / np.sum(p)


def _build_btle_bit_stream_2_2_0(num_bits, rng):
    bits = []
    while sum(len(b) for b in bits) < num_bits:
        pdu_payload_bytes = int(rng.integers(0, 38))
        pdu_bytes = 2 + pdu_payload_bytes
        pdu_bits = np.unpackbits(rng.integers(0, 256, pdu_bytes, dtype=np.uint8)).astype(np.float64)
        crc_bits = np.unpackbits(rng.integers(0, 256, 3, dtype=np.uint8)).astype(np.float64)
        bits.append(np.concatenate([_PREAMBLE_BITS_2_2_0, _AA_BITS_2_2_0, pdu_bits, crc_bits]))
    stream = np.concatenate(bits)[:num_bits]
    return 2.0 * stream - 1.0


def _btle_modulator_baseband_2_2_0(max_num_samples, oversampling_rate_nominal, rng):
    sps = oversampling_rate_nominal
    rect = np.ones(sps)
    gauss = _gaussian_pulse_2_2_0(sps, 0.5)
    pulse_shape = sp.convolve(gauss, rect)
    max_minus_filter = max_num_samples - len(pulse_shape) + 1
    num_bits = max(1, int(np.floor(max_minus_filter / sps)))
    symbols = _build_btle_bit_stream_2_2_0(num_bits, rng)
    freq = sp.upfirdn(pulse_shape, symbols, up=sps, down=1)
    phase = np.cumsum(freq) * (np.pi * 0.5 / np.sum(rect))
    modulated = np.exp(1j * phase)
    if len(modulated) > max_num_samples:
        modulated = slice_tail_to_length(modulated, max_num_samples)
    elif len(modulated) < max_num_samples:
        modulated = pad_head_tail_to_length(modulated, max_num_samples)
    return modulated


def _btle_modulator_2_2_0(bandwidth, sample_rate, num_samples, rng):
    oversampling_rate_nominal = 4
    oversampling_rate = sample_rate / bandwidth
    resample_rate_ideal = oversampling_rate / oversampling_rate_nominal
    max_num_samples = max(oversampling_rate_nominal, int(np.floor(num_samples / resample_rate_ideal)))
    baseband = _btle_modulator_baseband_2_2_0(max_num_samples, oversampling_rate_nominal, rng)
    correct_bw = multistage_polyphase_resampler(baseband, resample_rate_ideal)
    correct_bw *= 1 / resample_rate_ideal
    correct_bw = slice_head_tail_to_length(correct_bw, num_samples) if len(correct_bw) > num_samples else pad_head_tail_to_length(correct_bw, num_samples)
    return correct_bw.astype(TorchSigComplexDataType)


@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "num_samples"),
    [
        (1_000_000, 10_000_000, 4096),  # interpolate by 2.5
        (62_500, 10_000_000, 20_000),  # interpolate by 40
        (1_234_567, 10_000_000, 8191),  # fractional interpolation
        (1_000_000, 4_000_000, 777),  # no resampling
        (1_200_000, 2_500_000, 5000),  # decimate
        (5_000_000, 10_000_000, 64),  # bandwidth at sample_rate / 2
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_output_unchanged(bandwidth, sample_rate, num_samples, seed):
    """Criterion 1: the default output is bit-identical to TorchSig 2.2.0 for a fixed seed."""
    golden = _btle_modulator_2_2_0(bandwidth, sample_rate, num_samples, np.random.default_rng(seed))
    omitted = btle_modulator(bandwidth, sample_rate, num_samples, np.random.default_rng(seed))
    explicit = btle_modulator(bandwidth, sample_rate, num_samples, np.random.default_rng(seed), pin_rate_to_standard=False)
    for iq in (omitted, explicit):
        assert iq.dtype == golden.dtype
        # Compare bit patterns rather than values, so that -0.0 and 0.0 differ.
        np.testing.assert_array_equal(iq.view(np.uint64), golden.view(np.uint64))


@pytest.mark.parametrize("sample_rate", [2_000_000, 7_680_000, 10_000_000, 20_000_000])
def test_standard_rate_symbol_rate(sample_rate):
    """Criterion 2: pinned to the standard, the measured symbol rate is 1 Msym/s within 0.1 %.

    The same 250 kHz request gives 250 kBd in the default mode, which the same
    measurement confirms. Tolerance: the estimate lies on the FFT grid, within one
    step, sample_rate / 2**16, of the realized rate, and the resampler realizes the
    rate within 100 ppm (exactly at 2, 10 and 20 MS/s). For the standard rate that is
    at most 305 + 100 Hz, inside the 1 kHz of the criterion. 2 MS/s is the lowest
    sample rate at which the standard rate is representable.
    """
    num_samples = 2**16
    bandwidth = 250_000
    grid_step = sample_rate / num_samples

    scaled = btle_modulator(bandwidth, sample_rate, num_samples, np.random.default_rng(7))
    assert abs(estimate_symbol_rate(scaled, sample_rate) - bandwidth) <= grid_step + 1e-4 * bandwidth

    pinned = btle_modulator(bandwidth, sample_rate, num_samples, np.random.default_rng(7), pin_rate_to_standard=True)
    assert abs(estimate_symbol_rate(pinned, sample_rate) - BTLE_SYMBOL_RATE_HZ) <= grid_step + 1e-4 * BTLE_SYMBOL_RATE_HZ


@pytest.mark.parametrize("sample_rate", [250_000, 1_500_000, 1_999_999])
def test_standard_rate_unrepresentable_raises(sample_rate):
    """Criterion 3: below 2 MS/s the standard rate is not representable and pinning raises.

    The message names the class, the required rate and the sample rate. The same
    request in the default mode is valid, because the rate then follows the bandwidth.
    """
    bandwidth = sample_rate / 4
    expected = rf"^btle: .*{re.escape(str(BTLE_SYMBOL_RATE_HZ))} Hz.*sample_rate is {sample_rate} Hz"
    with pytest.raises(ValueError, match=expected):
        btle_modulator(bandwidth, sample_rate, 1024, np.random.default_rng(0), pin_rate_to_standard=True)
    assert len(btle_modulator(bandwidth, sample_rate, 1024, np.random.default_rng(0))) == 1024


@pytest.mark.parametrize("pin_rate_to_standard", [False, True])
@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "num_samples"),
    [
        (1_000_000, 10_000_000, 4096),
        (62_500, 2_000_000, 1000),
        (2_500_000, 7_680_000, 12_345),
        (5_000_000, 10_000_000, 1),
        (400_000, 4_000_000, 7),
    ],
)
def test_dtype_length_finite(pin_rate_to_standard, bandwidth, sample_rate, num_samples):
    """Every mode returns finite complex IQ of exactly the requested length."""
    iq = btle_modulator(
        bandwidth,
        sample_rate,
        num_samples,
        np.random.default_rng(11),
        pin_rate_to_standard=pin_rate_to_standard,
    )
    assert iq.dtype == TorchSigComplexDataType
    assert iq.shape == (num_samples,)
    assert np.all(np.isfinite(iq))
