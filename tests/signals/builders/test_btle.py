"""Unit tests for Bluetooth Low Energy (btle) signal builder."""

import re

import numpy as np
import pytest
import scipy.signal as sp
from test_builders_utils import estimate_symbol_rate

from torchsig.signals.builders.btle import (
    _AA_BITS,
    _BLE_GFSK_BT,
    _BLE_MOD_INDEX,
    _PREAMBLE_BITS,
    BTLE_ACCESS_ADDRESS,
    BTLE_SYMBOL_RATE_HZ,
    BTLESignalGenerator,
    _gaussian_pulse,
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
# Timing modes: default output unchanged, standard symbol rate, inter-frame gaps
# ---------------------------------------------------------------------------

# Preamble plus access address, the first 40 symbols of every packet.
HEADER_SYMBOLS = 2.0 * np.concatenate([_PREAMBLE_BITS, _AA_BITS]) - 1.0

# Shortest packet: preamble, access address, 2-byte PDU header and CRC.
MIN_PACKET_SYMBOLS = 8 + 32 + 16 + 24


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


def _header_template(samples_per_symbol):
    """GFSK waveform of the 40 header symbols at an integer number of samples per symbol."""
    pulse = np.convolve(_gaussian_pulse(samples_per_symbol, _BLE_GFSK_BT), np.ones(samples_per_symbol))
    phase = np.cumsum(sp.upfirdn(pulse, HEADER_SYMBOLS, up=samples_per_symbol)) * (np.pi * _BLE_MOD_INDEX / samples_per_symbol)
    # The Gaussian pulse reaches two symbols ahead, so symbol k starts at sample (k + 2) * samples_per_symbol.
    first = 2 * samples_per_symbol
    return np.exp(1j * phase[first : first + len(HEADER_SYMBOLS) * samples_per_symbol])


def _packet_starts(num_symbols, seed, inter_frame_gap_symbols):
    """Symbol indices at which packets start in the stream drawn for ``seed``.

    btle_modulator draws its packets from rng through build_btle_bit_stream alone, and
    the packet sequence does not depend on the requested length, so the same seed
    reproduces the packets of any btle_modulator call.
    """
    stream = build_btle_bit_stream(num_symbols, np.random.default_rng(seed), inter_frame_gap_symbols=inter_frame_gap_symbols)
    windows = np.lib.stride_tricks.sliding_window_view(stream, len(HEADER_SYMBOLS))
    return np.flatnonzero(np.all(windows == HEADER_SYMBOLS, axis=1))


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
    explicit = btle_modulator(bandwidth, sample_rate, num_samples, np.random.default_rng(seed), pin_rate_to_standard=False, inter_frame_gap_symbols=0)
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


@pytest.mark.parametrize(("sample_rate", "pin_rate_to_standard"), [(10_000_000, True), (7_680_000, True), (10_000_000, False)])
@pytest.mark.parametrize("inter_frame_gap_symbols", [16, 150])
def test_inter_frame_gap_duty_cycle(sample_rate, pin_rate_to_standard, inter_frame_gap_symbols):
    """Criterion 4: the silent fraction is within 20 % of g / (g + frame length).

    The expected fraction is the share of gap symbols in the stream drawn for this
    seed, which is g / (g + frame length) averaged over the packet lengths drawn.
    The shaped envelope stays above 1 % of its peak for about 0.6 symbol at each end
    of a gap (2.3 standard deviations of the BT = 0.5 Gaussian pulse, whose standard
    deviation is 0.265 symbol), so the measured fraction falls short by about 1.2 / g:
    under 8 % at g = 16 and under 1 % at g = 150. Gaps of 150 symbols at 1 Msym/s
    are the standard's 150 microsecond inter frame space.
    """
    num_samples = 2**16
    bandwidth = 250_000
    seed = 3
    iq = btle_modulator(
        bandwidth,
        sample_rate,
        num_samples,
        np.random.default_rng(seed),
        pin_rate_to_standard=pin_rate_to_standard,
        inter_frame_gap_symbols=inter_frame_gap_symbols,
    )
    envelope = np.abs(iq)
    burst_rms = np.sqrt(np.mean(envelope**2))
    measured = np.mean(envelope < 0.01 * burst_rms)

    symbol_rate = BTLE_SYMBOL_RATE_HZ if pin_rate_to_standard else bandwidth
    num_symbols = round(num_samples * symbol_rate / sample_rate)
    stream = build_btle_bit_stream(num_symbols, np.random.default_rng(seed), inter_frame_gap_symbols=inter_frame_gap_symbols)
    expected = np.mean(stream == 0)
    assert abs(measured - expected) <= 0.2 * expected, f"silent fraction {measured:.4f}, expected {expected:.4f}"


@pytest.mark.parametrize("sample_rate", [10_000_000, 20_000_000])
@pytest.mark.parametrize("inter_frame_gap_symbols", [0, 16, 150])
def test_frame_period_matches_standard(sample_rate, inter_frame_gap_symbols):
    """Criterion 5: successive preamble correlation peaks are one packet plus gap apart.

    Each interval must equal (preamble + access address + PDU + CRC + gap) symbols
    divided by BTLE_SYMBOL_RATE_HZ, within one symbol; the packet lengths come from
    the stream drawn for the same seed. The correlation with the 40 header symbols is
    scaled so that a matching header reads 1. Each header symbol that differs shifts
    the phase difference by pi, flipping the sign of every later contribution, so
    random data reads about |sum of 40 random signs| / 40. Reaching the 0.8 threshold
    takes 36 agreeing symbols out of 40, with probability 1.9e-7 per position, or
    about 6e-4 per record here.
    """
    samples_per_symbol = round(sample_rate / BTLE_SYMBOL_RATE_HZ)
    num_symbols = 3000
    seed = 4
    iq = btle_modulator(
        250_000,
        sample_rate,
        num_symbols * samples_per_symbol,
        np.random.default_rng(seed),
        pin_rate_to_standard=True,
        inter_frame_gap_symbols=inter_frame_gap_symbols,
    )

    envelope = np.abs(iq)
    amplitude = np.median(envelope[envelope > 0.5 * envelope.max()])
    template = _header_template(samples_per_symbol)
    correlation = np.abs(np.correlate(iq, template, mode="valid")) / (amplitude * len(template))
    peaks, _ = sp.find_peaks(correlation, height=0.8, distance=MIN_PACKET_SYMBOLS * samples_per_symbol)
    assert len(peaks) >= 5, f"found only {len(peaks)} packet headers"

    starts = _packet_starts(num_symbols + 1000, seed, inter_frame_gap_symbols)
    first = int(np.argmin(np.abs(starts * samples_per_symbol - peaks[0])))
    measured = np.diff(peaks) / sample_rate
    expected = np.diff(starts)[first : first + len(measured)] / BTLE_SYMBOL_RATE_HZ
    np.testing.assert_allclose(measured, expected, rtol=0, atol=1 / BTLE_SYMBOL_RATE_HZ)


@pytest.mark.parametrize(("pin_rate_to_standard", "inter_frame_gap_symbols"), [(False, 0), (True, 0), (False, 16), (True, 150)])
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
def test_dtype_length_finite(pin_rate_to_standard, inter_frame_gap_symbols, bandwidth, sample_rate, num_samples):
    """Every mode returns finite complex IQ of exactly the requested length."""
    iq = btle_modulator(
        bandwidth,
        sample_rate,
        num_samples,
        np.random.default_rng(11),
        pin_rate_to_standard=pin_rate_to_standard,
        inter_frame_gap_symbols=inter_frame_gap_symbols,
    )
    assert iq.dtype == TorchSigComplexDataType
    assert iq.shape == (num_samples,)
    assert np.all(np.isfinite(iq))


def test_bit_stream_inter_frame_gap():
    """Gap zeros follow every packet and leave the packets themselves unchanged."""
    gap = 7
    plain = build_btle_bit_stream(3000, np.random.default_rng(3))
    gapped = build_btle_bit_stream(3000, np.random.default_rng(3), inter_frame_gap_symbols=gap)
    assert len(gapped) == 3000

    packets = gapped[gapped != 0]
    np.testing.assert_array_equal(packets, plain[: len(packets)])

    is_gap = np.concatenate(([0], (gapped == 0).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(is_gap))
    run_starts, run_ends = edges[0::2], edges[1::2]
    assert np.all(run_ends[:-1] - run_starts[:-1] == gap)
    assert 0 < run_ends[-1] - run_starts[-1] <= gap
    for end in run_ends[run_ends + len(HEADER_SYMBOLS) <= len(gapped)]:
        np.testing.assert_array_equal(gapped[end : end + len(HEADER_SYMBOLS)], HEADER_SYMBOLS)


@pytest.mark.parametrize(("gap", "error"), [(-1, ValueError), (2.5, TypeError), (True, TypeError)])
def test_inter_frame_gap_invalid_raises(gap, error):
    """A negative or non-integer gap is rejected."""
    with pytest.raises(error, match="inter_frame_gap_symbols"):
        btle_modulator(1_000_000, 10_000_000, 4096, np.random.default_rng(0), inter_frame_gap_symbols=gap)
