"""Unit tests for Cellular type signal builders."""

import numpy as np
import pytest

from torchsig.signals.builders.cellular import (
    GSM_TSC,
    GSMSignalGenerator,
    build_gsm_bit_stream,
    build_gsm_burst,
    gsm_differential_encode,
    gsm_gmsk_modulate,
    gsm_modulator,
    gsm_modulator_baseband,
    gsm_timeslot_guard_bits,
)
from torchsig.signals.signal_lists import CLASS_FAMILY_DICT, TorchSigSignalLists
from torchsig.utils.dsp import TorchSigComplexDataType
from torchsig.utils.signal_building import lookup_signal_generator_by_string

# --------------------------------------------------------------------------- #
# GSM
# --------------------------------------------------------------------------- #
GSM_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 1_500_000,
    "bandwidth_max": 2_500_000,
    "signal_duration_in_samples_min": 4096,
    "signal_duration_in_samples_max": 4096,
}


def test_gsm_tsc_count_and_length():
    """There are 8 TSCs, each 26 bits."""
    assert len(GSM_TSC) == 8
    assert all(len(t) == 26 for t in GSM_TSC)


# TSC Set 1, 3GPP TS 45.002 V15.1.0 Table 5.2.3a
TS_45002_TSC_SET_1 = (
    "00100101110000100010010111",
    "00101101110111100010110111",
    "01000011101110100100001110",
    "01000111101101000100011110",
    "00011010111001000001101011",
    "01001110101100000100111010",
    "10100111110110001010011111",
    "11101111000100101110111100",
)


def _tsc_positions(bits):
    """Start index of every occurrence of each TSC in a bit stream."""
    windows = np.lib.stride_tricks.sliding_window_view(bits, 26)
    return [np.flatnonzero(np.all(windows == np.array(tsc), axis=1)) for tsc in GSM_TSC]


def test_gsm_tsc_table_matches_ts_45002():
    """GSM_TSC is TSC Set 1 of TS 45.002."""
    assert tuple("".join(map(str, t)) for t in GSM_TSC) == TS_45002_TSC_SET_1


def test_gsm_burst_length_and_values():
    """A timeslot is 148 burst bits plus guard bits, with tails, TSC and dummy-bit guard in place."""
    rng = np.random.default_rng(0)
    burst = build_gsm_burst(3, rng)
    assert len(burst) == 156
    assert set(np.unique(burst)).issubset({0, 1})
    np.testing.assert_array_equal(burst[:3], 0)
    np.testing.assert_array_equal(burst[61:87], GSM_TSC[3])  # 3 tail + 57 data + 1 steal = 61
    np.testing.assert_array_equal(burst[145:148], 0)
    np.testing.assert_array_equal(burst[148:], 1)
    assert len(build_gsm_burst(3, rng, num_guard_bits=9)) == 157


def test_gsm_differential_encoding_follows_ts_45004():
    """alpha_i = 1 - 2 * (d_i XOR d_{i-1}), with the dummy bit d_{-1} = 1 before the stream."""
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    expected = np.array([-1, 1, 1, -1, 1, -1, -1, -1, 1, -1], dtype=float)
    np.testing.assert_array_equal(gsm_differential_encode(bits), expected)


@pytest.mark.parametrize("sps", [2, 4, 8])
def test_gsm_instantaneous_frequency_sign_matches_alpha(sps):
    """The frequency at each bit centre has the sign of the TS 45.004 alpha_i: +1 raises the phase."""
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    expected_alpha = np.array([-1, 1, 1, -1, 1, -1, -1, -1, 1, -1])
    iq = gsm_gmsk_modulate(bits, sps)
    inst_freq = np.angle(iq[1:] * np.conj(iq[:-1]))
    # alpha_i's pulse peaks at sample i * sps + 3.5 * sps - 0.5; inst_freq[k] spans samples k to k + 1
    centres = np.round(np.arange(len(bits)) * sps + 3.5 * sps - 1.5).astype(int)
    np.testing.assert_array_equal(np.sign(inst_freq[centres]), expected_alpha)


def test_gsm_all_zero_bits_give_positive_quarter_rate_tone():
    """All-zero data (as in the FCCH) is a tone at +1/4 of the bit rate after the first bit."""
    sps = 8
    iq = gsm_gmsk_modulate(np.zeros(40, dtype=np.int8), sps)
    inst_freq = np.angle(iq[1:] * np.conj(iq[:-1]))
    np.testing.assert_allclose(inst_freq[10 * sps : 30 * sps], np.pi / (2 * sps), rtol=1e-9)


def test_gsm_timeslot_guard_bits_alternate_157_156():
    """Timeslots 0 and 4 carry 9 guard bits and the others 8: 1250 bits per TDMA frame."""
    guards = [gsm_timeslot_guard_bits(tn) for tn in range(8)]
    assert guards == [9, 8, 8, 8, 9, 8, 8, 8]
    assert sum(148 + g for g in guards) == 8 * 156.25


@pytest.mark.parametrize("seed", range(8))
def test_gsm_bit_stream_uses_one_tsc(seed):
    """Every burst of a signal carries the same TSC (one cell), one of the eight of TSC Set 1."""
    bits = build_gsm_bit_stream(40 * 157, np.random.default_rng(seed))
    positions = _tsc_positions(bits)
    used = [i for i, p in enumerate(positions) if len(p)]
    assert len(used) == 1
    assert len(positions[used[0]]) >= 39


@pytest.mark.parametrize("seed", range(8))
def test_gsm_bit_stream_timeslots_average_156_25_bits(seed):
    """Burst starts are 156 or 157 bits apart, with every 8 consecutive timeslots spanning 1250 bits."""
    bits = build_gsm_bit_stream(80 * 157, np.random.default_rng(seed))
    starts = next(p for p in _tsc_positions(bits) if len(p))
    spacing = np.diff(starts)
    assert set(spacing.tolist()) == {156, 157}
    frame_spans = np.convolve(spacing, np.ones(8, dtype=int), mode="valid")
    np.testing.assert_array_equal(frame_spans, 1250)
    np.testing.assert_allclose(spacing[: len(spacing) // 8 * 8].mean(), 156.25)


@pytest.mark.parametrize("sps", [2, 4, 8])
def test_gsm_carrier_has_no_gap_at_burst_boundaries(sps):
    """The BTS carrier has constant envelope and continuous phase through every guard period."""
    iq = gsm_modulator_baseband(30 * 157 * sps, sps, np.random.default_rng(5))
    interior = iq[8 * sps : -8 * sps]  # skip the zero padding at the ends
    np.testing.assert_allclose(np.abs(interior), 1.0, rtol=1e-9)
    phase_step = np.abs(np.angle(interior[1:] * np.conj(interior[:-1])))
    assert phase_step.max() <= np.pi / (2 * sps) + 1e-9


def test_gsm_modulator_output():
    """The modulator returns finite complex IQ of the right length."""
    rng = np.random.default_rng(42)
    num_samples = 4096
    iq = gsm_modulator(2_000_000, 10_000_000, num_samples, rng)
    assert iq.dtype == TorchSigComplexDataType
    assert len(iq) == num_samples
    assert np.all(np.isfinite(iq))


def test_gsm_modulator_invalid_args():
    """Invalid bandwidth/sample-rate raise."""
    with pytest.raises(ValueError):
        gsm_modulator(0, 10_000_000, 4096)
    with pytest.raises(ValueError):
        gsm_modulator(6_000_000, 10_000_000, 4096)


def test_gsm_generator_generate():
    """Generator produces a Signal with correct metadata."""
    signal = GSMSignalGenerator(metadata=GSM_METADATA, seed=1)()
    assert signal.class_name == "gsm"
    assert len(signal.data) == GSM_METADATA["signal_duration_in_samples_min"]


def test_gsm_generator_reproducible():
    """Same seed yields identical IQ."""
    a = GSMSignalGenerator(metadata=GSM_METADATA, seed=4).generate()
    b = GSMSignalGenerator(metadata=GSM_METADATA, seed=4).generate()
    np.testing.assert_array_equal(a.data, b.data)


def test_gsm_registered_and_in_signal_lists():
    """'gsm' resolves through lookup and is in the 'cellular' family."""
    assert isinstance(lookup_signal_generator_by_string("gsm"), GSMSignalGenerator)
    assert CLASS_FAMILY_DICT["gsm"] == "cellular"
    lists = TorchSigSignalLists()
    assert "gsm" in lists.cellular_signals
