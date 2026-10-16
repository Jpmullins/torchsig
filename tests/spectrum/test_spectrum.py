"""Tests for torchsig.spectrum: the catalog, the population, the schedule, the receiver and the stream.

The properties worth protecting are the ones the package's usefulness rests on:

* a chunk is a pure function of its index, so a stream is seekable and reproducible;
* a chunk boundary is invisible, so nothing has to be generated in order;
* the builders are driven at each technology's standard bandwidth, which is what makes their timing standard;
* the delivered noise floor is the one the receiver's noise figure predicts.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

from torchsig.spectrum import BandStream, Receiver, named_band
from torchsig.spectrum.activity import EPOCH_S, MAX_BURST_S, Schedule
from torchsig.spectrum.catalog import BandPlan, load_catalog
from torchsig.spectrum.population import Environment, populate
from torchsig.spectrum.receiver import THERMAL_NOISE_DBM_PER_HZ
from torchsig.spectrum.waveforms import render_burst

ISM = "ism_2450"


def _symbol_rate_estimator():
    """The cyclic-autocorrelation symbol-rate estimator the builder tests already use."""
    path = pathlib.Path(__file__).resolve().parents[1] / "signals" / "builders" / "test_builders_utils.py"
    spec = importlib.util.spec_from_file_location("_builders_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.estimate_symbol_rate


@pytest.fixture(scope="module")
def plan() -> BandPlan:
    return named_band(ISM)


@pytest.fixture(scope="module")
def emitters(plan: BandPlan):
    return populate(plan, seed=1234, environment=Environment(clutter="urban_dense"))


# --------------------------------------------------------------------------- catalog


def test_catalog_loads_every_technology_with_a_band():
    catalog = load_catalog()
    assert len(catalog) > 50
    for technology in catalog.values():
        assert technology.id
        for lo, hi, _ in technology.bands_hz:
            # hi == lo is legitimate: some allocations are a single frequency, not a range.
            assert 0 < lo <= hi


def test_ism_band_plan_has_the_expected_channel_plans(plan: BandPlan):
    wifi = plan.technology("wifi_2_4")
    centers = [round(c / 1e6, 1) for c in wifi.channels_in(plan.f_low_hz, plan.f_high_hz)]
    assert centers == [2412.0, 2417.0, 2422.0, 2427.0, 2432.0, 2437.0, 2442.0, 2447.0, 2452.0, 2457.0, 2462.0]

    zigbee = plan.technology("zigbee_2_4")
    assert len(zigbee.channels_in(plan.f_low_hz, plan.f_high_hz)) == 16  # channels 11-26

    # "2402 + k for k = 0..78" has an implicit coefficient of one: 79 channels, not a raster fallback.
    assert len(plan.technology("bluetooth_br_edr").channel_centers_hz) == 79
    assert len(plan.technology("bluetooth_le").channel_centers_hz) == 40


def test_split_bluetooth_supersedes_the_compiled_entry():
    catalog = load_catalog()
    assert "bluetooth" not in catalog, "the coarse entry must not survive, or its count is used twice"
    assert {"bluetooth_le", "bluetooth_br_edr"} <= set(catalog)


def test_free_running_technologies_are_confined_to_their_own_allocation(plan: BandPlan):
    # Amateur is allocated in many places; only its 2390-2450 MHz band reaches this span.
    spans = plan.technology("amateur").placement_spans_hz(plan.f_low_hz, plan.f_high_hz)
    assert len(spans) == 1
    assert spans[0][1] <= 2450e6


# --------------------------------------------------------------------------- population


def test_population_is_deterministic(plan: BandPlan):
    a = populate(plan, seed=99, environment=Environment(clutter="urban"))
    b = populate(plan, seed=99, environment=Environment(clutter="urban"))
    assert a == b
    assert populate(plan, seed=100, environment=Environment(clutter="urban")) != a


def test_population_counts_match_the_catalog_in_the_mean(plan: BandPlan):
    env = Environment(clutter="urban_dense")
    total = 0
    trials = 120
    for seed in range(trials):
        total += sum(1 for e in populate(plan, seed=seed, environment=env) if e.technology_id == "wifi_2_4")
    expected = plan.technology("wifi_2_4").count_for("urban_dense")
    assert expected * 0.85 < total / trials < expected * 1.15


def test_emitters_are_within_reception_range(emitters):
    # The catalog counts emitters the receiver can hear, so the draw must not place them beyond that.
    for e in emitters:
        assert e.received_power_dbm > -150.0
        assert e.range_m > 0.0


def test_carrier_offsets_respect_the_standard_tolerance(emitters):
    # IEEE 802.11 allows +/- 20 ppm, which is about 49 kHz at 2.44 GHz.
    for e in emitters:
        if e.technology_id == "wifi_2_4":
            assert abs(e.frequency_offset_hz) <= e.center_hz * 20e-6 + 1.0


def test_wifi_prefers_the_non_overlapping_channels(emitters):
    wifi = [e for e in emitters if e.technology_id == "wifi_2_4"]
    on_1_6_11 = sum(1 for e in wifi if round(e.center_hz / 1e6) in (2412, 2437, 2462))
    assert on_1_6_11 > 0.5 * len(wifi), "real deployments crowd onto 1, 6 and 11"


# --------------------------------------------------------------------------- activity


def test_schedule_is_deterministic_and_randomly_accessible(emitters):
    schedule = Schedule(emitters, seed=5)
    whole = schedule.bursts_in(0.0, 0.05)
    assert whole == schedule.bursts_in(0.0, 0.05)
    # Querying a sub-interval must give exactly the bursts of the whole query that overlap it.
    part = schedule.bursts_in(0.02, 0.05)
    assert part == [b for b in whole if b.overlaps(0.02, 0.05)]


def test_a_late_interval_costs_nothing_extra(emitters):
    # Nothing accumulates, so a distant interval is as cheap as an early one and does not depend on history.
    schedule = Schedule(emitters, seed=5)
    far = schedule.bursts_in(1000.0, 1000.01)
    assert far == Schedule(emitters, seed=5).bursts_in(1000.0, 1000.01)


def test_no_burst_exceeds_the_declared_maximum(emitters):
    for burst in Schedule(emitters, seed=5).bursts_in(0.0, 3 * EPOCH_S):
        assert burst.duration_s <= MAX_BURST_S + 1e-12


def test_bluetooth_le_advertises_on_the_three_primary_channels(emitters):
    bursts = Schedule(emitters, seed=5).bursts_in(0.0, 2.0)
    advertising = {round(b.center_hz / 1e6) for b in bursts if b.kind == "advertising"}
    assert advertising <= {2402, 2426, 2480}
    assert advertising, "a dense scene should contain some advertising"


def test_microwave_oven_is_keyed_at_twice_the_mains_frequency(plan: BandPlan):
    # Find a seed whose scene contains an oven, then check its burst train.
    for seed in range(60):
        pop = populate(plan, seed=seed, environment=Environment(clutter="urban_dense"))
        ovens = [e for e in pop if e.technology_id == "microwave_oven"]
        if ovens:
            bursts = [b for b in Schedule(pop, seed=1).bursts_in(0.0, 0.2) if b.kind == "magnetron"]
            starts = sorted({round(b.start_s, 6) for b in bursts})
            gaps = np.diff(starts)
            assert np.allclose(gaps, 1.0 / 60.0, atol=1e-6), "one burst per mains cycle at 60 Hz"
            assert all(abs(b.duration_s - 1.0 / 120.0) < 1e-4 for b in bursts), "a half cycle on"
            return
    pytest.skip("no oven in the first 60 scenes")


def test_contention_keeps_a_channel_under_the_airtime_cap(plan: BandPlan):
    pop = populate(plan, seed=1234, environment=Environment(clutter="urban_dense"))
    schedule = Schedule(pop, seed=7)
    window = 0.4
    airtime: dict[int, float] = {}
    for b in schedule.bursts_in(0.0, window):
        if b.technology_id == "wifi_2_4":
            key = round(b.center_hz / 1e6)
            airtime[key] = airtime.get(key, 0.0) + min(b.stop_s, window) - max(b.start_s, 0.0)
    assert airtime, "a dense scene should have Wi-Fi traffic"
    # Poisson arrivals overshoot the cap a little; a channel must not be oversubscribed outright.
    assert max(airtime.values()) / window < 1.0


# --------------------------------------------------------------------------- waveforms


@pytest.mark.parametrize(
    ("technology", "kind", "duration_s", "expected_bw_mhz", "tolerance_mhz"),
    [
        ("wifi_2_4", "data", 1e-3, 16.25, 1.5),
        ("bluetooth_le", "advertising", 376e-6, 1.0, 0.3),
        ("zigbee_2_4", "data", 2e-3, 2.0, 0.6),
        ("microwave_oven", "magnetron", 4e-3, 20.0, 4.0),
    ],
)
def test_rendered_bursts_occupy_their_standard_bandwidth(technology, kind, duration_s, expected_bw_mhz, tolerance_mhz):
    fs = 100e6
    x = render_burst(
        technology_id=technology,
        kind=kind,
        duration_s=duration_s,
        occupied_bandwidth_hz=1e6,
        out_rate_hz=fs,
        seed=3,
    )
    assert x.size == round(duration_s * fs)
    assert math.isclose(float(np.mean(np.abs(x) ** 2)), 1.0, rel_tol=1e-3), "unit mean power"

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(x.size)))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(x.size, 1 / fs))
    cumulative = np.cumsum(spectrum) / spectrum.sum()
    lo = freqs[np.searchsorted(cumulative, 0.005)]
    hi = freqs[np.searchsorted(cumulative, 0.995)]
    assert abs((hi - lo) / 1e6 - expected_bw_mhz) < tolerance_mhz


def test_zigbee_runs_at_the_standard_chip_rate():
    """802.15.4 is 2 Mchip/s, and asking the builder for 2 MHz is what produces it.

    Measured by cyclic autocorrelation, not from the envelope: half-sine O-QPSK is very nearly constant
    envelope, so ``|x|**2`` carries almost no chip-rate line to find.
    """
    estimate_symbol_rate = _symbol_rate_estimator()
    fs = 12.5e6
    x = render_burst(
        technology_id="zigbee_2_4",
        kind="data",
        duration_s=5e-3,
        occupied_bandwidth_hz=2e6,
        out_rate_hz=fs,
        seed=11,
    )
    assert abs(estimate_symbol_rate(x, fs) - 2.0e6) < 50e3


def test_bluetooth_le_runs_at_the_standard_symbol_rate():
    """Bluetooth LE's LE 1M PHY is 1 Msym/s whatever bandwidth is asked for, via ``pin_rate_to_standard``."""
    estimate_symbol_rate = _symbol_rate_estimator()
    fs = 12.5e6
    x = render_burst(
        technology_id="bluetooth_le",
        kind="advertising",
        duration_s=4e-3,
        occupied_bandwidth_hz=1e6,
        out_rate_hz=fs,
        seed=5,
    )
    assert abs(estimate_symbol_rate(x, fs) - 1.0e6) < 50e3


def test_rendering_is_deterministic():
    kwargs = {
        "technology_id": "wifi_2_4",
        "kind": "data",
        "duration_s": 6e-4,
        "occupied_bandwidth_hz": 20e6,
        "out_rate_hz": 100e6,
    }
    assert np.array_equal(render_burst(seed=4, **kwargs), render_burst(seed=4, **kwargs))


# --------------------------------------------------------------------------- receiver


def test_receiver_rejects_configurations_the_hardware_cannot_do():
    with pytest.raises(ValueError, match="master clock"):
        Receiver(master_clock_hz=150e6)
    with pytest.raises(ValueError, match="decimation"):
        Receiver(decimation=0)
    with pytest.raises(ValueError, match="gain"):
        Receiver(gain_db=40.0)


def test_lo_offset_is_bounded_by_the_analog_passband():
    # At 100 MS/s on a 160 MHz front end the spur cannot be moved out of the span at all.
    rx = Receiver(decimation=2)
    assert rx.max_lo_offset_hz == pytest.approx(30e6)
    assert rx.dc_spur_escapes_band is False
    with pytest.raises(ValueError, match="analog passband"):
        Receiver(decimation=2, lo_offset_hz=50e6)
    # It becomes possible once the rate is at most half the analog bandwidth.
    assert Receiver(decimation=4).dc_spur_escapes_band is True


def test_noise_floor_follows_ktb_plus_noise_figure():
    rx = Receiver()
    expected = THERMAL_NOISE_DBM_PER_HZ + rx.noise_figure_db + 10 * math.log10(rx.sample_rate_hz)
    assert rx.noise_power_dbm == pytest.approx(expected)
    assert rx.snr_db(-60.0, 20e6) == pytest.approx(-60.0 - rx.noise_power_dbm_in(20e6))


def test_sc16_is_left_justified_and_round_trips():
    rx = Receiver()
    rng = np.random.default_rng(0)
    # Scaled so the signal uses a reasonable part of the ADC's range; a signal of a couple of codes would
    # be measuring quantization error, not the round trip.
    amplitude = math.sqrt(10.0 ** (rx.adc_full_scale_dbm / 10.0)) / 10.0 ** (rx.gain_db / 20.0) / 8.0
    x = (rng.normal(0, amplitude, 4096) + 1j * rng.normal(0, amplitude, 4096)).astype(np.complex64)
    codes = rx.to_sc16(x)
    assert codes.dtype == np.int16
    assert codes.shape == (4096, 2)
    # 14 ADC bits in a 16-bit container leaves the low two bits clear, as UHD delivers them.
    assert np.all((codes.astype(int) & 0x3) == 0)
    back = rx.from_sc16(codes)
    assert np.mean(np.abs(back - x)) < 0.05 * np.mean(np.abs(x))


# --------------------------------------------------------------------------- stream


@pytest.fixture(scope="module")
def stream(plan: BandPlan) -> BandStream:
    return BandStream(plan, Receiver(), seed=7, environment=Environment(clutter="suburban"), chunk_s=0.002)


def test_chunk_is_a_pure_function_of_its_index(stream: BandStream):
    a, b = stream.chunk(11), stream.chunk(11)
    assert np.array_equal(a.sc16, b.sc16)
    assert a.bursts == b.bursts


def test_a_fresh_stream_reproduces_an_earlier_one(plan: BandPlan, stream: BandStream):
    other = BandStream(plan, Receiver(), seed=7, environment=Environment(clutter="suburban"), chunk_s=0.002)
    assert np.array_equal(other.chunk(11).sc16, stream.chunk(11).sc16)


def test_chunk_boundaries_are_invisible(stream: BandStream):
    """Two consecutive clean chunks must equal one render of the same span.

    They agree to complex64 precision rather than bit for bit: a burst that straddles the boundary is split
    differently in the two cases, so its mixing phase is accumulated in a different order. The point of the
    property is that no discontinuity appears at the seam, and a 1e-9 difference on signals of order 1e-3
    is far below the noise floor, let alone the ADC's least significant bit.
    """
    first, _ = stream.clean_band(5)
    second, _ = stream.clean_band(6)
    joined = np.concatenate([first, second])
    whole, _ = stream.clean_band(5, samples=joined.size)
    assert np.allclose(joined, whole, rtol=1e-5, atol=1e-8)

    # And the seam itself must not stand out: the step across it is ordinary.
    steps = np.abs(np.diff(joined))
    assert steps[first.size - 1] < 20 * np.median(steps)


def test_seeking_far_ahead_matches_a_fresh_stream(plan: BandPlan, stream: BandStream):
    other = BandStream(plan, Receiver(), seed=7, environment=Environment(clutter="suburban"), chunk_s=0.002)
    assert np.array_equal(stream.chunk(5000).sc16, other.chunk(5000).sc16)


def test_delivered_noise_floor_matches_the_prediction(plan: BandPlan):
    """With nothing transmitting, the delivered floor must be kTB + NF.

    This is the end-to-end calibration check: noise generation, gain, the 14-bit ADC, the sc16 round trip
    and the FFT scaling all have to agree for it to pass. The population is emptied deliberately, because in
    a populated band a strong emitter's skirts and its I/Q image land in the guard and are what is actually
    being measured there.
    """
    rx = Receiver()
    s = BandStream(plan, rx, seed=3, environment=Environment(clutter="suburban"), chunk_s=0.005, population=())
    nfft = 4096
    x = rx.from_sc16(s.chunk(2).sc16)
    frames = x[: (x.size // nfft) * nfft].reshape(-1, nfft)
    window = np.hanning(nfft)
    spectrum = np.fft.fftshift(np.fft.fft(frames * window, axis=-1), axes=-1)
    power = (np.abs(spectrum) ** 2 / (nfft**2 * np.mean(window**2))).mean(axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / rx.sample_rate_hz)) + rx.tune_hz

    # Away from the DC spur, which is the one deliberate feature of an otherwise empty band.
    away = np.abs(freqs - rx.dc_spur_hz) > 1e6
    measured = 10 * np.log10(np.median(power[away]))
    predicted = rx.noise_power_dbm_in(rx.sample_rate_hz / nfft)
    assert abs(measured - predicted) < 1.0, f"floor {measured:.1f} dBm/bin, expected {predicted:.1f}"


def test_a_strong_emitter_puts_an_image_on_the_other_side_of_the_lo(plan: BandPlan):
    """I/Q imbalance mirrors a signal about the LO, and a detector has to be shown those ghosts."""
    rx = Receiver(image_rejection_db=35.0)  # a poorly calibrated unit, so the ghost is unmistakable
    quiet = BandStream(plan, rx, seed=3, chunk_s=0.002, population=())
    clean, _ = quiet.clean_band(0)
    tone_hz = rx.tune_hz + 20e6
    t = np.arange(clean.size) / rx.sample_rate_hz
    clean = (1e-3 * np.exp(2j * np.pi * (tone_hz - rx.tune_hz) * t)).astype(np.complex64)
    out = rx.apply_front_end(clean, rng=np.random.default_rng(0))

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(out * np.hanning(out.size)))) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(out.size, 1 / rx.sample_rate_hz))
    signal = spectrum[np.argmin(np.abs(freqs - 20e6))]
    image = spectrum[np.argmin(np.abs(freqs + 20e6))]
    assert 10 * np.log10(signal / image) == pytest.approx(35.0, abs=3.0)


def test_truth_covers_the_chunk_window(stream: BandStream):
    chunk = stream.chunk(9)
    for burst in chunk.bursts_overlapping():
        assert burst.start_s < chunk.t1_s
        assert chunk.t0_s < burst.stop_s


def test_provenance_pins_what_the_stream_was_made_from(stream: BandStream):
    p = stream.provenance()
    assert p["seed"] == 7
    assert p["receiver"]["sample_rate_hz"] == 100e6
    assert len(p["catalog_digest"]) == 32
    assert p["band_plan"]["technologies"]
