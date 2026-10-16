"""Render one burst to complex baseband through the TorchSig signal builders.

Each technology names a builder and the bandwidth to ask it for. The bandwidth is not a free parameter: the
builders derive their symbol, chip and subcarrier timing from it, so asking for the technology's **standard
channel bandwidth** is what makes the output standard-compliant on the air. Measured against the builders at
TorchSig 2.2:

======================  ====================  =========================  ==========================
Technology              Builder, bandwidth    What the standard requires Measured
======================  ====================  =========================  ==========================
Wi-Fi 2.4 GHz           ``80211a`` @ 20 MHz   3.2 us IFFT, 0.8 us CP     CP correlation at 3.2 us
IEEE 802.15.4           ``zigbee`` @ 2 MHz    2 Mchip/s                  chip line at 2.0000 MHz
Bluetooth LE            ``btle`` @ 1 MHz      1 Msym/s                   symbol line at 1.0000 MHz
======================  ====================  =========================  ==========================

Bluetooth LE needs ``pin_rate_to_standard``, a fork addition: without it the builder scales the symbol rate to
whatever bandwidth is requested, and a 2 MHz request gives 2 Msym/s.

**Render rate.** A burst is rendered at the lowest ``out_rate / M`` (M a power of two) that still gives
2.5 times oversampling, then resampled up by M and mixed to its place in the band. Rendering a 1 MHz
Bluetooth packet at the receiver's 100 MS/s would cost a hundred times more for nothing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from torchsig.utils.dsp import multistage_polyphase_resampler
from torchsig.utils.signal_building import signal_generator_lookup_table

__all__ = [
    "MIN_OVERSAMPLING",
    "WAVEFORM_VARIANTS",
    "WaveformSpec",
    "cache_info",
    "clear_cache",
    "render_burst",
    "spec_for",
]

MIN_OVERSAMPLING = 2.5  # render rate must be at least this many times the occupied bandwidth
_LOOKUP_ENTRY_LEN = 2  # a concrete generator's lookup entry is (class, default metadata)

# How many distinct payloads are rendered per (technology, kind, length). A burst picks one by its waveform
# seed, so the choice is deterministic, and repeat bursts of the same shape reuse a rendered variant instead
# of paying the builder again. Rendering a Wi-Fi frame costs about twenty times its own airtime, so without
# this a stream runs hundreds of times slower than real time. Raise it for training data, where a model could
# otherwise learn the variants; the payload bits are not what a spectrum view shows.
WAVEFORM_VARIANTS = 4
_CACHE_ENTRIES = 512


@dataclass(frozen=True)
class WaveformSpec:
    """How to render one technology: which builder, at what bandwidth, with what options.

    ``builder_by_kind`` overrides the builder for particular burst kinds, which is how a Wi-Fi acknowledgement
    becomes a real 802.11a control frame rather than a shortened data frame.
    """

    builder: str
    bandwidth_hz: float
    metadata: dict[str, Any] | None = None
    builder_by_kind: dict[str, str] | None = None

    def builder_for(self, kind: str) -> str:
        """The builder to use for this burst kind, which may differ from the technology's default."""
        return (self.builder_by_kind or {}).get(kind, self.builder)


# Technology id -> how to render it. Bandwidths are the standard channel bandwidths, which is what makes the
# builders produce standard timing; see the table above.
_SPECS: dict[str, WaveformSpec] = {
    "wifi_2_4": WaveformSpec(
        builder="80211a",
        bandwidth_hz=20.0e6,
        builder_by_kind={"ack": "80211a_ack", "rts": "80211a_rts", "cts": "80211a_cts"},
    ),
    "bluetooth_le": WaveformSpec(
        builder="btle",
        bandwidth_hz=1.0e6,
        metadata={"pin_rate_to_standard": True, "inter_frame_gap_symbols": 150},
    ),
    # Tier 2: BR/EDR is GFSK at 1 Msym/s on the right raster and slot clock, without packet structure.
    "bluetooth_br_edr": WaveformSpec(builder="2gfsk", bandwidth_hz=1.0e6),
    "zigbee_2_4": WaveformSpec(builder="zigbee", bandwidth_hz=2.0e6),
    # Tier 2: wideband FM standing in for an analog video sender's FM video plus audio subcarrier.
    "video_sender_2_4": WaveformSpec(builder="fm-data", bandwidth_hz=6.0e6),
    "microwave_oven": WaveformSpec(builder="oven_magnetron", bandwidth_hz=20.0e6),
    # Tier 2: an LTE-shaped downlink carrier. 600 subcarriers at 15 kHz is a 9 MHz occupied 10 MHz carrier.
    "3gpp_band_53": WaveformSpec(builder="ofdm-600", bandwidth_hz=9.0e6),
    "amateur": WaveformSpec(builder="2gfsk", bandwidth_hz=20.0e3),
    "mss_lband_iridium_globalstar_inmarsat": WaveformSpec(builder="qpsk", bandwidth_hz=40.0e3),
}

_FALLBACK = WaveformSpec(builder="qpsk", bandwidth_hz=0.0)


def spec_for(technology_id: str, occupied_bandwidth_hz: float = 0.0) -> WaveformSpec:
    """The waveform specification for a technology, or a generic QPSK carrier at its occupied bandwidth."""
    spec = _SPECS.get(technology_id)
    if spec is not None:
        return spec
    return WaveformSpec(builder=_FALLBACK.builder, bandwidth_hz=occupied_bandwidth_hz or 1.0e6)


def _render_rate(bandwidth_hz: float, out_rate_hz: float) -> tuple[float, int]:
    """The lowest ``out_rate / 2**k`` giving at least MIN_OVERSAMPLING, with its integer interpolation factor."""
    need = MIN_OVERSAMPLING * bandwidth_hz
    if need >= out_rate_hz:
        return out_rate_hz, 1
    m = 1
    while out_rate_hz / (2 * m) >= need:
        m *= 2
    return out_rate_hz / m, m


def _oven_magnetron(
    n_samples: int,
    render_rate_hz: float,
    bandwidth_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """A microwave oven's emission: band-limited noise whose centre sweeps across the burst.

    A free-running magnetron emits broadband noise, and its frequency pulls by several MHz within a single
    mains half-cycle as the anode voltage rises and falls. The result is a noisy, chirping blob tens of MHz
    wide, which is exactly how an oven shows up in a 2.4 GHz waterfall.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.complex64)
    noise = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / math.sqrt(2.0)
    # Shape the noise to the emission bandwidth in the frequency domain.
    spectrum = np.fft.fft(noise)
    freqs = np.fft.fftfreq(n_samples, 1.0 / render_rate_hz)
    core = 0.35 * bandwidth_hz
    spectrum *= np.exp(-0.5 * (freqs / core) ** 4)  # a flat-topped, steep-skirted blob
    shaped = np.fft.ifft(spectrum)
    # Sweep the centre across the burst. The magnetron's frequency follows its anode voltage, which follows
    # the rectified mains half-cycle, so the centre arcs up and back down rather than chirping one way.
    u = np.arange(n_samples) / max(n_samples - 1, 1)
    half_cycle = np.sin(np.pi * u)
    span = 0.25 * bandwidth_hz
    # Phase is the integral of the instantaneous frequency offset over time.
    dt = 1.0 / render_rate_hz
    phase = 2.0 * np.pi * span * np.cumsum(half_cycle - half_cycle.mean()) * dt
    shaped = shaped * np.exp(1j * phase)
    # Envelope: the magnetron rises and falls with the same half-cycle.
    return (shaped * half_cycle**0.5).astype(np.complex64)


def _derive_variant_seed(technology_id: str, kind: str, n_out: int, variant: int) -> int:
    """A stable seed for one cached variant, so the cache and a cold render agree."""
    h = hashlib.blake2b(f"{technology_id}|{kind}|{n_out}|{variant}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def _unit_power(x: np.ndarray) -> np.ndarray:
    power = float(np.mean(x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2))
    if power <= 0.0 or not math.isfinite(power):
        return np.zeros_like(x)
    return (x / math.sqrt(power)).astype(np.complex64)


@lru_cache(maxsize=_CACHE_ENTRIES)
def _render_cached(
    technology_id: str,
    kind: str,
    n_out: int,
    out_rate_hz: float,
    occupied_bandwidth_hz: float,
    variant: int,
) -> np.ndarray:
    """One rendered variant, cached. The returned array is shared, so callers must not write to it."""
    out = _render(technology_id, kind, n_out, out_rate_hz, occupied_bandwidth_hz, variant)
    out.flags.writeable = False
    return out


def clear_cache() -> None:
    """Drop every cached waveform. Rendering is deterministic, so this only costs time."""
    _render_cached.cache_clear()


def cache_info() -> Any:
    """Hit and miss counts of the waveform cache, for checking a stream is not re-rendering needlessly."""
    return _render_cached.cache_info()


def render_burst(
    *,
    technology_id: str,
    kind: str,
    duration_s: float,
    occupied_bandwidth_hz: float,
    out_rate_hz: float,
    seed: int,
) -> np.ndarray:
    """One burst at complex baseband and unit mean power, sampled at ``out_rate_hz``.

    The burst carries no frequency offset and no power scaling: the stream mixes it to its channel and scales
    it to its received power, so the same rendered burst can be reused at any place in the band.

    One of :data:`WAVEFORM_VARIANTS` payloads is chosen by ``seed``, rendered once and cached, so a stream
    pays the builder once per (technology, kind, length, variant) rather than once per burst.

    Args:
        technology_id: Which technology, which picks the builder and its bandwidth.
        kind: Burst kind from the schedule ("data", "ack", "beacon", "advertising", "magnetron", ...).
        duration_s: How long the burst is on the air.
        occupied_bandwidth_hz: Used only when the technology has no specification of its own.
        out_rate_hz: The receiver's sample rate.
        seed: Selects the payload variant, so a burst always renders the same samples.

    Returns:
        A read-only complex64 view of length ``round(duration_s * out_rate_hz)`` at unit mean power. Empty if
        the duration rounds to no samples. Do not write to it; copy first if you must.
    """
    n_out = round(duration_s * out_rate_hz)
    if n_out <= 0:
        return np.zeros(0, dtype=np.complex64)
    return _render_cached(
        technology_id,
        kind,
        n_out,
        float(out_rate_hz),
        float(occupied_bandwidth_hz),
        int(seed) % WAVEFORM_VARIANTS,
    )


def _render(
    technology_id: str,
    kind: str,
    n_out: int,
    out_rate_hz: float,
    occupied_bandwidth_hz: float,
    variant: int,
) -> np.ndarray:
    """Render one burst from scratch. ``variant`` seeds the payload."""
    spec = spec_for(technology_id, occupied_bandwidth_hz)
    bandwidth = spec.bandwidth_hz or occupied_bandwidth_hz or 1.0e6
    render_rate, factor = _render_rate(bandwidth, out_rate_hz)
    n_render = max(math.ceil(n_out / factor), 2)
    seed = _derive_variant_seed(technology_id, kind, n_out, variant)
    rng = np.random.default_rng(seed)

    if spec.builder == "oven_magnetron":
        baseband = _oven_magnetron(n_render, render_rate, bandwidth, rng)
    else:
        name = spec.builder_for(kind)
        entry = signal_generator_lookup_table.get(name)
        if entry is None or len(entry) != _LOOKUP_ENTRY_LEN:
            raise KeyError(f"{name!r} is not a concrete TorchSig signal generator")
        cls, defaults = entry
        pinned = {
            "sample_rate": render_rate,
            "bandwidth_min": bandwidth,
            "bandwidth_max": bandwidth,
            "signal_duration_in_samples_min": n_render,
            "signal_duration_in_samples_max": n_render,
            **(spec.metadata or {}),
        }
        generator = cls(metadata={**defaults, **pinned}, seed=0)
        generator.seed(int(seed) % (2**63))
        baseband = np.asarray(generator().data, dtype=np.complex64)

    if factor > 1:
        baseband = np.asarray(multistage_polyphase_resampler(baseband, float(factor)), dtype=np.complex64)
    if baseband.size < n_out:
        baseband = np.pad(baseband, (0, n_out - baseband.size))
    return _unit_power(baseband[:n_out])
