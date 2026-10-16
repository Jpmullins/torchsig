"""The emitter catalog: what transmits where, how often, and how loud.

Three compiled reference files describe US spectrum from 30 MHz to 6 GHz, plus a fourth holding
technologies added after they were compiled:

* ``channelization.json`` -- allocated bands, channel rasters and channel plans.
* ``emitter_catalog.json`` -- the renderer, tier, default activity model and plausible EIRP of each technology.
* ``deployment_density.json`` -- how many emitters of each technology are within reception range, by clutter
  class, with diurnal curves.
* ``additions.json`` -- technologies added later, each with its own basis and sources.

``PROVENANCE.md`` records how the first three were compiled and which fields are verified against a primary
source rather than estimated. Channel plans and PHY parameters are verified; counts, EIRP and activity figures
are engineering estimates to be fitted against measurements.

A :class:`BandPlan` is the subset of all this that applies to one span of spectrum, which is what the band
model, the population draw and the stream are built from.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = [
    "BandPlan",
    "Technology",
    "load_catalog",
    "named_band",
]

MHZ = 1.0e6
KHZ = 1.0e3

CLUTTER_CLASSES = ("urban_dense", "urban", "suburban", "rural", "open", "indoor")

# Named spans worth streaming, as (centre, span) the receiver can actually tune and sample.
_NAMED_BANDS: dict[str, tuple[str, float, float]] = {
    # 2400-2483.5 MHz ISM with 8.25 MHz of guard either side: one X310 channel at 100 MS/s.
    "ism_2450": ("2.4 GHz ISM", 2391.75e6, 2491.75e6),
    "ism_915": ("902-928 MHz ISM", 900.0e6, 930.0e6),
    "uhf_lmr": ("UHF land mobile", 380.0e6, 512.0e6),
}


@dataclass(frozen=True)
class Technology:
    """One technology's allocation, channel plan, activity model and plausible power.

    Frequencies are absolute Hz. ``channel_centers_hz`` is empty when the technology has no channel plan
    (a free-running emitter, or an allocation row with no channelization metadata).
    """

    id: str
    name: str
    category: str
    tier: int
    modulation_family: str
    renderer: str
    bands_hz: tuple[tuple[float, float, str], ...]
    channel_raster_hz: float | None
    channel_bandwidths_hz: tuple[float, ...]
    channel_centers_hz: tuple[float, ...]
    occupied_bandwidth_hz: float | None
    eirp_dbm: tuple[float, float, float]  # (min, typical, max)
    activity: dict[str, Any]
    expected_count: dict[str, float]
    reuse_channels: bool
    scale_with_observed_bandwidth: bool
    phy: dict[str, Any] = field(default_factory=dict)
    source: str = "compiled"  # "compiled" or "additions"

    @property
    def eirp_typical_dbm(self) -> float:
        """The catalog's typical EIRP for this technology, in dBm."""
        return self.eirp_dbm[1]

    def channels_in(self, f_low_hz: float, f_high_hz: float) -> tuple[float, ...]:
        """Channel centres whose occupied bandwidth falls wholly inside ``[f_low_hz, f_high_hz]``."""
        half = 0.5 * (self.occupied_bandwidth_hz or 0.0)
        return tuple(c for c in self.channel_centers_hz if c - half >= f_low_hz and c + half <= f_high_hz)

    @property
    def free_running(self) -> bool:
        """True when the technology has no channel plan, so a centre is drawn across its allocation."""
        return not self.channel_centers_hz

    def placement_spans_hz(self, f_low_hz: float, f_high_hz: float) -> tuple[tuple[float, float], ...]:
        """Centre-frequency ranges a free-running emitter may sit in, one per overlapping allocation.

        Each of this technology's allocated bands is intersected with the observed span and shrunk by half the
        occupied bandwidth, so the emitter fits whole. Technologies allocated in many places across the
        spectrum (amateur, for one) contribute only the bands that reach into this span.
        """
        half = 0.5 * (self.occupied_bandwidth_hz or 0.0)
        out = []
        for lo, hi, _ in self.bands_hz:
            if hi <= lo:  # a single-frequency allocation: the centre is pinned, not drawn
                if f_low_hz + half <= lo <= f_high_hz - half:
                    out.append((lo, lo))
                continue
            a, b = max(f_low_hz, lo) + half, min(f_high_hz, hi) - half
            if b > a:
                out.append((a, b))
        return tuple(out)

    def overlaps(self, f_low_hz: float, f_high_hz: float) -> bool:
        """Whether any of this technology's allocations reaches into ``[f_low_hz, f_high_hz]``.

        The comparison is inclusive because some allocations are a single frequency rather than a range (the
        aviation transponder frequencies, for one), and a strict comparison would make those unreachable.
        """
        return any(lo <= f_high_hz and f_low_hz <= hi for lo, hi, _ in self.bands_hz)

    def count_for(self, clutter: str) -> float:
        """Expected number of emitters of this technology in the given clutter class."""
        if clutter not in CLUTTER_CLASSES:
            raise ValueError(f"unknown clutter class {clutter!r}; expected one of {CLUTTER_CLASSES}")
        return float(self.expected_count.get(clutter, 0.0))


def _centers_from_plan(plan: dict[str, Any], bands_hz: tuple, raster_hz: float | None) -> tuple[float, ...]:
    """Absolute channel centres from a channel plan, however the plan spells them.

    Plans in the compiled files give centres in several ways: an explicit list, a first/last/count triple, or
    a formula string. Explicit lists win; a formula is parsed only for the two shapes that actually occur
    (``a + b*k`` and ``a + b*(k - c)``). Anything else falls back to the raster across the allocated band.
    """
    for key in ("centers_mhz", "us_centers_mhz"):
        if plan.get(key):
            return tuple(float(c) * MHZ for c in plan[key])

    first, last, n = plan.get("first_center_mhz"), plan.get("last_center_mhz"), plan.get("n_channels")
    if first is not None and last is not None and n:
        n = int(n)
        step = 0.0 if n == 1 else (float(last) - float(first)) / (n - 1)
        return tuple((float(first) + i * step) * MHZ for i in range(n))

    formula = plan.get("formula_mhz") or plan.get("center_frequency_mhz_formula") or ""
    # "2402 + 2*k for k = 0..39", "2405 + 5*(k - 11) for k = 11..26", "2407 + 5*n for n = 1..13"
    m = re.match(
        r"\s*([\d.]+)\s*\+\s*(?:([\d.]+)\s*\*\s*)?\(?\s*[a-z]\s*(?:-\s*([\d.]+))?\s*\)?\s*"
        r"for\s+[a-z]\s*=\s*(\d+)\s*\.\.\s*(\d+)",
        formula,
    )
    if m:
        base, step, offset, k0, k1 = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        coeff = float(step) if step else 1.0  # "2402 + k" has an implicit coefficient of 1
        off = float(offset) if offset else 0.0
        return tuple((float(base) + coeff * (k - off)) * MHZ for k in range(int(k0), int(k1) + 1))

    if raster_hz and bands_hz:
        lo, hi, _ = bands_hz[0]
        n = int((hi - lo) // raster_hz)
        return tuple(lo + raster_hz * (i + 0.5) for i in range(max(n, 0)))
    return ()


def _as_bands(raw: Any) -> tuple[tuple[float, float, str], ...]:
    """Normalise the two band spellings: catalog ``[[lo, hi, dir]]`` and channelization ``[{lower_mhz: ...}]``."""
    out: list[tuple[float, float, str]] = []
    for b in raw or []:
        if isinstance(b, dict):
            lo, hi, d = b.get("lower_mhz"), b.get("upper_mhz"), b.get("direction") or "unknown"
        else:
            lo, hi, d = ([*list(b), None, None, None])[:3]
        if lo is None or hi is None:
            continue
        out.append((float(lo) * MHZ, float(hi) * MHZ, str(d or "unknown")))
    return tuple(out)


def _khz_values(raw: Any) -> tuple[float, ...]:
    """Flatten a kHz field to numbers. A few entries key it by sub-band (``{"402-405 MHz": 300}``)."""
    if raw is None:
        return ()
    if isinstance(raw, dict):
        out: list[float] = []
        for v in raw.values():
            out.extend(_khz_values(v))
        return tuple(out)
    if isinstance(raw, (list, tuple)):
        out = []
        for v in raw:
            out.extend(_khz_values(v))
        return tuple(out)
    if isinstance(raw, bool):
        return ()
    if isinstance(raw, (int, float)):
        return (float(raw),)
    return ()


def _first(*values: Any) -> Any:
    for v in values:
        if v not in (None, (), [], {}):
            return v
    return None


def _eirp_triple(eirp: dict[str, Any]) -> tuple[float, float, float]:
    """(min, typical, max) dBm, tolerating entries that give only some of the three, or none."""
    vals = {k: eirp.get(k) for k in ("min", "typical", "max")}
    known = [float(v) for v in vals.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not known:
        return (0.0, 0.0, 0.0)
    typical = float(vals["typical"]) if isinstance(vals["typical"], (int, float)) else sum(known) / len(known)
    lo = float(vals["min"]) if isinstance(vals["min"], (int, float)) else min(known)
    hi = float(vals["max"]) if isinstance(vals["max"], (int, float)) else max(known)
    return (lo, typical, hi)


@lru_cache(maxsize=1)
def load_catalog() -> dict[str, Technology]:
    """Every technology in the catalog, keyed by id. Cached; the files are read-only data."""
    root = resources.files(__package__).joinpath("catalog")
    chan = json.loads(root.joinpath("channelization.json").read_text())
    emit = json.loads(root.joinpath("emitter_catalog.json").read_text())
    dens = json.loads(root.joinpath("deployment_density.json").read_text())
    adds = json.loads(root.joinpath("additions.json").read_text())

    chan_by_id = {t["id"]: t for t in chan["technologies"]}
    out: dict[str, Technology] = {}

    overrides = {k: v for k, v in adds.get("activity_overrides", {}).items() if not k.startswith("_")}

    def build(tid: str, cat: dict, source: str) -> Technology:
        c = chan_by_id.get(tid, {})
        d = dens["technologies"].get(tid) or adds["density"].get(tid) or dens["global_defaults"]
        plan = _first(cat.get("channel_plan"), c.get("channel_plan")) or {}
        rasters = _khz_values(_first(cat.get("channel_raster_khz"), c.get("channel_raster_khz")))
        raster_hz = min(rasters) * KHZ if rasters else None
        bands = _as_bands(_first(cat.get("bands_mhz"), c.get("bands")))
        bandwidths = tuple(b * KHZ for b in _khz_values(_first(cat.get("channel_bandwidth_khz"), c.get("channel_bandwidth_khz"))))
        phy = cat.get("phy") or {}
        occupied = _first(
            phy.get("occupied_bandwidth_hz"),
            d.get("occupied_bandwidth_hz"),
            bandwidths[0] if bandwidths else None,
        )
        eirp = cat.get("eirp_dbm") or {}
        return Technology(
            id=tid,
            name=cat.get("name") or c.get("name") or tid,
            category=cat.get("category") or c.get("category") or "unknown",
            tier=int(cat.get("tier", 3)),
            modulation_family=cat.get("modulation_family") or "unknown",
            renderer=cat.get("renderer") or "generic_service",
            bands_hz=bands,
            channel_raster_hz=raster_hz,
            channel_bandwidths_hz=bandwidths,
            channel_centers_hz=_centers_from_plan(plan, bands, raster_hz),
            occupied_bandwidth_hz=float(occupied) if occupied else None,
            eirp_dbm=_eirp_triple(eirp),
            activity={**(cat.get("activity_default") or {}), **overrides.get(tid, {})},
            expected_count=dict(d.get("expected_count") or {}),
            reuse_channels=bool(d.get("reuse_channels", False)),
            scale_with_observed_bandwidth=bool(d.get("scale_with_observed_bandwidth", False)),
            phy=phy,
            source=source,
        )

    for tid, cat in emit["technologies"].items():
        out[tid] = build(tid, cat, "compiled")
    for tid, cat in adds["technologies"].items():
        out[tid] = build(tid, cat, "additions")  # additions win on id collision
    for tid in adds["metadata"].get("supersedes", ()):
        # A compiled entry that an addition splits into finer technologies would otherwise double-count.
        out.pop(tid, None)
    return out


@dataclass(frozen=True)
class BandPlan:
    """One span of spectrum and every technology allocated inside it.

    ``f_low_hz``/``f_high_hz`` are the span the receiver observes, which is wider than the allocation of
    interest when guard is included: the guard is where adjacent services show up, and they belong in the
    stream.
    """

    name: str
    f_low_hz: float
    f_high_hz: float
    technologies: tuple[Technology, ...]

    @property
    def span_hz(self) -> float:
        """Width of the observed span, in Hz."""
        return self.f_high_hz - self.f_low_hz

    @property
    def center_hz(self) -> float:
        """Centre of the observed span, in Hz."""
        return 0.5 * (self.f_low_hz + self.f_high_hz)

    def technology(self, tid: str) -> Technology:
        """The technology with this id, or raise :class:`KeyError` if the plan does not contain it."""
        for t in self.technologies:
            if t.id == tid:
                return t
        raise KeyError(f"{tid!r} is not in band plan {self.name!r}")

    @classmethod
    def for_span(
        cls,
        name: str,
        f_low_hz: float,
        f_high_hz: float,
        *,
        include: tuple[str, ...] | None = None,
        exclude: tuple[str, ...] = (),
        max_tier: int = 3,
    ) -> BandPlan:
        """The band plan for ``[f_low_hz, f_high_hz]``.

        ``include`` pins the technology set explicitly; otherwise every catalog technology whose allocation
        overlaps the span and whose tier is at most ``max_tier`` is taken. ``exclude`` drops ids from either.
        """
        if f_high_hz <= f_low_hz:
            raise ValueError(f"f_high_hz must exceed f_low_hz, got {f_low_hz} and {f_high_hz}")
        catalog = load_catalog()
        if include is not None:
            missing = [t for t in include if t not in catalog]
            if missing:
                raise KeyError(f"not in the catalog: {missing}")
            chosen = [catalog[t] for t in include]
        else:
            chosen = [t for t in catalog.values() if t.overlaps(f_low_hz, f_high_hz) and t.tier <= max_tier]
        chosen = [t for t in chosen if t.id not in exclude]
        return cls(name, float(f_low_hz), float(f_high_hz), tuple(sorted(chosen, key=lambda t: t.id)))


def named_band(key: str, **kwargs: Any) -> BandPlan:
    """A :class:`BandPlan` for one of the named spans, e.g. ``named_band("ism_2450")``."""
    if key not in _NAMED_BANDS:
        raise KeyError(f"unknown band {key!r}; known: {sorted(_NAMED_BANDS)}")
    name, lo, hi = _NAMED_BANDS[key]
    return BandPlan.for_span(name, lo, hi, **kwargs)
