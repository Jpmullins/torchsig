"""Draw a plausible population of emitters for a band.

The population is what the receiver could hear, before anything is transmitted: who is out there, on which
channel, how far away and how loud. It is drawn once per stream and then held fixed, so an emitter keeps its
channel and its received power for the whole run, the way a neighbour's access point does. What varies with
time is activity (:mod:`torchsig.spectrum.activity`), not the population.

How a count becomes emitters, for each technology in the band plan:

1. ``expected_count`` for the clutter class is the mean number within reception range, counted over the
   technology's whole allocation. When the technology is channelized and the observed span covers only part
   of that allocation, the mean scales by the covered fraction (``scale_with_observed_bandwidth``).
2. An activity model of kind ``diurnal`` scales the mean by the hour-of-day curve.
3. The realized count is Poisson about that mean, so a scene is sometimes busier than average.
4. Each emitter draws a channel (honouring the plan, and the deployment preferences that real bands show:
   Wi-Fi crowds 1/6/11, 802.15.4 prefers the gaps between them), an EIRP (lognormal about the catalog
   typical, truncated to its min and max), and a range (uniform in the annulus between ``min_range_m`` and
   ``max_range_km``, which makes area, not range, uniform).
5. Received power is EIRP minus a log-distance path loss with a clutter-dependent exponent and lognormal
   shadowing, referred to an isotropic receive antenna.

Units: Hz, metres, dBm, dB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from torchsig.geo.utils.propagation import free_space_path_loss_db

if TYPE_CHECKING:
    from torchsig.spectrum.catalog import BandPlan, Technology

__all__ = [
    "Emitter",
    "Environment",
    "populate",
]

# Log-distance path-loss exponent by clutter class. 2.0 is free space; the rest are the usual indoor/urban
# ranges for 2-6 GHz. Shadowing standard deviation grows with clutter.
_PATH_LOSS_EXPONENT = {
    "open": 2.0,
    "rural": 2.2,
    "suburban": 2.7,
    "urban": 3.0,
    "urban_dense": 3.3,
    "indoor": 3.0,
}
_SHADOWING_SIGMA_DB = {
    "open": 3.0,
    "rural": 4.0,
    "suburban": 6.0,
    "urban": 7.0,
    "urban_dense": 8.0,
    "indoor": 7.0,
}
_REFERENCE_DISTANCE_M = 1.0

# Rician K factor by clutter, in dB: how much of the received power arrives on a dominant path rather than
# through scatter. Low K means deep burst-to-burst fades; a high K is a near line-of-sight link.
_RICIAN_K_DB = {
    "open": 12.0,
    "rural": 10.0,
    "suburban": 7.0,
    "urban": 4.0,
    "urban_dense": 2.0,
    "indoor": 5.0,
}

# Carrier frequency tolerance by technology, in parts per million. Consumer radios sit near their standard's
# limit, and at 2.44 GHz 20 ppm is about 49 kHz, which is visible at a fine FFT.
_FREQUENCY_TOLERANCE_PPM = {
    "wifi_2_4": 20.0,  # IEEE 802.11 clause 17: +/- 20 ppm
    "bluetooth_le": 50.0,  # Bluetooth Core, LE: +/- 50 ppm
    "bluetooth_br_edr": 20.0,  # Bluetooth Core, BR: +/- 20 ppm
    "zigbee_2_4": 40.0,  # IEEE 802.15.4: +/- 40 ppm
    "3gpp_band_53": 0.05,  # a base station is disciplined to a far tighter reference
}
_DEFAULT_TOLERANCE_PPM = 20.0

# The range at which a clutter class's K factor applies, and the range K is allowed to move over.
_K_REFERENCE_RANGE_M = 10.0
_K_MIN_DB = -5.0  # Rayleigh: no dominant path at all
_K_MAX_DB = 15.0  # a strong line of sight across a room

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
_RANGE_PAIR = 2  # a catalog field written as [low, high] is a range to draw from

# Deployment preferences that real bands show and a uniform channel draw misses.
# Weight per channel centre in MHz; channels absent from a map keep weight 1.
_CHANNEL_PREFERENCE: dict[str, dict[float, float]] = {
    # US 2.4 GHz Wi-Fi crowds onto the three non-overlapping channels.
    "wifi_2_4": {2412.0: 8.0, 2437.0: 8.0, 2462.0: 8.0},
    # 802.15.4 deployment guides point installers at the channels between Wi-Fi 1/6/11.
    "zigbee_2_4": {2405.0: 3.0, 2410.0: 3.0, 2425.0: 3.0, 2430.0: 3.0, 2450.0: 3.0, 2455.0: 3.0, 2475.0: 4.0, 2480.0: 4.0},
}


@dataclass(frozen=True)
class Environment:
    """Where the receiver is and when, which together set how many emitters there are and how loud.

    ``clutter`` is one of the catalog's classes and drives both the expected counts and the path-loss
    exponent. ``hour_of_day`` is local hour (0-23, fractional allowed) and scales technologies whose activity
    model is diurnal. ``noise_figure_db`` belongs to the receiver, not here.
    """

    clutter: str = "urban"
    hour_of_day: float = 20.0  # the diurnal peak, i.e. the busy hour
    day_of_week: int = 2  # Monday = 0

    def __post_init__(self) -> None:
        """Validate the clutter class and the time of day."""
        if self.clutter not in _PATH_LOSS_EXPONENT:
            raise ValueError(f"unknown clutter class {self.clutter!r}; expected one of {sorted(_PATH_LOSS_EXPONENT)}")
        if not 0.0 <= self.hour_of_day < HOURS_PER_DAY:
            raise ValueError(f"hour_of_day must be in [0, 24), got {self.hour_of_day}")
        if not 0 <= self.day_of_week <= DAYS_PER_WEEK - 1:
            raise ValueError(f"day_of_week must be in [0, 6], got {self.day_of_week}")

    @property
    def path_loss_exponent(self) -> float:
        """Log-distance path-loss exponent for this clutter class."""
        return _PATH_LOSS_EXPONENT[self.clutter]

    @property
    def rician_k_db(self) -> float:
        """Rician K factor of a typical link in this clutter class, in dB."""
        return _RICIAN_K_DB[self.clutter]

    @property
    def shadowing_sigma_db(self) -> float:
        """Standard deviation of lognormal shadowing for this clutter class, in dB."""
        return _SHADOWING_SIGMA_DB[self.clutter]


@dataclass(frozen=True)
class Emitter:
    """One emitter in the population: fixed for the life of the stream.

    ``received_power_dbm`` is the power an isotropic antenna at the receiver sees while the emitter is
    transmitting, which the receiver chain turns into an SNR against its own noise floor. ``activity`` is the
    resolved activity specification, with every drawn parameter filled in, that
    :mod:`torchsig.spectrum.activity` turns into a burst schedule.
    """

    id: str
    technology_id: str
    center_hz: float
    occupied_bandwidth_hz: float
    eirp_dbm: float
    range_m: float
    path_loss_db: float
    received_power_dbm: float
    activity: dict[str, Any]
    channel_index: int | None = None
    tier: int = 3
    renderer: str = "generic_service"
    frequency_offset_hz: float = 0.0  # this radio's carrier error, fixed for its lifetime
    rician_k_db: float = 6.0  # dominant-path ratio, which sets how deeply its bursts fade

    @property
    def transmit_center_hz(self) -> float:
        """Where this emitter actually transmits: its channel, plus its own carrier error."""
        return self.center_hz + self.frequency_offset_hz

    @property
    def band_hz(self) -> tuple[float, float]:
        """The emitter's occupied band as (low, high) in absolute Hz."""
        half = 0.5 * self.occupied_bandwidth_hz
        return (self.center_hz - half, self.center_hz + half)


def _diurnal_factor(technology: Technology, env: Environment) -> float:
    """Hour-of-day and day-of-week scaling, for technologies whose activity model is diurnal."""
    if technology.activity.get("kind") != "diurnal":
        return 1.0
    curve = technology.activity.get("hour_of_day") or _DEFAULT_DIURNAL
    h = env.hour_of_day % 24.0
    lo, hi = math.floor(h) % 24, math.ceil(h) % 24
    frac = h - math.floor(h)
    hourly = (1.0 - frac) * float(curve[lo]) + frac * float(curve[hi])
    weekly = _DEFAULT_WEEKLY[env.day_of_week]
    return hourly * weekly


_DEFAULT_DIURNAL = (
    0.45,
    0.36,
    0.30,
    0.27,
    0.28,
    0.34,
    0.48,
    0.64,
    0.76,
    0.82,
    0.85,
    0.87,
    0.88,
    0.88,
    0.87,
    0.87,
    0.89,
    0.93,
    0.97,
    1.00,
    0.98,
    0.90,
    0.75,
    0.58,
)
_DEFAULT_WEEKLY = (1.0, 1.0, 1.0, 1.0, 1.02, 0.88, 0.80)


def _expected_count(technology: Technology, plan: BandPlan, env: Environment) -> float:
    """Mean number of emitters of this technology the receiver can hear in the observed span."""
    mean = technology.count_for(env.clutter)
    if technology.scale_with_observed_bandwidth and technology.bands_hz:
        allocated = sum(hi - lo for lo, hi, _ in technology.bands_hz)
        observed = sum(max(0.0, min(plan.f_high_hz, hi) - max(plan.f_low_hz, lo)) for lo, hi, _ in technology.bands_hz)
        if allocated > 0:
            mean *= observed / allocated
    return max(0.0, mean * _diurnal_factor(technology, env))


def _draw_eirp_dbm(technology: Technology, rng: np.random.Generator, spread_db: float) -> float:
    """EIRP lognormal about the catalog typical, truncated to the catalog min and max."""
    lo, typical, hi = technology.eirp_dbm
    if hi <= lo:
        return typical
    return float(np.clip(rng.normal(typical, spread_db), lo, hi))


def _reception_range_m(eirp_dbm: float, center_hz: float, sensitivity_dbm: float, exponent: float) -> float:
    """The range at which this emitter's median received power falls to ``sensitivity_dbm``.

    The catalog counts emitters *within reception range*, so this is what bounds the range draw: a +0 dBm
    Bluetooth tag is audible across a room, a +28 dBm access point across the street, and drawing both from
    the same annulus would put most of the population below the noise floor.
    """
    budget_db = eirp_dbm - sensitivity_dbm - free_space_path_loss_db(_REFERENCE_DISTANCE_M, center_hz)
    if budget_db <= 0.0:
        return 0.0
    return float(_REFERENCE_DISTANCE_M * 10.0 ** (budget_db / (10.0 * exponent)))


def _draw_range_m(
    technology: Technology,
    eirp_dbm: float,
    center_hz: float,
    env: Environment,
    rng: np.random.Generator,
    sensitivity_dbm: float,
) -> float | None:
    """Range drawn uniformly in area, out to the smaller of the category range and this emitter's own reach.

    Returns None when the emitter cannot be heard even at the category's minimum range, in which case it is
    not part of the population at all.
    """
    r_min, r_category = _SPATIAL_BY_CATEGORY.get(technology.category, _SPATIAL_DEFAULT)
    r_reach = _reception_range_m(eirp_dbm, center_hz, sensitivity_dbm, env.path_loss_exponent)
    r_max = min(r_category, r_reach)
    if r_max <= r_min:
        return None
    # Uniform in area over the annulus: r = sqrt(U(r_min^2, r_max^2)).
    return float(math.sqrt(rng.uniform(r_min**2, r_max**2)))


# Reception-range annulus per catalog category, in metres (from deployment_density category defaults).
_SPATIAL_BY_CATEGORY: dict[str, tuple[float, float]] = {
    "unlicensed": (5.0, 600.0),
    "unintentional": (2.0, 150.0),
    "personal_radio": (20.0, 3000.0),
    "amateur": (100.0, 20000.0),
    "cellular_3gpp": (100.0, 5000.0),
    "satellite": (500000.0, 2000000.0),
}
_SPATIAL_DEFAULT = (100.0, 30000.0)


def _path_loss_db(range_m: float, center_hz: float, env: Environment, rng: np.random.Generator) -> float:
    """Log-distance path loss with clutter-dependent exponent and lognormal shadowing."""
    reference = free_space_path_loss_db(_REFERENCE_DISTANCE_M, center_hz)
    excess = 10.0 * env.path_loss_exponent * math.log10(max(range_m, _REFERENCE_DISTANCE_M) / _REFERENCE_DISTANCE_M)
    shadowing = float(rng.normal(0.0, env.shadowing_sigma_db))
    return reference + excess + shadowing


def _choose_channel(
    technology: Technology,
    plan: BandPlan,
    rng: np.random.Generator,
    taken: set[int],
) -> tuple[float | None, int | None]:
    """A channel centre for one emitter: from the plan when there is one, else a free-running draw."""
    if technology.free_running:
        spans = technology.placement_spans_hz(plan.f_low_hz, plan.f_high_hz)
        if not spans:
            return None, None
        # Weight by width, so a wide allocation takes proportionally more emitters. A single-frequency
        # allocation has zero width, so give every span a floor to keep the weights usable.
        widths = np.array([max(hi - lo, 1.0) for lo, hi in spans], dtype=float)
        lo, hi = spans[int(rng.choice(len(spans), p=widths / widths.sum()))]
        return float(rng.uniform(lo, hi)) if hi > lo else float(lo), None

    centers = technology.channels_in(plan.f_low_hz, plan.f_high_hz)
    if not centers:
        return None, None
    indices = list(range(len(centers)))
    if not technology.reuse_channels:
        indices = [i for i in indices if i not in taken]
        if not indices:
            return None, None
    prefs = _CHANNEL_PREFERENCE.get(technology.id, {})
    weights = np.array([prefs.get(round(centers[i] / 1e6, 3), 1.0) for i in indices], dtype=float)
    pick = int(rng.choice(indices, p=weights / weights.sum()))
    return float(centers[pick]), pick


def _draw_offered_load(activity: dict[str, Any], rng: np.random.Generator) -> float:
    """One emitter's share of the air, drawn from a mostly-idle population with a long tail.

    A deployment count is a count of *audible* emitters, and a contention-based emitter is audible from its
    beacons alone. So a load drawn here is zero for most emitters and lognormal for the few that are busy.
    """
    active_fraction = float(activity.get("active_fraction", 1.0))
    if active_fraction < 1.0 and rng.random() >= active_fraction:
        return 0.0  # audible, but carrying nothing beyond its beacons
    median = float(activity.get("offered_load_median", activity.get("offered_load", 0.25)))
    sigma = float(activity.get("offered_load_log_sigma", 0.0))
    if sigma <= 0.0:
        return median
    return float(np.clip(median * math.exp(rng.normal(0.0, sigma)), 0.0, 0.95))


def _resolve_activity(technology: Technology, rng: np.random.Generator) -> dict[str, Any]:
    """The technology's default activity model with every range drawn to a value for this emitter."""
    spec: dict[str, Any] = {"kind": technology.activity.get("kind", "always_on")}
    for key, value in technology.activity.items():
        if key == "kind":
            continue
        if isinstance(value, list) and len(value) == _RANGE_PAIR and all(isinstance(v, (int, float)) for v in value):
            spec[key] = float(rng.uniform(float(value[0]), float(value[1])))  # a range: draw per emitter
        else:
            spec[key] = value
    # PHY facts the schedule needs and the catalog records on the technology, not the activity model.
    for key in ("slot_duration_s", "hop_rate_hz", "line_frequency_hz", "on_duration_s", "symbol_rate_hz"):
        if key in technology.phy and key not in spec:
            spec[key] = technology.phy[key]

    # Per-emitter traffic draws, for the technologies whose population is mostly idle.
    if spec["kind"] == "csma":
        spec["offered_load"] = _draw_offered_load(technology.activity, rng)
    elif spec["kind"] == "ble_advertising":
        # A device that is paired and in use advertises *and* holds a connection on the data channels.
        spec["connected"] = bool(rng.random() < float(technology.activity.get("connected_fraction", 0.0)))
        spec["hop_increment"] = int(rng.integers(5, 17))  # Core Spec: 5 to 16 inclusive
        spec["hop_start"] = int(rng.integers(0, 37))
    elif spec["kind"] == "bt_hopping" and "slot_occupancy_active" in technology.activity:
        active = rng.random() < float(technology.activity.get("active_fraction", 1.0))
        key = "slot_occupancy_active" if active else "slot_occupancy_idle"
        spec["slot_occupancy"] = float(technology.activity[key])
        spec["link_active"] = bool(active)
    return spec


def populate(
    plan: BandPlan,
    *,
    seed: int,
    environment: Environment | None = None,
    eirp_spread_db: float = 6.0,
    sensitivity_dbm: float = -105.0,
) -> tuple[Emitter, ...]:
    """Draw the population of emitters for ``plan``.

    Deterministic in ``seed``: the same seed, plan and environment give the same emitters, in the same order,
    with the same channels and powers.

    Args:
        plan: The band plan, which fixes the observed span and the technologies in it.
        seed: Seed for the population draw. Independent of the per-emitter activity and waveform seeds.
        environment: Clutter class and time of day. Defaults to an urban busy hour.
        eirp_spread_db: Standard deviation of the EIRP draw about the catalog typical, in dB.
        sensitivity_dbm: The received power that defines the edge of reception range. It bounds how far away
            an emitter of a given EIRP may be drawn, so the population is one the receiver can actually hear.
            About -105 dBm is a narrow-channel noise floor plus a few dB.

    Returns:
        The emitters, ordered by technology id then by the order drawn.
    """
    env = environment or Environment()
    rng = np.random.default_rng(seed)
    emitters: list[Emitter] = []

    for technology in plan.technologies:
        mean = _expected_count(technology, plan, env)
        if mean <= 0.0:
            continue
        count = int(rng.poisson(mean))
        taken: set[int] = set()
        for n in range(count):
            center_hz, channel_index = _choose_channel(technology, plan, rng, taken)
            if center_hz is None:
                continue  # no channel of this technology fits the observed span
            if channel_index is not None:
                taken.add(channel_index)
            eirp = _draw_eirp_dbm(technology, rng, eirp_spread_db)
            range_m = _draw_range_m(technology, eirp, center_hz, env, rng, sensitivity_dbm)
            if range_m is None:
                continue  # not audible even at the category's minimum range
            loss = _path_loss_db(range_m, center_hz, env, rng)
            # A short link keeps a dominant path; a long one through clutter is almost all scatter. The
            # clutter class sets K for a typical link at the reference range, and range moves it from there.
            k_db = env.rician_k_db + 6.0 * math.log10(_K_REFERENCE_RANGE_M / max(range_m, 0.5))
            k_db = float(np.clip(k_db, _K_MIN_DB, _K_MAX_DB))
            # A radio's carrier sits within its standard's tolerance of the channel, not on it. The error is
            # fixed per radio: it is a crystal, not a random draw per packet.
            ppm = _FREQUENCY_TOLERANCE_PPM.get(technology.id, _DEFAULT_TOLERANCE_PPM)
            offset_hz = float(rng.normal(0.0, center_hz * ppm * 1e-6 / 2.0))
            offset_hz = float(np.clip(offset_hz, -center_hz * ppm * 1e-6, center_hz * ppm * 1e-6))
            emitters.append(
                Emitter(
                    id=f"{technology.id}-{n:03d}",
                    technology_id=technology.id,
                    center_hz=center_hz,
                    occupied_bandwidth_hz=float(technology.occupied_bandwidth_hz or 0.0),
                    eirp_dbm=eirp,
                    range_m=range_m,
                    path_loss_db=loss,
                    received_power_dbm=eirp - loss,
                    activity=_resolve_activity(technology, rng),
                    channel_index=channel_index,
                    tier=technology.tier,
                    renderer=technology.renderer,
                    frequency_offset_hz=offset_hz,
                    rician_k_db=k_db,
                )
            )
    return tuple(emitters)
