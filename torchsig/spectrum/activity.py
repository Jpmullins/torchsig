"""Turn a fixed population into a schedule of bursts on a continuous timeline.

The population says who is out there; this module says when each one transmits, and on which channel when the
technology hops. It is what makes a 2.4 GHz waterfall recognisable: Wi-Fi frames clumping on channels 1, 6
and 11, Bluetooth LE advertising in three stripes at 2402, 2426 and 2480 MHz, BR/EDR speckle hopping across
the whole band on a 625 microsecond slot clock, and a microwave oven keyed at twice the mains frequency.

**Random access.** The timeline is cut into fixed epochs. Every burst that starts in epoch *k* is drawn from
an RNG seeded by ``(master seed, emitter id, k)`` alone, so any interval can be generated without having
generated the intervals before it, and the same interval always comes out the same. Asking for the bursts in
``[t0, t1)`` visits the epochs overlapping ``[t0 - MAX_BURST_S, t1)``, which catches bursts that started
earlier and are still on the air. This is what lets a stream be seeked, resumed and reproduced, and it is why
there is no replay buffer anywhere in this package.

**Contention.** Technologies that share a channel by listening before transmitting cannot each have their
own offered load: 110 access points on 11 channels, each wanting 25% of the air, would ask for 27 times the
airtime that exists. Co-channel members of a contending technology therefore share a channel airtime cap
(:data:`MAX_CHANNEL_AIRTIME`), which is the one place this module looks at the population as a whole rather
than one emitter at a time.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torchsig.spectrum.population import Emitter

__all__ = [
    "EPOCH_S",
    "MAX_BURST_S",
    "MAX_CHANNEL_AIRTIME",
    "Burst",
    "Schedule",
]

EPOCH_S = 0.1  # the timeline is drawn in 100 ms epochs
MAX_BURST_S = 0.1  # no burst may exceed this, so the look-back for a query is bounded
MAX_CHANNEL_AIRTIME = 0.85  # the most of one channel that all contending emitters together may occupy

# Bluetooth LE primary advertising channels, by channel index (Core Spec Vol 6 Part B 1.4).
_BLE_ADV_CHANNELS_HZ = (2402.0e6, 2426.0e6, 2480.0e6)
_BLE_ADV_PACKET_S = 376e-6  # a 31-byte advertising PDU at 1 Msym/s
_BLE_INTER_CHANNEL_S = 300e-6  # gap between the three packets of one advertising event

# Bluetooth LE data channels: the band on a 2 MHz raster, minus the three advertising channels.
_BLE_DATA_CHANNELS_HZ = tuple([2404.0e6 + 2.0e6 * k for k in range(11)] + [2428.0e6 + 2.0e6 * k for k in range(26)])
_BLE_CONNECTION_EVENT_S = 0.00035  # a short master-and-slave exchange at 1 Msym/s
_BLE_T_IFS_S = 150e-6

# Bluetooth BR/EDR: 79 channels at 1 MHz from 2402 MHz, 625 us slots, packets of 1, 3 or 5 slots.
_BT_CHANNELS_HZ = tuple(2402.0e6 + 1.0e6 * k for k in range(79))
_BT_PACKET_SLOTS = (1, 3, 5)
_BT_PACKET_SLOT_WEIGHTS = (0.5, 0.2, 0.3)

# Wi-Fi: a beacon every target beacon transmission time, and data frames as aggregates.
_WIFI_BEACON_INTERVAL_S = 0.102400  # 100 TU
_WIFI_BEACON_S = 200e-6
_WIFI_ACK_S = 30e-6
# Frame durations are a discrete set, not a continuum: a real frame's length follows from its aggregation
# level and modulation-and-coding scheme. Discrete lengths also let the renderer cache each one.
_WIFI_FRAME_S = (0.0002, 0.0004, 0.0006, 0.0009, 0.0012, 0.0015)
_WIFI_FRAME_WEIGHTS = (0.30, 0.20, 0.15, 0.15, 0.10, 0.10)
# 802.15.4 payloads: a bare request, a short sensor report, and a full 127-byte PSDU at 250 kbps.
_ZIGBEE_PACKET_S = (0.00035, 0.0008, 0.0016, 0.004)


MIN_FADE_DB = -25.0  # a deeper fade puts the burst far below the noise floor; clamping keeps numbers sane


def _rician_fade_db(k_db: float, rng: np.random.Generator) -> float:
    """One burst's fading gain in dB, Rician with the given K factor and unit mean power.

    K is the ratio of dominant-path power to scattered power. A large K barely fades; K near zero is Rayleigh
    and fades deeply. The returned gain averages 0 dB over many bursts.
    """
    k = 10.0 ** (k_db / 10.0)
    dominant = math.sqrt(k / (k + 1.0))
    scatter = math.sqrt(1.0 / (2.0 * (k + 1.0)))
    h = complex(dominant + scatter * rng.normal(), scatter * rng.normal())
    gain = abs(h) ** 2
    if gain <= 0.0:
        return MIN_FADE_DB
    return max(10.0 * math.log10(gain), MIN_FADE_DB)


def _derive_seed(master_seed: int, *parts: Any) -> int:
    """A stable 64-bit seed from a master seed and any labels. Stable across runs and processes."""
    h = hashlib.blake2b(digest_size=8)
    h.update(int(master_seed).to_bytes(8, "little", signed=False))
    for part in parts:
        h.update(b"\x00")
        h.update(str(part).encode())
    return int.from_bytes(h.digest(), "little")


@dataclass(frozen=True)
class Burst:
    """One transmission: when, where in frequency, how wide, how loud, and what to render.

    ``center_hz`` is per burst, not per emitter, because hopping technologies change channel between bursts.
    ``waveform_seed`` seeds the payload so the same burst always renders the same samples.
    """

    emitter_id: str
    technology_id: str
    start_s: float
    duration_s: float
    center_hz: float
    bandwidth_hz: float
    received_power_dbm: float
    waveform_seed: int
    kind: str = "data"

    @property
    def stop_s(self) -> float:
        """Stream time at which the burst ends."""
        return self.start_s + self.duration_s

    def overlaps(self, t0: float, t1: float) -> bool:
        """Whether the burst is on the air at any point in ``[t0, t1)``."""
        return self.start_s < t1 and t0 < self.stop_s


class Schedule:
    """The burst schedule of a whole population, queryable over any time interval.

    Args:
        emitters: The fixed population.
        seed: Master seed for activity. Independent of the population seed, so the same population can be
            run with different traffic.
    """

    def __init__(self, emitters: tuple[Emitter, ...], *, seed: int) -> None:
        self.emitters = emitters
        self.seed = int(seed)
        self.airtime_scale = self._contention_scales(emitters)

    @staticmethod
    def _contention_scales(emitters: tuple[Emitter, ...]) -> dict[str, float]:
        """Per-emitter airtime scaling so co-channel contenders together stay under the channel cap."""
        groups: dict[tuple[str, int], list[Emitter]] = defaultdict(list)
        for e in emitters:
            if e.activity.get("kind") == "csma":
                groups[(e.technology_id, round(e.center_hz))].append(e)
        scales: dict[str, float] = {}
        mean_frame_s = float(np.dot(_WIFI_FRAME_S, _WIFI_FRAME_WEIGHTS))
        for members in groups.values():
            demand = sum(float(m.activity.get("offered_load", 0.25)) for m in members)
            if demand <= 0:
                continue
            # Beacons and acknowledgements occupy the channel too, so the data load gets what is left.
            beacon_airtime = len(members) * _WIFI_BEACON_S / _WIFI_BEACON_INTERVAL_S
            ack_overhead = 1.0 + _WIFI_ACK_S / mean_frame_s
            available = max(0.0, MAX_CHANNEL_AIRTIME - beacon_airtime)
            scale = min(1.0, available / (demand * ack_overhead))
            for m in members:
                scales[m.id] = scale
        return scales

    def bursts_in(self, t0: float, t1: float) -> list[Burst]:
        """Every burst on the air at any point in ``[t0, t1)``, ordered by start time.

        Deterministic and independent of call order: the same interval always returns the same bursts.
        """
        if t1 <= t0:
            return []
        first = math.floor((t0 - MAX_BURST_S) / EPOCH_S)
        last = math.floor(t1 / EPOCH_S)
        out: list[Burst] = []
        for emitter in self.emitters:
            for k in range(first, last + 1):
                out.extend(b for b in self._epoch_bursts(emitter, k) if b.overlaps(t0, t1))
        out.sort(key=lambda b: (b.start_s, b.emitter_id))
        return out

    def _epoch_bursts(self, emitter: Emitter, epoch: int) -> list[Burst]:
        """Bursts of one emitter starting inside epoch ``epoch``. Pure function of the seed and the epoch."""
        if epoch < 0:
            return []
        kind = emitter.activity.get("kind", "always_on")
        base = emitter.activity.get("base")
        if kind == "diurnal":
            # The hour-of-day curve already scaled the population count; the schedule follows the base model.
            kind = (base or {}).get("kind", "always_on") if isinstance(base, dict) else "always_on"
        rng = np.random.default_rng(_derive_seed(self.seed, emitter.id, epoch))
        t_epoch = epoch * EPOCH_S
        handler = _HANDLERS.get(kind, _always_on)
        return handler(self, emitter, t_epoch, rng)

    def make_burst(
        self,
        emitter: Emitter,
        start_s: float,
        duration_s: float,
        rng: np.random.Generator,
        *,
        center_hz: float | None = None,
        bandwidth_hz: float | None = None,
        kind: str = "data",
    ) -> Burst:
        """Build one burst for ``emitter``, applying its fading and carrier offset.

        Activity models call this rather than constructing a :class:`Burst` directly, so that every burst
        picks up the per-burst channel gain and the emitter's own carrier error.
        """
        # The channel is a sum of a dominant path and scatter, and the scatter changes whenever anything in
        # the room moves. Over one short burst it is effectively constant, but it is redrawn between bursts,
        # which is why a real emitter's packets arrive at visibly different levels. The mean power gain is 1,
        # so the emitter's mean received power is still what the population drew.
        fade_db = _rician_fade_db(emitter.rician_k_db, rng)
        # A hopping burst names its channel; the emitter's own carrier error applies either way.
        base_hz = emitter.center_hz if center_hz is None else center_hz
        return Burst(
            emitter_id=emitter.id,
            technology_id=emitter.technology_id,
            start_s=float(start_s),
            duration_s=float(min(duration_s, MAX_BURST_S)),
            center_hz=float(base_hz + emitter.frequency_offset_hz),
            bandwidth_hz=float(emitter.occupied_bandwidth_hz if bandwidth_hz is None else bandwidth_hz),
            received_power_dbm=float(emitter.received_power_dbm + fade_db),
            waveform_seed=int(rng.integers(0, 2**63)),
            kind=kind,
        )


# --------------------------------------------------------------------------- #
# Activity models. Each returns the bursts of one emitter starting in one epoch.
# --------------------------------------------------------------------------- #


def _csma(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """Wi-Fi: a beacon on the beacon clock, plus data frames as a Poisson arrival process.

    Each data frame is followed by an acknowledgement after a short interframe space, which is what gives a
    real Wi-Fi channel its paired long-burst/short-burst texture.
    """
    out: list[Burst] = []
    load = float(e.activity.get("offered_load", 0.25)) * sched.airtime_scale.get(e.id, 1.0)

    # Beacons: every emitter has its own fixed offset into the beacon interval (its TBTT).
    offset = _derive_seed(sched.seed, e.id, "tbtt") % 1_000_000 / 1_000_000 * _WIFI_BEACON_INTERVAL_S
    n = math.floor((t0 - offset) / _WIFI_BEACON_INTERVAL_S) + 1
    while True:
        t = offset + n * _WIFI_BEACON_INTERVAL_S
        if t >= t0 + EPOCH_S:
            break
        if t >= t0:
            out.append(sched.make_burst(e, t, _WIFI_BEACON_S, rng, kind="beacon"))
        n += 1

    # Data frames: Poisson arrivals at the rate that realises the offered load.
    mean_duration = float(np.dot(_WIFI_FRAME_S, _WIFI_FRAME_WEIGHTS))
    if load > 0:
        rate = load / mean_duration
        for t in _poisson_times(rng, rate, t0, EPOCH_S):
            duration = float(rng.choice(_WIFI_FRAME_S, p=_WIFI_FRAME_WEIGHTS))
            out.append(sched.make_burst(e, t, duration, rng, kind="data"))
            # The receiving station acknowledges after a short interframe space.
            out.append(sched.make_burst(e, t + duration + 16e-6, _WIFI_ACK_S, rng, kind="ack"))
    return out


def _ble_advertising(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """Bluetooth LE: advertising events on the three primary channels, with the standard random delay.

    An event sends the same payload on 37, 38 and 39 in turn. The advertising interval is fixed per device;
    the standard adds a pseudo-random delay of up to 10 ms to each event to stop devices colliding forever.
    """
    interval = float(e.activity.get("advertising_interval_s", 0.5))
    interval = min(max(interval, 0.02), 10.24)
    packets = int(e.activity.get("packets_per_event", 3))
    duration = float(e.activity.get("packet_duration_s", _BLE_ADV_PACKET_S))
    offset = _derive_seed(sched.seed, e.id, "adv") % 1_000_000 / 1_000_000 * interval

    out: list[Burst] = []
    n = math.floor((t0 - offset) / interval) + 1
    while True:
        event_t = offset + n * interval
        if event_t >= t0 + EPOCH_S:
            break
        if event_t >= t0:
            # advDelay: 0-10 ms, drawn per event from the event's own stream so it is stable.
            jitter = np.random.default_rng(_derive_seed(sched.seed, e.id, "advdelay", n)).uniform(0.0, 0.01)
            for i in range(min(packets, len(_BLE_ADV_CHANNELS_HZ))):
                t = event_t + jitter + i * (duration + _BLE_INTER_CHANNEL_S)
                out.append(
                    sched.make_burst(
                        e,
                        t,
                        duration,
                        rng,
                        center_hz=_BLE_ADV_CHANNELS_HZ[i],
                        bandwidth_hz=1.0e6,
                        kind="advertising",
                    )
                )
        n += 1

    if e.activity.get("connected"):
        out.extend(_ble_connection_events(sched, e, t0, rng))
    return out


def _ble_connection_events(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """Connection events: a master and slave exchange every connection interval, hopping the data channels.

    The channel advances by the connection's hop increment each event, modulo 37, which is what spreads a
    connected device's traffic evenly across the band between the advertising stripes.
    """
    interval = float(np.clip(e.activity.get("connection_interval_s", 0.05), 0.0075, 4.0))
    increment = int(e.activity.get("hop_increment", 7))
    start = int(e.activity.get("hop_start", 0))
    offset = _derive_seed(sched.seed, e.id, "conn") % 1_000_000 / 1_000_000 * interval

    out: list[Burst] = []
    n = math.floor((t0 - offset) / interval) + 1
    while True:
        t = offset + n * interval
        if t >= t0 + EPOCH_S:
            break
        if t >= t0:
            channel = _BLE_DATA_CHANNELS_HZ[(start + n * increment) % len(_BLE_DATA_CHANNELS_HZ)]
            out.append(sched.make_burst(e, t, _BLE_CONNECTION_EVENT_S, rng, center_hz=channel, bandwidth_hz=1.0e6, kind="connection"))
            # The slave answers one inter-frame space later, on the same channel.
            out.append(sched.make_burst(e, t + _BLE_CONNECTION_EVENT_S + _BLE_T_IFS_S, _BLE_CONNECTION_EVENT_S, rng, center_hz=channel, bandwidth_hz=1.0e6, kind="connection"))
        n += 1
    return out


def _bt_hopping(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """Bluetooth BR/EDR: a 625 microsecond slot clock, hopping pseudo-randomly over the 79 channels.

    A packet occupies 1, 3 or 5 slots and holds its channel for the whole packet, as the standard requires.
    """
    slot = float(e.activity.get("slot_duration_s", 625e-6))
    occupancy = float(e.activity.get("slot_occupancy", 0.15))
    n_slots = round(EPOCH_S / slot)
    out: list[Burst] = []
    s = 0
    while s < n_slots:
        if rng.random() < occupancy:
            length = int(rng.choice(_BT_PACKET_SLOTS, p=_BT_PACKET_SLOT_WEIGHTS))
            channel = _BT_CHANNELS_HZ[int(rng.integers(0, len(_BT_CHANNELS_HZ)))]
            # A multi-slot packet transmits for its slots less the mandatory inter-packet guard.
            duration = min(length * slot - 30e-6, MAX_BURST_S)
            out.append(sched.make_burst(e, t0 + s * slot, duration, rng, center_hz=channel, bandwidth_hz=1.0e6, kind="data"))
            s += length
        else:
            s += 1
    return out


def _zigbee_network(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """802.15.4: short packets at a slow, roughly periodic reporting interval, plus their acknowledgements."""
    interval = float(e.activity.get("report_interval_s", 10.0))
    duration = float(min(_ZIGBEE_PACKET_S, key=lambda d: abs(d - float(e.activity.get("packet_duration_s", 0.002)))))
    offset = _derive_seed(sched.seed, e.id, "report") % 1_000_000 / 1_000_000 * interval
    out: list[Burst] = []
    n = math.floor((t0 - offset) / interval) + 1
    while True:
        t = offset + n * interval
        if t >= t0 + EPOCH_S:
            break
        if t >= t0:
            out.append(sched.make_burst(e, t, duration, rng, kind="data"))
            out.append(sched.make_burst(e, t + duration + 192e-6, 352e-6, rng, kind="ack"))  # turnaround + ACK
        n += 1
    return out


def _mains_keyed(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """A microwave oven: the magnetron conducts on alternate mains half-cycles.

    The result is a burst train at the line frequency with a duty cycle near 50%: 8.3 ms on, 8.3 ms off at
    60 Hz. The emission is tens of MHz wide and its centre sweeps during each burst, which the renderer does.
    """
    line_hz = float(e.activity.get("line_frequency_hz", 60.0))
    period = 1.0 / line_hz
    on_fraction = float(e.activity.get("on_fraction", 0.5))
    on_duration = min(period * on_fraction, MAX_BURST_S)
    out: list[Burst] = []
    n = math.floor(t0 / period)
    while True:
        t = n * period
        if t >= t0 + EPOCH_S:
            break
        if t >= t0:
            out.append(sched.make_burst(e, t, on_duration, rng, kind="magnetron"))
        n += 1
    return out


def _poisson(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """A generic Poisson burst process: ``rate_hz`` bursts a second of ``burst_duration_s`` each."""
    rate = float(e.activity.get("rate_hz", 1.0))
    duration = float(e.activity.get("burst_duration_s", 0.001))
    return [sched.make_burst(e, t, duration, rng) for t in _poisson_times(rng, rate, t0, EPOCH_S)]


def _ptt(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """Push to talk: long transmissions, rare. Modelled as a two-state chain sampled per epoch."""
    duty = float(e.activity.get("duty_cycle", 0.02))
    if rng.random() >= duty:
        return []
    start = t0 + float(rng.uniform(0.0, EPOCH_S))
    return [sched.make_burst(e, start, MAX_BURST_S, rng, kind="voice")]


def _duty_cycle(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """On for a fixed fraction of every epoch, in one block."""
    duty = float(np.clip(e.activity.get("duty_cycle", 0.5), 0.0, 1.0))
    if duty <= 0.0:
        return []
    return _tile(sched, e, t0, rng, duty * EPOCH_S, kind="data")


def _always_on(sched: Schedule, e: Emitter, t0: float, rng: np.random.Generator) -> list[Burst]:
    """A carrier that never stops: tiled into MAX_BURST_S pieces so the renderer stays bounded."""
    return _tile(sched, e, t0, rng, EPOCH_S, kind="continuous")


def _tile(
    sched: Schedule,
    e: Emitter,
    t0: float,
    rng: np.random.Generator,
    total_s: float,
    *,
    kind: str,
) -> list[Burst]:
    """Cover ``total_s`` from ``t0`` with back-to-back bursts of at most MAX_BURST_S."""
    out: list[Burst] = []
    done = 0.0
    while done < total_s - 1e-12:
        step = min(MAX_BURST_S, total_s - done)
        out.append(sched.make_burst(e, t0 + done, step, rng, kind=kind))
        done += step
    return out


def _poisson_times(rng: np.random.Generator, rate_hz: float, t0: float, window_s: float) -> list[float]:
    """Arrival times of a Poisson process of rate ``rate_hz`` inside ``[t0, t0 + window_s)``."""
    if rate_hz <= 0.0:
        return []
    n = int(rng.poisson(rate_hz * window_s))
    if n == 0:
        return []
    return sorted(t0 + float(u) * window_s for u in rng.random(n))


_HANDLERS = {
    "csma": _csma,
    "ble_advertising": _ble_advertising,
    "bt_hopping": _bt_hopping,
    "zigbee_network": _zigbee_network,
    "mains_keyed": _mains_keyed,
    "poisson": _poisson,
    "ptt": _ptt,
    "duty_cycle": _duty_cycle,
    "always_on": _always_on,
}
