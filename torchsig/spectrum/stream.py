"""An endless, seekable stream of synthetic I/Q for a band, as a receiver would hand it over.

This is what the rest of the package is for: :class:`BandStream` turns a band plan, a population, a schedule
and a receiver into chunks of sc16 samples, indefinitely, with the truth of what is in each chunk beside it.

**Seekable, not replayed.** A chunk is a pure function of its index. Nothing accumulates: the population is
drawn from one seed, each burst's schedule from its epoch's seed, each burst's payload from its own seed, and
the noise from the chunk's seed. So ``stream.chunk(10_000)`` costs what ``stream.chunk(0)`` costs, two
processes agree on every sample, and a chunk boundary is invisible because a burst crossing it is rendered
from the same cached waveform and the same absolute-time phase on both sides.

**What a chunk is.** ``[n, 2]`` int16 pairs, I and Q, exactly as the X3xx sends them over the wire, plus the
bursts that were on the air. The truth is the point: it is what a detector is scored against, and the same
chunk can be delivered with the truth withheld.

Example::

    from torchsig.spectrum import BandStream, Receiver, named_band

    stream = BandStream(named_band("ism_2450"), Receiver(), seed=7)
    for chunk in stream.take(10):
        print(chunk.t0_s, chunk.sc16.shape, len(chunk.bursts))
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np

import torchsig
from torchsig.spectrum.activity import Burst, Schedule
from torchsig.spectrum.population import Emitter, Environment, populate
from torchsig.spectrum.receiver import Receiver
from torchsig.spectrum.waveforms import render_burst

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torchsig.spectrum.catalog import BandPlan

__all__ = ["BandStream", "Chunk"]

DEFAULT_CHUNK_S = 0.01  # 10 ms: long enough to hold a whole mains half-cycle of oven emission


@dataclass(frozen=True)
class Chunk:
    """One block of delivered samples and the truth of what is in it.

    Attributes:
        index: Which chunk, counting from the start of the stream.
        t0_s: Stream time of the first sample.
        sc16: ``[n, 2]`` int16, I and Q, as the receiver hands them over.
        bursts: Every burst on the air at any point in this chunk, including those that started earlier or
            end later. A burst is reported whole, with its own start and duration, not clipped to the chunk.
        sample_rate_hz: Delivered sample rate.
        tune_hz: Centre of the delivered band.
    """

    index: int
    t0_s: float
    sc16: np.ndarray
    bursts: tuple[Burst, ...]
    sample_rate_hz: float
    tune_hz: float

    @property
    def num_samples(self) -> int:
        """Number of delivered samples in this chunk."""
        return int(self.sc16.shape[0])

    @property
    def duration_s(self) -> float:
        """Chunk duration in seconds."""
        return self.num_samples / self.sample_rate_hz

    @property
    def t1_s(self) -> float:
        """Stream time just past the chunk's last sample."""
        return self.t0_s + self.duration_s

    def bursts_overlapping(self) -> tuple[Burst, ...]:
        """The bursts actually audible in this chunk's window, clipped in time to it."""
        return tuple(b for b in self.bursts if b.overlaps(self.t0_s, self.t1_s))


class BandStream:
    """An endless stream of one band, through one receiver.

    Args:
        plan: The band plan. Its span should contain the receiver's; technologies outside the receiver's span
            are simply never heard.
        receiver: The receiver configuration, which fixes the sample rate, the tuned centre and every
            artifact.
        seed: Master seed. The population, the schedule, the payloads and the noise are all derived from it,
            so one seed reproduces the whole stream.
        environment: Clutter class and time of day for the population draw.
        chunk_s: Chunk duration in seconds. The default holds a whole 60 Hz mains half-cycle.
        population: An explicit population, if you want to pin it rather than draw it.
        sensitivity_dbm: Passed to the population draw; bounds how far an emitter may be.
    """

    def __init__(
        self,
        plan: BandPlan,
        receiver: Receiver | None = None,
        *,
        seed: int = 0,
        environment: Environment | None = None,
        chunk_s: float = DEFAULT_CHUNK_S,
        population: tuple[Emitter, ...] | None = None,
        sensitivity_dbm: float = -105.0,
    ) -> None:
        self.plan = plan
        self.receiver = receiver or Receiver()
        self.seed = int(seed)
        self.environment = environment or Environment()
        self.chunk_samples = round(chunk_s * self.receiver.sample_rate_hz)
        if self.chunk_samples <= 0:
            raise ValueError(f"chunk_s of {chunk_s} is shorter than one sample")
        self.population = (
            population
            if population is not None
            else populate(
                plan,
                seed=self._derive("population"),
                environment=self.environment,
                sensitivity_dbm=sensitivity_dbm,
            )
        )
        self.schedule = Schedule(self.population, seed=self._derive("activity"))
        self._audible = tuple(e for e in self.population if self._in_span(e))

    # --- derivation ------------------------------------------------------ #

    def _derive(self, label: str) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(self.seed.to_bytes(8, "little", signed=False))
        h.update(label.encode())
        return int.from_bytes(h.digest(), "little")

    def _in_span(self, emitter: Emitter) -> bool:
        """Whether any part of the emitter's occupied band falls in the delivered span."""
        lo, hi = emitter.band_hz
        return lo < self.receiver.f_high_hz and self.receiver.f_low_hz < hi

    # --- geometry -------------------------------------------------------- #

    @property
    def sample_rate_hz(self) -> float:
        """Delivered sample rate, in Hz."""
        return self.receiver.sample_rate_hz

    @property
    def chunk_s(self) -> float:
        """Chunk duration in seconds."""
        return self.chunk_samples / self.sample_rate_hz

    @property
    def audible_emitters(self) -> tuple[Emitter, ...]:
        """The emitters whose occupied band reaches the delivered span."""
        return self._audible

    def provenance(self) -> dict[str, Any]:
        """Everything needed to regenerate this stream, which is what makes a chunk citable rather than kept."""
        return {
            "torchsig_version": getattr(torchsig, "__version__", "unknown"),
            "band_plan": {
                "name": self.plan.name,
                "span_hz": [self.plan.f_low_hz, self.plan.f_high_hz],
                "technologies": [t.id for t in self.plan.technologies],
            },
            "receiver": self.receiver.describe(),
            "environment": {
                "clutter": self.environment.clutter,
                "hour_of_day": self.environment.hour_of_day,
                "day_of_week": self.environment.day_of_week,
            },
            "seed": self.seed,
            "chunk_samples": self.chunk_samples,
            "population_size": len(self.population),
            "audible_population_size": len(self._audible),
            "catalog_digest": _catalog_digest(),
        }

    # --- generation ------------------------------------------------------ #

    def clean_band(self, index: int, *, samples: int | None = None) -> tuple[np.ndarray, tuple[Burst, ...]]:
        """The noise-free sum of emitters for chunk ``index``, at the antenna in sqrt(mW).

        This is the band before the receiver touches it: no thermal noise, no image, no DC spur, no
        quantization. Paired with :meth:`chunk` it gives an exact clean/impaired pair for the same samples,
        which is what a denoising or separation objective needs.

        Continuous across chunk boundaries: a burst spanning two chunks contributes the same samples at the
        same mixing phase to both, so concatenating consecutive clean bands gives an unbroken signal.
        """
        if index < 0:
            raise ValueError(f"chunk index must be non-negative, got {index}")
        n = self.chunk_samples if samples is None else int(samples)
        c0 = index * self.chunk_samples
        t0 = c0 / self.sample_rate_hz
        t1 = (c0 + n) / self.sample_rate_hz
        bursts = tuple(self.schedule.bursts_in(t0, t1))
        band = np.zeros(n, dtype=np.complex64)
        for burst in bursts:
            self._add_burst(band, burst, c0, n)
        return band, bursts

    def chunk(self, index: int) -> Chunk:
        """Chunk ``index``, a pure function of the index and the stream's seed."""
        band, bursts = self.clean_band(index)
        n = band.size
        c0 = index * n
        t0 = c0 / self.sample_rate_hz
        rng = np.random.default_rng(self._derive(f"chunk:{index}"))
        at_adc = self.receiver.apply_front_end(band, rng=rng)
        return Chunk(
            index=index,
            t0_s=t0,
            sc16=self.receiver.to_sc16(at_adc),
            bursts=bursts,
            sample_rate_hz=self.sample_rate_hz,
            tune_hz=self.receiver.tune_hz,
        )

    def _add_burst(self, band: np.ndarray, burst: Burst, c0: int, n: int) -> None:
        """Mix one burst to its channel and add the part of it that lands in this chunk.

        The burst's samples are indexed from its own start in absolute samples, so the overlap with the chunk
        picks the same samples and the same mixing phase whichever chunk asks. That is what makes a chunk
        boundary invisible.
        """
        fs = self.sample_rate_hz
        k0 = round(burst.start_s * fs)  # absolute sample index of the burst's first sample
        n_burst = round(burst.duration_s * fs)
        if n_burst <= 0:
            return
        lo = max(k0, c0)
        hi = min(k0 + n_burst, c0 + n)
        if hi <= lo:
            return

        offset_hz = burst.center_hz - self.receiver.tune_hz
        if abs(offset_hz) >= 0.5 * fs:
            return  # outside the delivered span: aliasing it in would be a lie

        waveform = render_burst(
            technology_id=burst.technology_id,
            kind=burst.kind,
            duration_s=burst.duration_s,
            occupied_bandwidth_hz=burst.bandwidth_hz,
            out_rate_hz=fs,
            seed=burst.waveform_seed,
        )
        if waveform.size == 0:
            return
        src0, src1 = lo - k0, min(hi - k0, waveform.size)
        if src1 <= src0:
            return
        dst0 = lo - c0
        dst1 = dst0 + (src1 - src0)

        # Mixing phase from the absolute sample index, so a burst split across chunks mixes identically in
        # both. The exponential factorises as exp(i*w*(k_start + j)) = exp(i*w*k_start) * exp(i*w*j), and the
        # second factor depends only on (offset, length): both are discrete, because channel plans are
        # discrete and burst durations are drawn from discrete sets. So it is cached, and all that is left
        # per burst is one complex scalar. The reference resets every whole second to keep float64 exact; a
        # burst is far shorter than a second, so phase stays continuous within every burst.
        length = src1 - src0
        ramp = _mix_ramp(offset_hz / fs, length)
        k_start = (k0 % round(fs)) + src0
        phase0 = 2.0 * math.pi * ((burst.waveform_seed >> 20) % 1_000_000) / 1_000_000
        amplitude = math.sqrt(10.0 ** (burst.received_power_dbm / 10.0))
        scalar = np.complex64(amplitude * np.exp(1j * (2.0 * math.pi * (offset_hz / fs) * k_start + phase0)))

        band[dst0:dst1] += waveform[src0:src1] * ramp * scalar

    # --- iteration ------------------------------------------------------- #

    def __iter__(self) -> Iterator[Chunk]:
        """Chunks from the beginning, forever."""
        index = 0
        while True:
            yield self.chunk(index)
            index += 1

    def take(self, count: int, *, start: int = 0) -> Iterator[Chunk]:
        """``count`` chunks from ``start``."""
        for i in range(start, start + count):
            yield self.chunk(i)

    def chunk_at(self, t_s: float) -> Chunk:
        """The chunk containing stream time ``t_s``."""
        return self.chunk(math.floor(t_s / self.chunk_s))


@lru_cache(maxsize=4096)
def _mix_ramp(cycles_per_sample: float, length: int) -> np.ndarray:
    """``exp(2j*pi*cycles_per_sample*j)`` for ``j`` in ``[0, length)``, cached and read-only.

    Keyed on the normalised frequency offset and the length, both of which take few distinct values: channel
    plans are discrete and burst durations are drawn from discrete sets, so this is a high hit-rate cache
    that removes one ``exp`` over the whole burst from every burst.
    """
    j = np.arange(length, dtype=np.float64)
    out = np.exp(2j * np.pi * cycles_per_sample * j).astype(np.complex64)
    out.flags.writeable = False
    return out


def _mix_cache_info() -> Any:
    """Hit and miss counts of the mixing-ramp cache."""
    return _mix_ramp.cache_info()


def _catalog_digest() -> str:
    """A digest of the catalog files, so a stream's provenance pins the data it was drawn from."""
    root = resources.files("torchsig.spectrum").joinpath("catalog")
    h = hashlib.blake2b(digest_size=16)
    for name in ("additions.json", "channelization.json", "deployment_density.json", "emitter_catalog.json"):
        h.update(root.joinpath(name).read_bytes())
    return h.hexdigest()
