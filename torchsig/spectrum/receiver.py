"""The receiver: an Ettus USRP X310 with a UBX-160 daughterboard, and what it does to the band.

What reaches a host is not the band. It is the band as one particular direct-conversion receiver saw it, and
the artifacts are large enough to dominate a detector that has never been shown them:

* **Rate.** The X310 supports a 200 MHz or a 184.32 MHz master clock and nothing else, and the host rate is
  that clock divided by an integer from 1 to 1024. Nothing at 184.32 MHz divides into the 51.2 microsecond
  window grid, so 200 MHz is the practical choice.
* **Analog bandwidth.** The UBX-160 is 160 MHz wide above 500 MHz and 84 MHz below it. Beyond that the front
  end rolls off, so a span wider than the analog bandwidth has soft edges rather than hard ones.
* **A DC spur that cannot always be escaped.** Direct conversion puts LO leakage at the centre of the analog
  passband. Tuning the LO away from the band centre by ``lo_offset_hz`` moves the spur, but two constraints
  fight: the spur leaves the captured span only when ``|lo_offset| >= sample_rate/2``, while the span stays
  inside the analog passband only when ``|lo_offset| <= (analog_bandwidth - sample_rate)/2``. Both hold only
  when ``sample_rate <= analog_bandwidth/2``. At 100 MS/s on a 160 MHz front end they cannot both hold, so
  the spur can be moved off the band centre by at most 30 MHz but not removed, and the stream carries a
  masked DC bin. :attr:`Receiver.max_lo_offset_hz` and :attr:`Receiver.dc_spur_escapes_band` report this.
* **I/Q imbalance images.** Gain and phase mismatch between the I and Q rails mirrors every signal about the
  LO. The specification is better than -30 dBc; a unit calibrated against real recordings measured 55 to
  70 dB of rejection. A strong Wi-Fi carrier therefore plants a ghost on the other side of the LO, and a
  detector that has not seen ghosts will report them as emitters.
* **A 14-bit ADC, and sc16 on the wire.** The X3xx sends sc16 only, so what the host gets is pairs of
  int16 holding 14 significant bits. Quantization and clipping are part of the signal.

Default values are those of ``x310_v3``, a configuration recalibrated against real X310 recordings; the
figures the specifications do not give (full scale referred to the antenna, residual DC, image level) are
marked provisional there and here.

Units: Hz, dB, dBm. I/Q is in sqrt(mW) at the antenna, so ``mean |x|^2`` is power in mW, until
:meth:`Receiver.to_sc16` hands over integers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

__all__ = [
    "MASTER_CLOCKS_HZ",
    "THERMAL_NOISE_DBM_PER_HZ",
    "Receiver",
    "noise_figure_db",
]

THERMAL_NOISE_DBM_PER_HZ = -174.0  # kT at 290 K
MASTER_CLOCKS_HZ = (200.0e6, 184.32e6)  # the only two the X310 offers
MAX_DECIMATION = 1024
GAIN_MIN_DB = 0.0  # the UBX-160's gain range, in 0.5 dB steps
GAIN_MAX_DB = 31.5
SC16_BITS = 16  # the X3xx sends sc16 only, so the ADC's bits must fit in 16
FLAT_RESPONSE_RIPPLE_DB = 0.01  # below this the analog roll-off is not worth a transform pair

# UBX-160 noise figure by band, from the daughterboard specifications: (f_min, f_max, nf_min, nf_max).
_NOISE_FIGURE_BANDS = (
    (10.0e6, 500.0e6, 3.0, 4.0),
    (500.0e6, 1500.0e6, 2.0, 3.0),
    (1500.0e6, 6000.0e6, 4.0, 10.0),
)


def noise_figure_db(center_hz: float, *, quantile: float = 0.5) -> float:
    """The UBX-160 noise figure at a tuned frequency, interpolated within its specified band range.

    ``quantile`` picks where in the specified min-to-max range to sit: 0 for the optimistic end, 1 for the
    pessimistic, 0.5 for the middle.
    """
    for f_min, f_max, nf_min, nf_max in _NOISE_FIGURE_BANDS:
        if f_min <= center_hz < f_max:
            return float(nf_min + (nf_max - nf_min) * np.clip(quantile, 0.0, 1.0))
    raise ValueError(f"{center_hz / 1e6:.3f} MHz is outside the UBX-160's 10 MHz to 6 GHz range")


@lru_cache(maxsize=16)
def _analog_response(n: int, sample_rate_hz: float, lo_offset_hz: float, analog_bandwidth_hz: float) -> np.ndarray:
    """Front-end magnitude response across ``n`` bins of the span. Cached: it is fixed per configuration."""
    offsets = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate_hz)) + lo_offset_hz
    edge = 0.5 * analog_bandwidth_hz
    out = 1.0 / np.sqrt(1.0 + (np.abs(offsets) / edge) ** 8)  # a soft, steep skirt beyond the edge
    out.flags.writeable = False
    return out


@lru_cache(maxsize=16)
def _tone(n: int, frequency_hz: float, sample_rate_hz: float) -> np.ndarray:
    """A unit complex tone of ``n`` samples. Cached: the spur and image axes are fixed per configuration."""
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    out = np.exp(2j * np.pi * frequency_hz * t).astype(np.complex64)
    out.flags.writeable = False
    return out


@dataclass(frozen=True)
class Receiver:
    """One X310 receive channel, tuned and configured.

    Args:
        tune_hz: The centre frequency the host asks for. The output band is centred here.
        master_clock_hz: 200e6 or 184.32e6.
        decimation: Host rate is ``master_clock_hz / decimation``; an integer from 1 to 1024.
        lo_offset_hz: How far the analog LO is placed from ``tune_hz``. Bounded by
            :attr:`max_lo_offset_hz`; the digital down-converter shifts the band back, which is what moves
            the DC spur and the image axis away from the band centre.
        gain_db: Receive gain, 0 to 31.5 dB in 0.5 dB steps.
        analog_bandwidth_hz: Front-end bandwidth. 160 MHz above 500 MHz for the UBX-160.
        adc_bits: 14 on the X310.
        adc_full_scale_dbm: Power of a complex sinusoid that just clips the ADC, referred to the ADC input
            after gain. Provisional: to be measured on the unit.
        noise_figure_quantile: Where in the specified noise-figure range this unit sits.
        image_rejection_db: I/Q imbalance image suppression. 55 to 70 dB was measured on real recordings;
            the daughterboard specification only promises better than 30 dB.
        dc_spur_dbfs: Residual DC level after UHD's correction, relative to full scale. UHD nulls the LO
            leakage digitally (``set_dc_offset(True)``, plus the stored calibration the ``uhd_cal_*``
            utilities produce), so what is left is drift with temperature and gain rather than the raw
            leakage. The default of -80 dBFS puts the spur about 7 dB above the noise floor of a 97.7 kHz
            bin at 100 MS/s, in the middle of the 0 to 15 dB that recalibrating against real X310
            recordings produced. Note that a spur is a tone while the floor is noise, so the same residual
            looks 6 dB worse each time the FFT length doubles. Provisional: measure it on the unit.
    """

    tune_hz: float = 2441.75e6
    master_clock_hz: float = 200.0e6
    decimation: int = 2
    lo_offset_hz: float = 0.0
    gain_db: float = 20.0
    analog_bandwidth_hz: float = 160.0e6
    adc_bits: int = 14
    adc_full_scale_dbm: float = -10.0
    noise_figure_quantile: float = 0.5
    image_rejection_db: float = 60.0
    dc_spur_dbfs: float = -80.0
    seed: int = 0
    _imbalance: tuple[float, float] = field(default=(0.0, 0.0), repr=False)

    def __post_init__(self) -> None:
        """Validate the configuration against what the hardware can actually be set to."""
        if not any(math.isclose(self.master_clock_hz, c) for c in MASTER_CLOCKS_HZ):
            raise ValueError(f"the X310 offers master clocks {MASTER_CLOCKS_HZ}, not {self.master_clock_hz / 1e6:.2f} MHz")
        if not isinstance(self.decimation, int) or not 1 <= self.decimation <= MAX_DECIMATION:
            raise ValueError(f"decimation must be an integer in [1, {MAX_DECIMATION}], got {self.decimation!r}")
        if not GAIN_MIN_DB <= self.gain_db <= GAIN_MAX_DB:
            raise ValueError(f"the UBX-160 gives 0 to 31.5 dB of gain, got {self.gain_db}")
        if abs(self.lo_offset_hz) > self.max_lo_offset_hz + 1.0:
            raise ValueError(
                f"|lo_offset_hz| of {abs(self.lo_offset_hz) / 1e6:.3f} MHz would push the {self.sample_rate_hz / 1e6:.3f} "
                f"MS/s span outside the {self.analog_bandwidth_hz / 1e6:.0f} MHz analog passband; the most this "
                f"configuration allows is {self.max_lo_offset_hz / 1e6:.3f} MHz"
            )
        if self.adc_bits < 1 or self.adc_bits > SC16_BITS:
            raise ValueError(f"adc_bits must be in [1, 16] to fit sc16, got {self.adc_bits}")

    # --- geometry -------------------------------------------------------- #

    @property
    def sample_rate_hz(self) -> float:
        """Host sample rate: the master clock divided by the decimation."""
        return self.master_clock_hz / self.decimation

    @property
    def f_low_hz(self) -> float:
        """Lower edge of the delivered span, in Hz."""
        return self.tune_hz - 0.5 * self.sample_rate_hz

    @property
    def f_high_hz(self) -> float:
        """Upper edge of the delivered span, in Hz."""
        return self.tune_hz + 0.5 * self.sample_rate_hz

    @property
    def lo_hz(self) -> float:
        """Where the analog local oscillator actually sits."""
        return self.tune_hz + self.lo_offset_hz

    @property
    def max_lo_offset_hz(self) -> float:
        """The largest LO offset that keeps the whole captured span inside the analog passband."""
        return max(0.0, 0.5 * (self.analog_bandwidth_hz - self.sample_rate_hz))

    @property
    def dc_spur_escapes_band(self) -> bool:
        """Whether an LO offset exists that puts the DC spur outside the captured span.

        True only when ``sample_rate <= analog_bandwidth / 2``. At 100 MS/s on a 160 MHz front end it is
        False, and the stream carries a DC spur that must be masked.
        """
        return self.sample_rate_hz <= 0.5 * self.analog_bandwidth_hz

    @property
    def dc_spur_hz(self) -> float:
        """Absolute frequency of the LO leakage spur in the delivered band."""
        return self.lo_hz

    @property
    def noise_figure_db(self) -> float:
        """The front end's noise figure at the tuned frequency, in dB."""
        return noise_figure_db(self.tune_hz, quantile=self.noise_figure_quantile)

    @property
    def noise_power_dbm(self) -> float:
        """Thermal noise power over the whole captured span, referred to the antenna."""
        return THERMAL_NOISE_DBM_PER_HZ + self.noise_figure_db + 10.0 * math.log10(self.sample_rate_hz)

    def noise_power_dbm_in(self, bandwidth_hz: float) -> float:
        """Thermal noise power in a given bandwidth, which is what an SNR is referred to."""
        if bandwidth_hz <= 0:
            raise ValueError(f"bandwidth_hz must be positive, got {bandwidth_hz}")
        return THERMAL_NOISE_DBM_PER_HZ + self.noise_figure_db + 10.0 * math.log10(bandwidth_hz)

    def snr_db(self, received_power_dbm: float, bandwidth_hz: float) -> float:
        """SNR of a signal of a given received power in its own occupied bandwidth."""
        return received_power_dbm - self.noise_power_dbm_in(bandwidth_hz)

    def baseband_offset_hz(self, absolute_hz: float) -> float:
        """Where an absolute frequency lands in the delivered baseband, which is centred on ``tune_hz``."""
        return absolute_hz - self.tune_hz

    def describe(self) -> dict[str, object]:
        """A record of the configuration, for a stream's provenance."""
        return {
            "model": "x310_ubx160",
            "tune_hz": self.tune_hz,
            "master_clock_hz": self.master_clock_hz,
            "decimation": self.decimation,
            "sample_rate_hz": self.sample_rate_hz,
            "span_hz": [self.f_low_hz, self.f_high_hz],
            "lo_hz": self.lo_hz,
            "lo_offset_hz": self.lo_offset_hz,
            "max_lo_offset_hz": self.max_lo_offset_hz,
            "dc_spur_escapes_band": self.dc_spur_escapes_band,
            "analog_bandwidth_hz": self.analog_bandwidth_hz,
            "gain_db": self.gain_db,
            "adc_bits": self.adc_bits,
            "adc_full_scale_dbm": self.adc_full_scale_dbm,
            "noise_figure_db": self.noise_figure_db,
            "noise_power_dbm": self.noise_power_dbm,
            "image_rejection_db": self.image_rejection_db,
            "dc_spur_dbfs": self.dc_spur_dbfs,
            "dc_spur_above_noise_db_at_100khz": self.dc_spur_above_noise_db(100e3),
            "provisional": ["adc_full_scale_dbm", "dc_spur_dbfs", "image_rejection_db"],
        }

    # --- the chain ------------------------------------------------------- #

    def analog_response(self, n: int) -> np.ndarray:
        """The front end's magnitude response across the captured span, as seen in the delivered baseband.

        Flat where the span is inside the analog passband, rolling off where it is not. The response is
        centred on the LO, not on the tuned frequency, so an LO offset tilts the band.
        """
        return _analog_response(n, self.sample_rate_hz, self.lo_offset_hz, self.analog_bandwidth_hz)

    @property
    def analog_response_is_flat(self) -> bool:
        """Whether the front end's response varies by less than 0.01 dB across the captured span.

        True for a span well inside the analog passband, which lets the chain skip a transform pair.
        """
        edge = 0.5 * self.analog_bandwidth_hz
        worst = 0.5 * self.sample_rate_hz + abs(self.lo_offset_hz)
        ripple_db = 10.0 * math.log10(1.0 + (worst / edge) ** 8)
        return ripple_db < FLAT_RESPONSE_RIPPLE_DB

    def apply_front_end(self, band: np.ndarray, *, rng: np.random.Generator) -> np.ndarray:
        """Everything between the antenna and the ADC, on a chunk of the summed band.

        Takes the clean sum of emitters at the antenna, in sqrt(mW), centred on ``tune_hz``. Applies the
        analog roll-off, thermal noise, the I/Q imbalance image and the DC spur, and returns the signal at
        the ADC input before quantization.
        """
        x = np.asarray(band, dtype=np.complex64)
        n = x.size
        if n == 0:
            return x

        # Analog roll-off, applied in the frequency domain across the span. Skipped when the span sits well
        # inside the passband, where the response is flat to better than 0.01 dB and the transform pair would
        # only cost time.
        if not self.analog_response_is_flat:
            response = self.analog_response(n).astype(np.float32)
            x = np.fft.ifft(np.fft.ifftshift(np.fft.fftshift(np.fft.fft(x)) * response)).astype(np.complex64)

        # Thermal noise over the whole span, referred to the antenna.
        sigma = math.sqrt(10.0 ** (self.noise_power_dbm / 10.0) / 2.0)
        x = x + (rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)).astype(np.complex64)

        # I/Q imbalance: a mirror image about the LO. In the delivered baseband the LO sits at
        # -lo_offset_hz, so the image of a component at f lands at -2*lo_offset - f.
        epsilon = 10.0 ** (-self.image_rejection_db / 20.0)
        if epsilon > 0.0:
            if self.lo_offset_hz == 0.0:
                # The LO is at the band centre, so the image axis is too and the rotation is unity.
                x = x + (epsilon * np.conj(x)).astype(np.complex64)
            else:
                rotate = _tone(n, -2.0 * self.lo_offset_hz, self.sample_rate_hz)
                x = x + (epsilon * np.conj(x) * rotate).astype(np.complex64)

        # Residual DC, which sits at the LO: a constant in the analog baseband, a tone at the LO here.
        full_scale_mw = 10.0 ** (self.adc_full_scale_dbm / 10.0)
        dc_amplitude = math.sqrt(full_scale_mw) * 10.0 ** (self.dc_spur_dbfs / 20.0)
        dc_amplitude /= 10.0 ** (self.gain_db / 20.0)  # referred back to the antenna
        if dc_amplitude > 0.0:
            x = x + np.complex64(dc_amplitude) if self.lo_offset_hz == 0.0 else x + (dc_amplitude * _tone(n, -self.lo_offset_hz, self.sample_rate_hz)).astype(np.complex64)
        return x

    def to_sc16(self, at_adc: np.ndarray) -> np.ndarray:
        """Quantize to the wire format: int16 pairs holding ``adc_bits`` significant bits.

        The X3xx sends sc16 only. The ADC's 14 bits are left-justified into 16, which is what UHD hands a
        host, so the low two bits of the delivered integers are always zero.

        Args:
            at_adc: Signal at the ADC input in sqrt(mW), from :meth:`apply_front_end`.

        Returns:
            int16 array of shape ``[n, 2]``, I in column 0 and Q in column 1.
        """
        x = np.asarray(at_adc, dtype=np.complex64)
        gain = 10.0 ** (self.gain_db / 20.0)
        # Full scale is the amplitude at which a rail just clips.
        full_scale_amplitude = math.sqrt(10.0 ** (self.adc_full_scale_dbm / 10.0))
        rails = np.stack([x.real, x.imag], axis=-1).astype(np.float64) * gain / full_scale_amplitude
        levels = 2 ** (self.adc_bits - 1)
        codes = np.clip(np.round(rails * levels), -levels, levels - 1)
        return (codes.astype(np.int16) << (16 - self.adc_bits)).astype(np.int16)

    def from_sc16(self, sc16: np.ndarray) -> np.ndarray:
        """The inverse of :meth:`to_sc16`: wire integers back to sqrt(mW) at the antenna.

        This is what a host-side front end does first, and what the viewer uses to show power in dBm.
        """
        codes = np.asarray(sc16, dtype=np.float64) / (1 << (16 - self.adc_bits))
        levels = 2 ** (self.adc_bits - 1)
        gain = 10.0 ** (self.gain_db / 20.0)
        full_scale_amplitude = math.sqrt(10.0 ** (self.adc_full_scale_dbm / 10.0))
        rails = codes / levels * full_scale_amplitude / gain
        return (rails[..., 0] + 1j * rails[..., 1]).astype(np.complex64)

    def dc_spur_above_noise_db(self, resolution_bandwidth_hz: float) -> float:
        """How far the residual DC spur stands above the noise floor of one analysis bin.

        This is what decides whether the spur is visible, and it depends on resolution: the spur is a tone
        and the floor is noise, so halving the bin width lifts the spur another 3 dB above it.
        """
        spur_dbm = self.adc_full_scale_dbm + self.dc_spur_dbfs - self.gain_db
        return spur_dbm - self.noise_power_dbm_in(resolution_bandwidth_hz)

    def clipping_headroom_db(self, total_power_dbm: float) -> float:
        """How far the total band power sits below the ADC's full scale, after gain."""
        return self.adc_full_scale_dbm - (total_power_dbm + self.gain_db)
