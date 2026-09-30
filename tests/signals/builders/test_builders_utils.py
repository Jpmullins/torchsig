"""Measurement helpers shared by the signal-builder tests, with tests of the helpers."""

import numpy as np
import pytest


def estimate_symbol_rate(x: np.ndarray, fs: float) -> float:
    """Estimates the symbol rate of a complex baseband signal from its cyclic autocorrelation.

    A signal with symbol rate R is cyclostationary: the mean of the lag product
    v[n] = Re(x[n + tau] * conj(x[n])) = I[n + tau] * I[n] + Q[n + tau] * Q[n]
    is periodic in n with period fs / R samples, which puts lines, the cyclic
    features, at cycle frequencies alpha = k * R. For the lags tau = 1, 2, 4, ...
    up to len(x) // 256 samples, this computes the cyclic autocorrelation magnitude

        C(alpha, tau) = |mean_n (v[n] - mean(v)) * exp(-2j * pi * alpha * n / fs)|

    on the FFT grid alpha = k * fs / len(x), 0 < alpha <= fs / 2. It keeps the lag
    whose largest C stands out most from that lag's RMS over alpha and returns the
    alpha of that largest C.

    Design notes:
        * The real part carries the whole symbol-rate feature, because the periodic
          mean of x[n + tau] * conj(x[n]) is real for a signal centered at 0 Hz with
          a symmetric constellation. It also drops the odd, first-order phase term
          of a constant-envelope signal, whose low-frequency spectrum would
          otherwise outweigh the feature.
        * The fundamental (k = 1) is strongest for lags near half a symbol, so the
          lag set assumes the record spans at least a few hundred symbols. Ranking
          lags by peak over RMS rather than by peak stops a long lag, whose spectrum
          is mostly low-frequency noise from the random phase, from winning.
        * The strongest feature is the symbol rate for a gapless burst. An on/off
          envelope, such as a burst with inter-frame gaps, adds stronger features
          at the frame rate.

    Args:
        x: Complex baseband samples centered at 0 Hz.
        fs: Sample rate (Hz).

    Returns:
        float: Estimated symbol rate (Hz), a multiple of fs / len(x). When the
        symbol-rate line dominates its neighborhood, the estimate is within one grid
        step, fs / len(x), of the rate.
    """
    x = np.asarray(x, dtype=np.complex128)
    num_samples = len(x)
    cycle_frequencies = np.arange(1, num_samples // 2 + 1) * fs / num_samples
    best_prominence, best_rate = -np.inf, np.nan
    lag = 1
    while lag <= max(1, num_samples // 256):
        v = (x[lag:] * np.conj(x[:-lag])).real
        c = np.abs(np.fft.fft(v - v.mean(), num_samples)[1 : num_samples // 2 + 1]) / len(v)
        prominence = c.max() / np.sqrt(np.mean(c**2))
        if prominence > best_prominence:
            best_prominence, best_rate = prominence, cycle_frequencies[np.argmax(c)]
        lag *= 2
    return float(best_rate)


@pytest.mark.parametrize("symbol_rate", [314_159.3, 1_234_567.0, 2_200_000.0])
def test_estimate_symbol_rate_rectangular_qpsk(symbol_rate):
    """The estimator recovers the rate of rectangular-pulse QPSK to one FFT grid step.

    Holding each random QPSK symbol for fs / symbol_rate samples (31.8, 8.1 and 4.5
    here, none of them an integer) gives a signal of exactly known rate that is not
    a multiple of the grid step fs / len(x), so the tolerance is that step: 153 Hz
    for 2**16 samples at 10 MS/s.
    """
    fs, num_samples = 10_000_000.0, 2**16
    rng = np.random.default_rng(5)
    constellation = np.exp(1j * np.pi * (2 * np.arange(4) + 1) / 4)
    symbols = constellation[rng.integers(0, 4, int(np.ceil(num_samples * symbol_rate / fs)) + 1)]
    x = symbols[(np.arange(num_samples) * symbol_rate / fs).astype(int)]
    assert abs(estimate_symbol_rate(x, fs) - symbol_rate) <= fs / num_samples
