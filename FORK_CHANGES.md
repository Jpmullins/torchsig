# Changes in this fork

This fork, [Jpmullins/torchsig](https://github.com/Jpmullins/torchsig), is TorchSig 2.2.0 (upstream `main` at
`1197261`) plus opt-in changes developed for [SigGPT](https://github.com/Jpmullins/siggpt), to be offered upstream.
Each change is on its own branch, cut from upstream `main`, so that it can be submitted as its own pull request.
The fork's `main` merges them.

**This file exists only on the fork's `main`.** It is not part of any change and belongs in no upstream pull
request.

| Change | Branch (commits) | Upstream issue | In the fork | Next step upstream |
|---|---|---|---|---|
| Pinnable pulse shape and rolloff for the constellation builder | `constellation-pinned-pulse-shape` (`24473ef`, `681d930`) | [#489](https://github.com/TorchDSP/torchsig/issues/489), open | merged (`348798a`) | open the PR once a maintainer replies on #489 |
| Bluetooth LE at its standard symbol rate, with optional inter-frame gaps | `pr12-btle-standard-rate` (`0a24d8c`, `e5bc624`, `bec2441`) | not posted yet: [appendix](#appendix-issue-g-not-posted-yet) | merged (`d90c232`) | post the issue; open the PR once it is agreed |
| Bounding the LFM and chirp-spread-spectrum sweep length | none | [#490](https://github.com/TorchDSP/torchsig/issues/490), open | no code yet | wait for a reply |
| Location of the cached polyphase filter weights | none | [#491](https://github.com/TorchDSP/torchsig/issues/491), open | no code yet | wait for a reply |

## What holds for every change

- **Opt-in.** With the new arguments or metadata keys unset, the random draws and the output are bit-identical to
  2.2.0 for a fixed seed. Each branch has a committed test against a frozen copy of 2.2.0's code path. Out of tree,
  SHA-256 digests of samples, metadata and generator state were compared before and after each change.
- **The merged `main` passes the suite.** `pytest --ignore=benchmarks --test-mode=fast tests`, with the GPU hidden
  as in CI: 2,851 passed, 9 skipped, 3 deselected. At `1197261`: 2,722 / 9 / 3.
  - The 129 new tests are 65 for the constellation builder and 64 for Bluetooth LE and the shared test helper.
- **Lint.**
  - `pylint --rcfile=.pylintrc torchsig`: 9.74/10, against 9.75 at `1197261`. The difference is two duplicate-code
    messages on argument-validation code that `btle.py` already shares with `lfm.py` and `wifi.py`; none are in
    `btle.py`.
  - `ruff`: no new findings in the changed files, which are formatted.
- **No version bump:** that is left to release PRs.
- **Not run yet:**
  - CI's Python 3.10 on x86_64. Everything above ran on aarch64 with Python 3.12, NumPy 2.5.3 and SciPy 1.18.1.
  - `make test` with pytest-xdist (`-n 2`); the same selection was run sequentially.
  - `make docs`: Sphinx was not installed. The changed docstrings were parsed with Sphinx napoleon/docutils without
    warnings.
  - The example notebooks. No example uses the new options, and every default is unchanged.

## How to submit upstream

1. **Issue first**, as `CONTRIBUTING.md` asks: discuss the change in an issue before the pull request. #489 is
   open; the Bluetooth LE change needs the ISSUE-G text below posted first.
2. **Open each PR from its feature branch, never from this `main`.** Each branch holds only its own files.
3. **Rebase the branch onto current upstream `main` right before opening it.** Rebase, don't merge, so the PR
   carries no upstream history. The two branches touch different files and can go in either order. The staged
   proposal in ISSUE-G would follow with one builder per PR.
4. **Run the checks on CI's platform:** `make test`, `pylint --rcfile=.pylintrc torchsig` and `make docs`.
5. **Use the PR descriptions below.** Fill in the issue number, and update any line numbers that upstream has moved
   since `1197261`.

---

## 1. Pinnable pulse shape and rolloff (`constellation-pinned-pulse-shape`)

`ConstellationSignalGenerator.generate()` draws the pulse shape (SRRC or rectangular, equally likely) and the SRRC
rolloff from its own generator. That choice sets the occupied bandwidth: the 99 % bandwidth is about 1.2 times the
requested bandwidth for SRRC and about 3.3 times for the rectangular pulse. So a caller that places signals itself
cannot know which it will get. Three optional metadata keys, read only when set, pin the choice, the way
`bandwidth_min == bandwidth_max` already pins the bandwidth:
- `pulse_shape_name`: `"srrc"` or `"rectangular"`;
- `alpha_rolloff_min` and `alpha_rolloff_max`: the SRRC rolloff range, pinned when they are equal.

Changed: `torchsig/signals/builders/constellation.py` (+70/−7) and `tests/signals/builders/test_constellation.py`
(+413).

**Choices #489 left open, for a reviewer to check:**
1. `None` counts as not set, so a generator can clear a key pinned at the dataset level.
2. The two rolloff keys must be set together.
3. With a rectangular pulse, the rolloff keys are ignored but still validated. So a bad range fails on every call,
   not only when SRRC is drawn.
4. Validation happens before any value is drawn.
5. A pinned rolloff is returned as a Python `float`, like a drawn one.
6. Keys are read the way the base generator reads optional keys (`self[key] if hasattr(self, key) else None`).
7. The `__init__` docstring gained a blank line so its key list renders as a list.
8. The issue's question, metadata keys or constructor arguments: metadata keys. Constructor keyword arguments
   already become metadata.
9. Out of scope: the dataset's `per_signal_metadata` accepts only SNR, duration and bandwidth keys, so it cannot pin
   the pulse shape per class.

### Draft PR description

```markdown
## Summary

Enhancement (new optional, backward-compatible feature). Refs #489.

`ConstellationSignalGenerator.generate()` draws the pulse shape (SRRC or rectangular, equally likely) and the SRRC rolloff (uniform 0.1-0.5) from its own generator. That choice sets the occupied bandwidth. For the same requested bandwidth, the 99 % occupied bandwidth is about 1.2x the request for SRRC (alpha = 0.35) and about 3.3x for the rectangular pulse, so a caller that places signals itself cannot know which it will get. This PR adds optional metadata keys, consulted only when set, so the pulse shape can be pinned the way bandwidth and duration already are:

- `pulse_shape_name`: `"srrc"` or `"rectangular"`. Skips the pulse-shape draw.
- `alpha_rolloff_min`, `alpha_rolloff_max`: the SRRC rolloff range, set together, with `0 < min <= max < 1`.
  - The rolloff is drawn with `uniform(min, max)`, or is exactly `min` with no draw when `min == max`. This is the convention a pinned `bandwidth_min == bandwidth_max` already follows.
  - The range has no effect on a rectangular pulse but is validated whenever it is set, so a bad range fails on every call, not only when SRRC happens to be drawn.

Behaviour:
- With none of the keys set (or all set to `None`), the draws and the output are bit-identical to 2.2.0 for a fixed seed.
- The keys resolve through the metadata hierarchy, so dataset-level metadata can pin every constellation generator. `None` on a generator clears an inherited pin.
- Invalid values raise `ValueError` before any value is drawn; an unknown pulse shape name lists the valid ones.
- Docstrings are updated (class, `__init__`, `generate()`), including the blank line the `__init__` metadata list needs to render as a list.

On the question in the issue: this uses metadata keys, matching how bandwidth and duration are pinned. Constructor keyword arguments already become metadata, so `ConstellationSignalGenerator(constellation_name="qpsk", pulse_shape_name="srrc")` works without a separate constructor API. No version bump, following the practice of bumping only in release PRs.

## Test Plan

New tests in `tests/signals/builders/test_constellation.py` (65 cases). The module's 40 existing tests pass unchanged, including the two that assert the exact `integers(0, 2)` and `uniform(0.1, 0.5)` calls.

- **Default path:** bit-identical to a verbatim copy of the 2.2.0 `generate()` (samples as bytes, `to_dict()` metadata, final generator state), for 7 classes x 5 seeds x 4 consecutive signals, covering both pulse shapes.
- **Pinned `"srrc"`:** metadata reports SRRC, the rolloff lies in the pinned range, and samples and generator state equal a replay of exactly the expected draws (no pulse-shape draw).
- **Pinned `"rectangular"`:** rolloff `None`, index 0, target 0.0. The replay matches, and adding a rolloff range changes nothing.
- **`alpha_rolloff_min == alpha_rolloff_max`:** the rolloff is exactly the pinned value, and the generator's stream ends where a replay without a rolloff draw ends. A replay that also makes the skipped `uniform(a, a)` draw (which returns `a`, so the value alone cannot reveal it) must not match, so the check detects one extra draw.
- **Other cases:** a rolloff range without `pulse_shape_name` keeps the pulse-shape draw; the keys resolve through a parent's metadata; `None` behaves as absent.
- **Invalid inputs:** unknown pulse shapes and bad rolloff ranges (bounds outside (0, 1), min > max, NaN, only one bound set) raise `ValueError`, and the generator state is unchanged afterwards.
- **Occupied bandwidth:** 99 % occupied bandwidth (Welch, 0.5 %/99.5 % points) for BPSK, QPSK, 16-QAM and OOK.
  - SRRC must be at most (1 + alpha) x BW plus one Welch bin: SRRC is band-limited to (1 + alpha) times the symbol rate, which the modulator sets equal to the requested bandwidth.
  - Rectangular must exceed 2 x BW: it keeps only about 93 % of its power (about 96 % for OOK) inside its main lobe (+/- BW).
  - Measured: about 1.2x and 3.3x.

The tests were checked by injecting defects. Each of these makes at least one new test fail: an extra `uniform(a, a)` draw, a pulse-shape draw when pinned, a default range changed from 0.5 to 0.49, validation moved after the draws, and range validation skipped for the rectangular pulse.

Out-of-tree golden check: SHA-256 of the samples, all metadata, and the generator state after every call, recorded before and after the change. It covered 22 constellation classes x 7 seeds x 2 configurations x 4 consecutive signals, via both `__call__` and `generate()` (2,464 signals), plus 50 samples from a `TorchSigIterableDataset` mixing constellation, AM and FSK classes. The results are byte-identical.

Run on Linux aarch64 with Python 3.12, NumPy 2.5.3 and SciPy 1.18.1; not yet run on CI's Python 3.10 on x86_64.
- `pytest --ignore=benchmarks --test-mode=fast tests`: 2787 passed, 9 skipped, 3 deselected (`main`: 2722 / 9 / 3). This is `make test` without `-n 2`, as pytest-xdist was not installed.
- `pylint --rcfile=.pylintrc torchsig`: 9.75/10, unchanged, with no new messages.
- `ruff check` and `ruff format --check`: no new findings in the changed files.
- Docstrings render through Sphinx napoleon/docutils without warnings; `make docs` was not run.

## Before Submitting
- [x] Check for bugs/errors
    - [ ] Run example notebooks: not run. No notebook uses the builder's pulse-shape selection, and `examples/scripts/train_modrec_metadata_targets.py` trains on the default 50/50 draw, which is unchanged.
    - [x] Write or update unit tests in `tests/`
    - [x] Run Pytest: `make test` (same command, run sequentially); all tests pass.
- [x] Run Pylint: `pylint --rcfile=.pylintrc torchsig`
    - [x] Score > 9/10 (9.75)
    - [x] Code conforms with PEP 8 (ruff format)
    - [x] Google-style docstrings used to document code
- [ ] Added contributers to PR: n/a, single author.
```

---

## 2. Bluetooth LE at its standard symbol rate (`pr12-btle-standard-rate`)

The protocol builders set their symbol or chip rate equal to the requested `bandwidth`, so a Bluetooth LE burst is
generated at whatever rate the dataset draws. This change adds, all opt-in with unchanged defaults:
- `BTLE_SYMBOL_RATE_HZ = 1.0e6`, citing the Bluetooth Core Specification;
- `btle_modulator(..., pin_rate_to_standard=True)`, which generates at 1 Msym/s and raises a `ValueError` below
  2 MS/s;
- `inter_frame_gap_symbols`, silent symbols after each packet, with the carrier gated off;
- a shared test helper, `estimate_symbol_rate`, that later builders can reuse.

It is the first builder of the staged proposal in ISSUE-G.

Changed: `torchsig/signals/builders/btle.py` (+109/−22), `tests/signals/builders/test_btle.py`
(+293/−1) and the new `tests/signals/builders/test_builders_utils.py` (+74).

Measured symbol rate in standard mode: 1,000,000.00 Hz at 2 MS/s, and within 61 ppm at 7.68, 10 and 20 MS/s (the
estimator's grid; the rate generated at 10 MS/s is exactly 1 MHz).

**Choices beyond ISSUE-G's text, for a reviewer to check:**
- **Gaps gate the carrier.** In GFSK a zero symbol leaves the carrier on at full power, so zeros alone never make
  a gap. The envelope is also gated, with the same Gaussian pulse, so the ramps take about 0.6 symbol at each end.
  The duty-cycle criterion therefore holds for gaps of about 7 symbols or more. Tests use 16 and 150 (150 µs, the
  standard inter-frame space).
- **The symbol-rate test helper** uses the real part of the lag product and picks the lag whose feature stands out
  most. The textbook estimator locked onto 125 kHz, because every BLE field is a whole number of bytes. It measures
  only bursts without gaps.
- **Golden tests** compare against a frozen copy of the old code, not a stored array.
- **Tolerance:** the rate test's is tighter than the spec's 0.1 %: one grid step plus 100 ppm, at most 405 Hz.
- **Docstring corrections:** the physical layer is cited as Vol 6 Part A (it said Part B, the link layer). A note
  was added that this model sends bits MSB first, while BLE sends LSB first.
- **Left for the dataset-level PR:** the `protocol_timing` key, and the documentation rows.
- **Noted for the dataset-level PR:** `update_signal_snr_bandwidth` averages the spectrogram in dB over time. So
  silent gap frames would label packets below their actual SNR (see #488 and question 6 of ISSUE-G).

### Draft PR description

```markdown
## Summary

New feature (opt-in, additive): standard-pinned symbol timing and inter-frame gaps for the BLE builder (`torchsig/signals/builders/btle.py`). Refs #<ISSUE-G number>. This is builder 1 of the staged series proposed there; the dataset-level `protocol_timing` key will be a separate PR.

- `BTLE_SYMBOL_RATE_HZ = 1.0e6`, citing Bluetooth Core Specification v5.4, Vol 6, Part A, Section 1 ("The symbol rate is 1 Msym/s").
- `btle_modulator(..., *, pin_rate_to_standard=False)`: when True, generates at `BTLE_SYMBOL_RATE_HZ` instead of deriving the symbol rate from `bandwidth`. If `sample_rate < 2 * BTLE_SYMBOL_RATE_HZ` (the builder's existing `bandwidth <= sample_rate / 2` rule applied to the standard rate), it raises `ValueError` naming the class, the required rate and the sample rate. The multistage resampler adds no constraint; it realizes the rate within 100 ppm (exactly at 10 MS/s).
- `inter_frame_gap_symbols: int = 0` (keyword-only, threaded through `btle_modulator` and `btle_modulator_baseband` to `build_btle_bit_stream`): zero symbols after every packet, before pulse shaping. Because GFSK has a constant envelope, a zero symbol alone would leave the carrier on, so the baseband modulator also gates the envelope with the on/off symbol sequence shaped by the same Gaussian pulse (about one symbol of ramp at each gap).
- Defaults are unchanged: output is bit-identical to 2.2.0 for a fixed seed.
- Module docstring: describes both timing modes and which toy simplifications remain in standard mode; the physical-layer citation now points to Vol 6 Part A (it said Part B, the link layer).

## Test Plan

New tests in `tests/signals/builders/test_btle.py`, plus a shared helper `estimate_symbol_rate(x, fs)` in `tests/signals/builders/test_builders_utils.py` that later builders can reuse. The helper takes the strongest cyclic-autocorrelation feature of Re(x[n+tau] conj(x[n])) and is itself tested on rectangular-pulse QPSK of known rate, to one FFT grid step.

- `test_default_output_unchanged`: bit-for-bit comparison with a frozen copy of the 2.2.0 default path, run on the test platform (a stored array cannot match bit for bit across x86-64 and aarch64), over the interpolating, fractional, pass-through, decimating and bandwidth = fs/2 paths.
- `test_standard_rate_symbol_rate`: at 2, 7.68, 10 and 20 MS/s the measured rate is within one FFT grid step plus 100 ppm of 1 Msym/s (at most 405 Hz, inside the 0.1 % criterion); the same request in the default mode measures as the bandwidth.
- `test_standard_rate_unrepresentable_raises`: 0.25, 1.5 and 1.999999 MS/s.
- `test_inter_frame_gap_duty_cycle`: samples below 1 % of burst RMS are within 20 % of g / (g + frame length), for g = 16 and 150.
- `test_frame_period_matches_standard`: intervals between preamble and access-address correlation peaks equal (preamble + AA + PDU + CRC + gap) / symbol rate within one symbol.
- `test_dtype_length_finite` (every mode), `test_bit_stream_inter_frame_gap`, `test_inter_frame_gap_invalid_raises`.

Run on aarch64 Linux with Python 3.12, numpy 2.5.3, scipy 1.18.1: `pytest --ignore=benchmarks --test-mode=fast tests` gave 2786 passed, 9 skipped, 0 failed. Not yet run on Python 3.10 or x86-64.

## Before Submitting
- [x] Check for bugs/errors
    - [ ] Run example notebooks: not run; no example uses the new arguments.
    - [x] Write or update unit tests in `tests/`
    - [x] Run Pytest: `make test` could not run here (pytest-xdist not installed), so the same selection was run sequentially: 2786 passed, 0 failed.
- [x] Run Pylint: `pylint --rcfile=.pylintrc torchsig`
    - [x] Score > 9/10: 9.74 (9.75 on main; two extra duplicate-code messages on the builders' shared argument-validation block, none in `btle.py`)
    - [x] Code conforms with PEP 8 (`ruff format --check` clean for the changed files)
    - [x] Google-style docstrings used to document code
- [ ] Added contributors to PR
```

---

## Appendix: ISSUE-G (not posted yet)

Drafted on 2026-09-30, with an existing-request search of issues, pull requests and discussions: no matching
request. Every file and line it cites was re-read on upstream `main` at `1197261`. It should be posted as a Feature
Request before the Bluetooth LE PR.

Two sequencing notes from the contribution plan:
- **ISSUE-A.** The plan puts ISSUE-G after ISSUE-A (what `bandwidth` denotes), which has not been posted. Question 4
  below carries the part ISSUE-G depends on, so ISSUE-G can go first.
- **PR-6.** A bandwidth-calibration change (PR-6), if it lands first, touches the same rate lines. The Bluetooth LE
  branch would then need a rebase.

**Title.** `[Feature] Let the protocol builders run at their standard symbol and chip rates, with optional gaps between frames`

**Body.**

> ### Is there an existing feature already?
>
> - [x] Yes, I have checked the existing features.
>
> ### Description
>
> The protocol builders added in 2.2.0 build their frames in symbols, then set the symbol or chip rate (for 802.11a,
> the channel's sample rate) equal to the requested `bandwidth`. Each modulator computes
> `oversampling_rate = sample_rate / bandwidth`. It then resamples, by a factor of `oversampling_rate / 4`, a
> baseband stream built at 4 samples per symbol, chip or channel sample (`torchsig/signals/builders/`:
> `adsb.py:157`, `btle.py:167`, `cellular.py:189`, `dvb.py:291`, `lmr.py:241` and `502`, `lora.py:177`,
> `wifi.py:354`, `zigbee.py:220`). The DMR notes in `lmr.py` state the intent: "The absolute deviation/symbol-rate
> ratios (i.e. the 4FSK *structure*) are preserved, while the occupied bandwidth is scaled to the dataset-requested
> bandwidth via resampling, exactly like the other TorchSig builders." (`lmr.py:38-40`).
>
> So frames keep their length in symbols, and their duration scales with the requested bandwidth. At 10 MS/s:
>
> - `p25` requested at 1 MHz runs at 1.0 MBd (measured) and occupies 1.51 MHz (99 % power, measured). P25 Phase 1 is
>   4,800 Bd in a 12.5 kHz channel (`lmr.py:342-346`).
> - `80211a` requested at 62.5 kHz occupies 52.5 kHz (measured). Its 64-point FFT spans 62.5 kHz, so the subcarrier
>   spacing is 977 Hz and an OFDM symbol lasts 1.28 ms. 802.11a uses 312.5 kHz and 4 µs.
>
> With the wideband defaults (`bandwidth_min: 62_500`, `bandwidth_max: 1_000_000`), every protocol class draws its
> rate from the same range, so symbol duration carries no information about the class. Frame durations are off by
> the same factor: a BLE advertising packet (80 to 376 symbols in this builder) lasts 80–376 µs at 1 Msym/s and up
> to 6 ms at 62.5 kHz. The narrowband defaults draw 2.5–3.33 MHz, above every standard rate here except 802.11a's.
>
> Second, frames are sent back to back. `build_btle_bit_stream` (`btle.py:56-83`) appends packets until the
> requested length is reached, and the ADS-B, ZigBee, DMR and P25 stream builders do the same. `btle.py:19` and
> `zigbee.py:19` list "Packets are concatenated without inter-frame gaps" as a toy simplification. GSM has 9 guard
> bits per burst, documented as "modeled as zeros (off-air silence)" (`cellular.py:37`). But `build_gsm_burst` maps
> them to −1 symbols along with the data (`cellular.py:73-74`), and the output is `np.exp(1j * phase)`
> (`cellular.py:145`), so the envelope stays at 1 through the guard period. In the constant-envelope builders (GFSK,
> GMSK, CPFSK) an idle symbol still transmits. A gap needs the carrier switched off.
>
> Part of this is possible today. `per_signal_metadata` with `bandwidth_min == bandwidth_max` pins one class's rate
> exactly, even outside the dataset's range. `{"btle": {"bandwidth_min": 1_000_000, "bandwidth_max": 1_000_000}}`
> gives 1 Msym/s. The same with 2 MHz for `zigbee` gives 2 Mchips/s while the dataset's `bandwidth_max` is 1 MHz
> (both measured). But this route has four limits:
>
> - the user has to know each builder's mapping and each standard's value;
> - it cannot be reached from a YAML config: `TorchSigDatasetConfig` has no per-class field, and the DataModule does
>   not pass `per_signal_metadata`;
> - it cannot express LoRa's set of bandwidths;
> - nothing in it adds gaps.
>
> Standard rates against the shipped wideband configs (10 MS/s, 62.5 kHz–1 MHz, bursts of 16,384–32,768 samples):
>
> | class | standard rate | inside 62.5 kHz–1 MHz | builder check at 10 MS/s (`rate <= sample_rate / 2`) | note |
> |---|---|---|---|---|
> | `btle` | 1 Msym/s | yes (the maximum) | passes | |
> | `gsm` | 270,833.3 sym/s (13/48 MHz) | yes | passes | |
> | `lora` | chip rate = bandwidth: 125, 250 or 500 kHz in LoRaWAN | yes | passes | drawn from a continuous range today |
> | `adsb-long`, `adsb-short` | 2 Mchips/s (0.5 µs chips, `adsb.py:7`) | no | passes | occupies about 6.6 MHz |
> | `zigbee` | 2 Mchips/s | no | passes | occupies about 2.5 MHz |
> | `dmr`, `p25` | 4,800 sym/s | no | passes | a 1.6–3.3 ms burst holds 8–16 symbols, less than one 24-symbol sync word |
> | `80211a` | 20 MHz channel (64 × 312.5 kHz) | no | fails below 40 MS/s | |
> | `dvbs2` | set by the service | – | – | no single standard rate |
>
> Proposal: opt-in and additive, one builder per PR. Bluetooth LE goes first, then GSM, LoRa, ADS-B, ZigBee,
> DMR/P25 and 802.11a; DVB-S2 is documented as not applicable. Per builder:
>
> 1. A module constant for the standard rate, with a citation (`BTLE_SYMBOL_RATE_HZ = 1.0e6`). `DMR_SYMBOL_RATE` and
>    `P25_SYMBOL_RATE` (`lmr.py:44`, `358`) already exist, but they only set the modulation index.
> 2. A keyword-only `pin_rate_to_standard: bool = False` on the modulator. When it is true, `bandwidth` no longer sets
>    the rate.
> 3. A `ValueError` naming the class, the required rate and the sample rate, raised when the standard rate breaks the
>    builder's own `rate <= sample_rate / 2` rule.
> 4. For BLE, ZigBee, ADS-B, GSM and DMR, a keyword-only `inter_frame_gap_symbols: int = 0`: silent symbols after
>    each frame. In the constant-envelope builders, the envelope is gated with the same pulse shape, so the carrier
>    ramps off and on over about one symbol.
> 5. With the defaults, output is bit-identical to the previous commit for a fixed seed.
>
> Test plan, per builder:
>
> - default output is bit-identical to the previous commit;
> - the symbol or chip rate is within 0.1 % of the standard, measured by cyclic autocorrelation with a shared test
>   helper that is itself tested on rectangular-pulse QPSK of known rate;
> - the unrepresentable case raises;
> - with gaps, the silent fraction and the spacing of preamble correlation peaks match the gap and frame lengths.
>
> A prototype of the Bluetooth LE step is on my fork. With `pin_rate_to_standard=True`, the measured rate is 1 Msym/s
> to within the estimator's FFT grid: 0 ppm at 2 MS/s, −39 ppm at 7.68 MS/s, and +61 ppm at 10 and 20 MS/s. With both
> options at their defaults, the output is byte-identical to 2.2.0. I will open it as a PR once the questions below
> are settled.
>
> Gaps interact with #488. `update_signal_snr_bandwidth` averages the spectrogram over time in dB
> (`torchsig/utils/dsp.py:1465`), and `compute_spectrogram` floors empty bins 100 dB below the peak
> (`dsp.py:1099-1102`). Silent frames therefore pull the estimate down, and the correction raises the packets.
>
> - **BLE with gaps (the prototype).** At 1 Msym/s with 150-symbol gaps (the 150 µs inter-frame space) in
>   32,768-sample bursts, 24 % of spectrogram frames are silent. After the correction, the packets sit 27 dB above
>   their SNR label (24–32 dB over 20 seeds), against 2 dB without gaps.
> - **802.11a control frames (today).** They are zero-padded to the burst length (`wifi.py:334`), and already show
>   this. `80211a_rts` and `80211a_ack` (registered, but not in the public class list) at 1 MHz in 32,768-sample
>   bursts end up 67 and 71 dB above their label.
> - **With #488's fix.** With the linear average proposed in #488, the BLE offset drops to about 2 dB, roughly the
>   ratio of packet power to burst-average power.
>
> Questions to settle before any code:
>
> 1. Given that `per_signal_metadata` can already pin a class's rate, do you want a standard-rate option at all? The
>    alternative is the constants, docstrings stating what `bandwidth` sets in each builder, and a documented
>    `per_signal_metadata` recipe. The gaps need code either way.
> 2. Keyword arguments on the modulator functions only, or also an optional metadata key read by `generate()`? A key
>    in `dataset_metadata` already resolves in every generator. One key, such as
>    `protocol_timing: "scaled" | "standard"` (default `"scaled"`), would therefore reach every protocol builder from
>    a YAML config with no change to the dataset classes, as #489 proposes for the pulse shape. Do you want a
>    dataset-level key, and under what name?
> 3. When the standard rate cannot be generated at the sample rate: raise, skip the class, or fall back to the scaled
>    rate with a warning? I suggest raising, so that a misconfigured dataset fails at construction.
> 4. In standard mode, what should `generate()` report in `Signal.bandwidth`: the drawn value, the standard rate, or
>    the occupied bandwidth at that rate (about 1.05 MHz for BLE)? The dataset replaces it with its own estimate only
>    when some bin of the burst's max-hold spectrum exceeds the noise floor by 3 dB (`dsp.py:1482-1517`), and callers
>    of the builders see it directly.
> 5. Is gating the envelope acceptable for gaps? And should the GSM guard bits become silent, which changes the
>    default output, or should `cellular.py:37` be corrected to describe what is generated?
> 6. For a burst with gaps, should the SNR label describe the packets or the burst average? And should the gap
>    option wait for #488?
