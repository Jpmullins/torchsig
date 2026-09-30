"""Unit tests for the constellation signal builder and modulator."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import scipy.signal as sp

from torchsig.signals.builders.constellation import (
    ConstellationSignalGenerator,
    constellation_modulator,
    constellation_modulator_baseband,
)
from torchsig.signals.signal_types import Signal
from torchsig.utils.abstractions import HierarchicalMetadataObject
from torchsig.utils.dsp import TorchSigComplexDataType

MODULE_PATH = "torchsig.signals.builders.constellation"

PINNING_METADATA = {
    "sample_rate": 10_000_000,
    "bandwidth_min": 500_000,
    "bandwidth_max": 2_000_000,
    "signal_duration_in_samples_min": 2048,
    "signal_duration_in_samples_max": 4096,
}


@pytest.mark.parametrize("max_num_samples", [0, -1, -100])
def test_constellation_modulator_baseband_rejects_nonpositive_max_samples(
    max_num_samples,
):
    """Nonpositive baseband output lengths should be rejected."""
    with pytest.raises(
        ValueError,
        match="max_num_samples must be positive",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            max_num_samples=max_num_samples,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("oversampling_rate", [0, -1, -10])
def test_constellation_modulator_baseband_rejects_nonpositive_oversampling_rate(
    oversampling_rate,
):
    """Nonpositive nominal oversampling rates should be rejected."""
    with pytest.raises(
        ValueError,
        match="oversampling_rate_nominal must be positive",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            max_num_samples=128,
            oversampling_rate_nominal=oversampling_rate,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_creates_default_rng():
    """A default NumPy generator should be created when none is supplied."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
    ):
        result = constellation_modulator_baseband(
            constellation_name="bpsk",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
        )

    default_rng.assert_called_once_with()
    assert result.shape == (8,)


def test_constellation_modulator_baseband_rejects_unknown_pulse_shape():
    """Unsupported pulse-shaping filters should be rejected."""
    with pytest.raises(
        ValueError,
        match="pulse shape invalid not supported",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="invalid",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_srrc_requires_alpha_rolloff():
    """SRRC pulse shaping should require an alpha-rolloff value."""
    with pytest.raises(
        ValueError,
        match="must define an alpha rolloff for SRRC filter",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="srrc",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            alpha_rolloff=None,
            rng=np.random.default_rng(42),
        )


@pytest.mark.parametrize("alpha_rolloff", [-1.0, 0.0, 1.0, 1.5])
def test_constellation_modulator_baseband_rejects_invalid_alpha_rolloff(
    alpha_rolloff,
):
    """SRRC alpha rolloff should be strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="alpha_rolloff must be between 0 and 1",
    ):
        constellation_modulator_baseband(
            constellation_name="qpsk",
            pulse_shape_name="srrc",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            alpha_rolloff=alpha_rolloff,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_raises_for_unknown_constellation():
    """An unknown constellation name should raise a KeyError."""
    with pytest.raises(KeyError):
        constellation_modulator_baseband(
            constellation_name="not-a-constellation",
            pulse_shape_name="rectangular",
            max_num_samples=128,
            oversampling_rate_nominal=4,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_baseband_rectangular_pulse_shape():
    """Rectangular pulse shaping should use one tap per sample per symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    normalized_map = np.array([-1 + 0j, 1 + 0j])
    expected_symbols = normalized_map[[0, 1]]

    np.testing.assert_array_equal(
        upfirdn.call_args.args[0],
        np.ones(4),
    )
    np.testing.assert_array_equal(
        upfirdn.call_args.args[1],
        expected_symbols,
    )
    assert upfirdn.call_args.kwargs == {
        "up": 4,
        "down": 1,
    }

    assert result.dtype == np.dtype(TorchSigComplexDataType)


def test_constellation_modulator_baseband_normalizes_symbol_map():
    """The constellation map should be normalized to average unit power."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    raw_map = np.array([-2 + 0j, 2 + 0j])
    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": raw_map},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    symbols = upfirdn.call_args.args[1]

    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)


def test_constellation_modulator_baseband_srrc_designs_expected_filter():
    """SRRC pulse shaping should derive its span and taps."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0])

    pulse_shape = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
    shaped = np.ones(32, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.estimate_filter_length",
            return_value=17,
        ) as estimate_length,
        patch(
            f"{MODULE_PATH}.srrc_taps",
            return_value=pulse_shape,
        ) as taps,
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="srrc",
            max_num_samples=32,
            oversampling_rate_nominal=4,
            alpha_rolloff=0.25,
            rng=rng,
        )

    estimate_length.assert_called_once_with(
        0.25,
        120,
        1,
    )

    # ceil((17 - 1) / (2 * 4)) = 2
    taps.assert_called_once_with(
        4,
        2,
        0.25,
    )

    np.testing.assert_array_equal(
        upfirdn.call_args.args[0],
        pulse_shape,
    )
    assert result.shape == (32,)


def test_constellation_modulator_baseband_accounts_for_srrc_filter_span():
    """SRRC transient symbols should be removed from the symbol count."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1, 0, 1])

    shaped = np.ones(32, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.estimate_filter_length",
            return_value=17,
        ),
        patch(
            f"{MODULE_PATH}.srrc_taps",
            return_value=np.ones(17),
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
    ):
        constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="srrc",
            max_num_samples=32,
            oversampling_rate_nominal=4,
            alpha_rolloff=0.25,
            rng=rng,
        )

    # floor(32 / 4) - 2 * span
    # 8 - 2 * 2 = 4 symbols
    rng.integers.assert_called_once_with(
        low=0,
        high=2,
        size=4,
    )


def test_constellation_modulator_baseband_generates_at_least_one_symbol():
    """Very short requests should still generate one symbol."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([1])

    shaped = np.ones(4, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=np.ones(1, dtype=np.complex64),
        ),
    ):
        constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=1,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    rng.integers.assert_called_once_with(
        low=0,
        high=2,
        size=1,
    )


def test_constellation_modulator_baseband_retries_all_zero_ook_symbols():
    """OOK generation should retry when every selected symbol is zero."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        np.array([0, 0]),
        np.array([0, 1]),
    ]

    raw_symbol_map = np.array([0 + 0j, 1 + 0j])
    shaped = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"ook-test": raw_symbol_map},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ) as upfirdn,
    ):
        constellation_modulator_baseband(
            constellation_name="ook-test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    assert rng.integers.call_count == 2

    expected_symbol_map = raw_symbol_map / np.sqrt(np.mean(np.abs(raw_symbol_map) ** 2))

    np.testing.assert_allclose(
        upfirdn.call_args.args[1],
        expected_symbol_map[[0, 1]],
    )


def test_constellation_modulator_baseband_pads_short_signal():
    """A short pulse-shaped result should be padded to the requested length."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(6, dtype=np.complex64)
    padded = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad,
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    pad.assert_called_once_with(
        shaped,
        8,
    )
    slice_tail.assert_not_called()

    np.testing.assert_array_equal(result, padded)


def test_constellation_modulator_baseband_slices_long_signal():
    """A long pulse-shaped result should be sliced from the tail."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.ones(10, dtype=np.complex64)
    sliced = np.ones(8, dtype=np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
            return_value=sliced,
        ) as slice_tail,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    slice_tail.assert_called_once_with(
        shaped,
        8,
    )
    pad.assert_not_called()

    np.testing.assert_array_equal(result, sliced)


def test_constellation_modulator_baseband_leaves_exact_length_unchanged():
    """An exact-length pulse-shaped result should need no adjustment."""
    rng = MagicMock(spec=np.random.Generator)
    rng.integers.return_value = np.array([0, 1])

    shaped = np.arange(8, dtype=np.float32).astype(np.complex64)

    with (
        patch.dict(
            f"{MODULE_PATH}.all_symbol_maps",
            {"test": np.array([-1 + 0j, 1 + 0j])},
        ),
        patch(
            f"{MODULE_PATH}.sp.upfirdn",
            return_value=shaped,
        ),
        patch(
            f"{MODULE_PATH}.slice_tail_to_length",
        ) as slice_tail,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad,
    ):
        result = constellation_modulator_baseband(
            constellation_name="test",
            pulse_shape_name="rectangular",
            max_num_samples=8,
            oversampling_rate_nominal=4,
            rng=rng,
        )

    slice_tail.assert_not_called()
    pad.assert_not_called()

    np.testing.assert_array_equal(
        result,
        shaped.astype(TorchSigComplexDataType),
    )


@pytest.mark.parametrize(
    ("bandwidth", "sample_rate", "num_samples", "expected_message"),
    [
        (0, 10_000, 128, "bandwidth must be positive"),
        (-1, 10_000, 128, "bandwidth must be positive"),
        (1_000, 0, 128, "sample_rate must be positive"),
        (1_000, -1, 128, "sample_rate must be positive"),
        (
            5_001,
            10_000,
            128,
            "bandwidth must be less than sample_rate/2",
        ),
        (1_000, 10_000, 0, "num_samples must be positive"),
        (1_000, 10_000, -1, "num_samples must be positive"),
    ],
)
def test_constellation_modulator_rejects_invalid_inputs(
    bandwidth,
    sample_rate,
    num_samples,
    expected_message,
):
    """Invalid top-level modulation parameters should be rejected."""
    with pytest.raises(ValueError, match=expected_message):
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=bandwidth,
            sample_rate=sample_rate,
            num_samples=num_samples,
            rng=np.random.default_rng(42),
        )


def test_constellation_modulator_creates_default_rng():
    """A default generator should be created when rng is omitted."""
    rng = MagicMock(spec=np.random.Generator)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.np.random.default_rng",
            return_value=rng,
        ) as default_rng,
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
        )

    default_rng.assert_called_once_with()


def test_constellation_modulator_calculates_resampling_parameters():
    """The wrapper should calculate the expected baseband length and rate."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
            return_value=baseband,
        ) as baseband_modulator,
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ) as resampler,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="srrc",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            alpha_rolloff=0.25,
            rng=rng,
        )

    # oversampling_rate = 10
    # resample_rate_ideal = 10 / 4 = 2.5
    # num_samples_baseband = floor(100 / 2.5) = 40
    baseband_modulator.assert_called_once_with(
        "qpsk",
        "srrc",
        40,
        4,
        0.25,
        rng,
    )

    resampler.assert_called_once_with(
        baseband,
        2.5,
    )


def test_constellation_modulator_uses_minimum_baseband_length():
    """At least four baseband samples should be requested."""
    rng = np.random.default_rng(42)
    baseband = np.ones(4, dtype=np.complex64)
    resampled = np.ones(1, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
            return_value=baseband,
        ) as baseband_modulator,
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=resampled,
        ),
    ):
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1,
            sample_rate=10_000,
            num_samples=1,
            rng=rng,
        )

    baseband_modulator.assert_called_once_with(
        "qpsk",
        "rectangular",
        4,
        4,
        None,
        rng,
    )


def test_constellation_modulator_slices_long_resampled_signal():
    """An oversized resampled signal should be sliced to the target length."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(110, dtype=np.complex64)
    sliced = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
            return_value=sliced,
        ) as slice_signal,
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
        ) as pad_signal,
    ):
        result = constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    slice_signal.assert_called_once_with(
        resampled,
        100,
    )
    pad_signal.assert_not_called()

    assert result.shape == (100,)
    assert result.dtype == np.dtype(TorchSigComplexDataType)


@pytest.mark.parametrize("resampled_length", [90, 100])
def test_constellation_modulator_pads_signal_not_longer_than_target(
    resampled_length,
):
    """A resampled signal no longer than the target should use padding."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(resampled_length, dtype=np.complex64)
    padded = np.ones(100, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=padded,
        ) as pad_signal,
        patch(
            f"{MODULE_PATH}.slice_head_tail_to_length",
        ) as slice_signal,
    ):
        result = constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )

    pad_signal.assert_called_once_with(
        resampled,
        100,
    )
    slice_signal.assert_not_called()

    assert result.shape == (100,)


def test_constellation_modulator_rejects_incorrect_final_length():
    """A malfunctioning length helper should trigger output validation."""
    rng = np.random.default_rng(42)
    baseband = np.ones(40, dtype=np.complex64)
    resampled = np.ones(90, dtype=np.complex64)
    incorrectly_padded = np.ones(99, dtype=np.complex64)

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator_baseband",
            return_value=baseband,
        ),
        patch(
            f"{MODULE_PATH}.multistage_polyphase_resampler",
            return_value=resampled,
        ),
        patch(
            f"{MODULE_PATH}.pad_head_tail_to_length",
            return_value=incorrectly_padded,
        ),
        pytest.raises(
            ValueError,
            match=("constellation mod producing incorrect number of samples: 99 but requested: 100"),
        ),
    ):
        constellation_modulator(
            constellation_name="qpsk",
            pulse_shape_name="rectangular",
            bandwidth=1_000,
            sample_rate=10_000,
            num_samples=100,
            rng=rng,
        )


def test_constellation_signal_generator_initialization():
    """The generator should configure metadata fields and class name."""
    metadata = {"constellation_name": "qpsk"}

    with (
        patch(
            f"{MODULE_PATH}.BaseSignalGenerator.__init__",
            autospec=True,
        ) as base_init,
        patch.object(
            ConstellationSignalGenerator,
            "__getitem__",
            side_effect=metadata.__getitem__,
        ),
        patch.object(
            ConstellationSignalGenerator,
            "set_default_class_name",
        ) as set_class_name,
    ):
        generator = ConstellationSignalGenerator(**metadata)

    base_init.assert_called_once_with(
        generator,
        **metadata,
    )
    set_class_name.assert_called_once_with("qpsk")

    assert generator.required_metadata_fields == [
        "constellation_name",
        "sample_rate",
        "bandwidth_min",
        "bandwidth_max",
        "signal_duration_in_samples_min",
        "signal_duration_in_samples_max",
    ]


def test_constellation_signal_generator_generate_with_srrc():
    """A pulse-shape draw of zero should select SRRC and an alpha value."""
    metadata = {
        "constellation_name": "qpsk",
        "sample_rate": 10_000,
        "bandwidth_min": 500,
        "bandwidth_max": 1_000,
        "signal_duration_in_samples_min": 100,
        "signal_duration_in_samples_max": 200,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        150,
        800,
        0,
    ]
    rng.uniform.return_value = 0.25

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(150, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = ConstellationSignalGenerator.generate(GeneratorStub())

    assert rng.integers.call_args_list == [
        call(low=100, high=201),
        call(low=500, high=1_001),
        call(0, 2),
    ]
    rng.uniform.assert_called_once_with(0.1, 0.5)

    modulator.assert_called_once_with(
        "qpsk",
        "srrc",
        800,
        10_000,
        150,
        0.25,
        rng,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=800,
        pulse_shape_name="srrc",
        alpha_rolloff=0.25,
        pulse_shape_index=1,
        alpha_rolloff_target=0.25,
    )

    assert result is expected_signal


def test_constellation_signal_generator_generate_with_rectangular():
    """A pulse-shape draw of one should select rectangular shaping."""
    metadata = {
        "constellation_name": "16qam",
        "sample_rate": 20_000,
        "bandwidth_min": 1_000,
        "bandwidth_max": 2_000,
        "signal_duration_in_samples_min": 200,
        "signal_duration_in_samples_max": 400,
    }

    rng = MagicMock(spec=np.random.Generator)
    rng.integers.side_effect = [
        300,
        1_500,
        1,
    ]

    class GeneratorStub:
        random_generator = rng

        def __getitem__(self, key):
            return metadata[key]

    signal_data = np.ones(300, dtype=TorchSigComplexDataType)
    expected_signal = MagicMock()

    with (
        patch(
            f"{MODULE_PATH}.constellation_modulator",
            return_value=signal_data,
        ) as modulator,
        patch(
            f"{MODULE_PATH}.Signal",
            return_value=expected_signal,
        ) as signal_class,
    ):
        result = ConstellationSignalGenerator.generate(GeneratorStub())

    rng.uniform.assert_not_called()

    modulator.assert_called_once_with(
        "16qam",
        "rectangular",
        1_500,
        20_000,
        300,
        None,
        rng,
    )

    signal_class.assert_called_once_with(
        data=signal_data,
        center_freq=0,
        bandwidth=1_500,
        pulse_shape_name="rectangular",
        alpha_rolloff=None,
        pulse_shape_index=0,
        alpha_rolloff_target=0.0,
    )

    assert result is expected_signal


def _generate_2_2_0(generator):
    """Return ConstellationSignalGenerator.generate() as released in TorchSig 2.2.0.

    Kept verbatim (commit 1197261) as the golden reference for the default path:
    without the optional pulse-shape keys, generate() must make the same draws
    and return the same signal.
    """
    sample_rate = generator["sample_rate"]
    num_iq_samples_signal = generator.random_generator.integers(
        low=generator["signal_duration_in_samples_min"],
        high=generator["signal_duration_in_samples_max"] + 1,
    )
    bandwidth = generator.random_generator.integers(low=generator["bandwidth_min"], high=generator["bandwidth_max"] + 1)
    constellation_name = generator["constellation_name"]

    if generator.random_generator.integers(0, 2) == 0:
        pulse_shape_name = "srrc"
        alpha_rolloff = generator.random_generator.uniform(0.1, 0.5)
    else:
        pulse_shape_name = "rectangular"
        alpha_rolloff = None

    signal_data = constellation_modulator(
        constellation_name,
        pulse_shape_name,
        bandwidth,
        sample_rate,
        num_iq_samples_signal,
        alpha_rolloff,
        generator.random_generator,
    )

    return Signal(
        data=signal_data,
        center_freq=0,
        bandwidth=bandwidth,
        pulse_shape_name=pulse_shape_name,
        alpha_rolloff=alpha_rolloff,
        pulse_shape_index=int(pulse_shape_name == "srrc"),
        alpha_rolloff_target=(float(alpha_rolloff) if alpha_rolloff is not None else 0.0),
    )


def _draw_duration_and_bandwidth(rng, metadata):
    """Make the two draws that open every generate() call."""
    num_samples = rng.integers(
        low=metadata["signal_duration_in_samples_min"],
        high=metadata["signal_duration_in_samples_max"] + 1,
    )
    bandwidth = rng.integers(low=metadata["bandwidth_min"], high=metadata["bandwidth_max"] + 1)
    return num_samples, bandwidth


def _occupied_bandwidth(iq, sample_rate, nperseg):
    """Return the 99% occupied bandwidth (Hz), between the 0.5% and 99.5% points of the power."""
    freqs, psd = sp.welch(iq, fs=sample_rate, nperseg=nperseg, return_onesided=False, detrend=False)
    order = np.argsort(freqs)
    freqs, psd = freqs[order], psd[order]
    cumulative = np.cumsum(psd) / np.sum(psd)
    return freqs[np.searchsorted(cumulative, 0.995)] - freqs[np.searchsorted(cumulative, 0.005)]


@pytest.mark.parametrize(
    "constellation_name",
    ["ook", "bpsk", "qpsk", "16qam", "64ask", "32apsk", "128qam_cross"],
)
def test_constellation_signal_generator_default_matches_2_2_0(constellation_name):
    """Without the optional keys, output and metadata are bit-identical to 2.2.0."""
    metadata = {"constellation_name": constellation_name, **PINNING_METADATA}
    pulse_shapes = set()

    for seed in (0, 1, 7, 42, 2026):
        generator = ConstellationSignalGenerator(metadata=metadata, seed=seed)
        reference = ConstellationSignalGenerator(metadata=metadata, seed=seed)

        # Consecutive signals share one stream, so every draw must line up.
        for _ in range(4):
            signal = generator.generate()
            expected = _generate_2_2_0(reference)

            assert signal.data.dtype == expected.data.dtype
            assert signal.data.tobytes() == expected.data.tobytes()
            assert signal.to_dict() == expected.to_dict()
            pulse_shapes.add(signal["pulse_shape_name"])

        assert generator.random_generator.bit_generator.state == reference.random_generator.bit_generator.state

    assert pulse_shapes == {"srrc", "rectangular"}


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_constellation_signal_generator_pinned_srrc(seed):
    """A pinned SRRC pulse skips the pulse-shape draw and draws the rolloff from the pinned range."""
    metadata = {
        "constellation_name": "qpsk",
        **PINNING_METADATA,
        "pulse_shape_name": "srrc",
        "alpha_rolloff_min": 0.2,
        "alpha_rolloff_max": 0.3,
    }
    generator = ConstellationSignalGenerator(metadata=metadata, seed=seed)

    signal = generator()

    assert signal["pulse_shape_name"] == "srrc"
    assert signal["pulse_shape_index"] == 1
    assert 0.2 <= signal["alpha_rolloff"] <= 0.3
    assert signal["alpha_rolloff_target"] == signal["alpha_rolloff"]

    # Replay the expected draws: duration, bandwidth, rolloff and symbols, with no pulse-shape draw.
    rng = np.random.default_rng(seed)
    num_samples, bandwidth = _draw_duration_and_bandwidth(rng, metadata)
    alpha_rolloff = rng.uniform(0.2, 0.3)
    expected = constellation_modulator("qpsk", "srrc", bandwidth, metadata["sample_rate"], num_samples, alpha_rolloff, rng)

    assert signal["alpha_rolloff"] == alpha_rolloff
    np.testing.assert_array_equal(signal.data, expected)
    assert generator.random_generator.bit_generator.state == rng.bit_generator.state


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_constellation_signal_generator_pinned_rectangular(seed):
    """A pinned rectangular pulse has no rolloff, makes neither pulse-shape nor rolloff draw, and ignores a rolloff range."""
    metadata = {"constellation_name": "16qam", **PINNING_METADATA, "pulse_shape_name": "rectangular"}
    generator = ConstellationSignalGenerator(metadata=metadata, seed=seed)
    with_rolloff_range = ConstellationSignalGenerator(
        metadata={**metadata, "alpha_rolloff_min": 0.2, "alpha_rolloff_max": 0.3},
        seed=seed,
    )

    signal = generator()
    signal_with_rolloff_range = with_rolloff_range()

    assert signal["pulse_shape_name"] == "rectangular"
    assert signal["alpha_rolloff"] is None
    assert signal["pulse_shape_index"] == 0
    assert signal["alpha_rolloff_target"] == 0.0

    # Replay the expected draws: duration, bandwidth and symbols only.
    rng = np.random.default_rng(seed)
    num_samples, bandwidth = _draw_duration_and_bandwidth(rng, metadata)
    expected = constellation_modulator("16qam", "rectangular", bandwidth, metadata["sample_rate"], num_samples, None, rng)

    np.testing.assert_array_equal(signal.data, expected)
    assert generator.random_generator.bit_generator.state == rng.bit_generator.state

    # The rolloff range has no effect on a rectangular pulse.
    assert signal_with_rolloff_range.data.tobytes() == signal.data.tobytes()
    assert signal_with_rolloff_range.to_dict() == signal.to_dict()


def _replay_pinned_srrc(seed, metadata, alpha_rolloff, *, extra_rolloff_draw):
    """Replay a pinned-SRRC generate() on a fresh stream; optionally add the rolloff draw that pinning skips."""
    rng = np.random.default_rng(seed)
    num_samples, bandwidth = _draw_duration_and_bandwidth(rng, metadata)
    if extra_rolloff_draw:
        # uniform(a, a) returns a but still advances the stream.
        assert rng.uniform(alpha_rolloff, alpha_rolloff) == alpha_rolloff
    data = constellation_modulator(
        metadata["constellation_name"],
        "srrc",
        bandwidth,
        metadata["sample_rate"],
        num_samples,
        alpha_rolloff,
        rng,
    )
    return data, rng.bit_generator.state


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_constellation_signal_generator_equal_rolloff_bounds_pin_exact_value_without_a_draw(seed):
    """alpha_rolloff_min == alpha_rolloff_max gives exactly that rolloff and consumes no draw.

    The generator's stream must end where a replay of the expected draws
    (duration, bandwidth, symbols) ends. A replay that also makes the skipped
    rolloff draw must end elsewhere and produce other samples, which shows the
    comparison detects a single extra draw.
    """
    alpha_rolloff = 0.35
    metadata = {
        "constellation_name": "qpsk",
        **PINNING_METADATA,
        "pulse_shape_name": "srrc",
        "alpha_rolloff_min": alpha_rolloff,
        "alpha_rolloff_max": alpha_rolloff,
    }
    generator = ConstellationSignalGenerator(metadata=metadata, seed=seed)

    signal = generator.generate()

    assert signal["alpha_rolloff"] == alpha_rolloff
    assert signal["alpha_rolloff_target"] == alpha_rolloff

    expected, expected_state = _replay_pinned_srrc(seed, metadata, alpha_rolloff, extra_rolloff_draw=False)
    with_extra_draw, with_extra_draw_state = _replay_pinned_srrc(seed, metadata, alpha_rolloff, extra_rolloff_draw=True)

    assert generator.random_generator.bit_generator.state == expected_state
    np.testing.assert_array_equal(signal.data, expected)

    assert with_extra_draw_state != expected_state
    assert not np.array_equal(with_extra_draw, expected)


@pytest.mark.parametrize(
    ("alpha_rolloff_min", "alpha_rolloff_max"),
    [(0.2, 0.3), (0.35, 0.35)],
)
def test_constellation_signal_generator_rolloff_range_keeps_pulse_shape_draw(alpha_rolloff_min, alpha_rolloff_max):
    """Without pulse_shape_name the pulse shape is still drawn, and SRRC takes its rolloff from the pinned range."""
    metadata = {
        "constellation_name": "8psk",
        **PINNING_METADATA,
        "alpha_rolloff_min": alpha_rolloff_min,
        "alpha_rolloff_max": alpha_rolloff_max,
    }
    pulse_shapes = set()

    for seed in range(8):
        generator = ConstellationSignalGenerator(metadata=metadata, seed=seed)
        signal = generator.generate()

        # Replay: duration, bandwidth, pulse shape, then a rolloff draw only for a ranged SRRC.
        rng = np.random.default_rng(seed)
        num_samples, bandwidth = _draw_duration_and_bandwidth(rng, metadata)
        pulse_shape_name = "srrc" if rng.integers(0, 2) == 0 else "rectangular"
        alpha_rolloff = None
        if pulse_shape_name == "srrc":
            alpha_rolloff = alpha_rolloff_min if alpha_rolloff_min == alpha_rolloff_max else rng.uniform(alpha_rolloff_min, alpha_rolloff_max)
        expected = constellation_modulator("8psk", pulse_shape_name, bandwidth, metadata["sample_rate"], num_samples, alpha_rolloff, rng)

        assert signal["pulse_shape_name"] == pulse_shape_name
        assert signal["alpha_rolloff"] == alpha_rolloff
        if alpha_rolloff is not None:
            assert alpha_rolloff_min <= alpha_rolloff <= alpha_rolloff_max
        np.testing.assert_array_equal(signal.data, expected)
        assert generator.random_generator.bit_generator.state == rng.bit_generator.state
        pulse_shapes.add(pulse_shape_name)

    assert pulse_shapes == {"srrc", "rectangular"}


def test_constellation_signal_generator_pinning_keys_resolve_through_metadata_hierarchy():
    """Keys set on a parent apply to its generators; a child's None restores the default draws."""
    parent = HierarchicalMetadataObject(
        metadata={
            **PINNING_METADATA,
            "pulse_shape_name": "srrc",
            "alpha_rolloff_min": 0.25,
            "alpha_rolloff_max": 0.25,
        },
        seed=3,
    )
    pinned = ConstellationSignalGenerator(constellation_name="qpsk", parent=parent)
    unpinned = ConstellationSignalGenerator(
        constellation_name="qpsk",
        parent=parent,
        pulse_shape_name=None,
        alpha_rolloff_min=None,
        alpha_rolloff_max=None,
    )

    pinned_signals = [pinned() for _ in range(8)]
    unpinned_signals = [unpinned() for _ in range(16)]

    assert {signal["pulse_shape_name"] for signal in pinned_signals} == {"srrc"}
    assert {signal["alpha_rolloff"] for signal in pinned_signals} == {0.25}
    assert {signal["pulse_shape_name"] for signal in unpinned_signals} == {"srrc", "rectangular"}
    for signal in unpinned_signals:
        if signal["pulse_shape_name"] == "srrc":
            assert 0.1 <= signal["alpha_rolloff"] <= 0.5


def test_constellation_signal_generator_none_values_match_absent_keys():
    """Keys set to None leave every draw and the output unchanged."""
    metadata = {"constellation_name": "qpsk", **PINNING_METADATA}
    for seed in (0, 1, 2):
        absent = ConstellationSignalGenerator(metadata=metadata, seed=seed)
        none_valued = ConstellationSignalGenerator(
            metadata={**metadata, "pulse_shape_name": None, "alpha_rolloff_min": None, "alpha_rolloff_max": None},
            seed=seed,
        )
        for _ in range(4):
            signal = absent.generate()
            none_valued_signal = none_valued.generate()
            assert none_valued_signal.data.tobytes() == signal.data.tobytes()
            assert none_valued_signal.to_dict() == signal.to_dict()


@pytest.mark.parametrize("pulse_shape_name", ["SRRC", "raised_cosine", "", 0])
def test_constellation_signal_generator_rejects_unknown_pulse_shape_name(pulse_shape_name):
    """An unknown pulse shape raises ValueError naming the valid values, before any draw."""
    generator = ConstellationSignalGenerator(
        metadata={"constellation_name": "qpsk", **PINNING_METADATA, "pulse_shape_name": pulse_shape_name},
        seed=0,
    )
    state = generator.random_generator.bit_generator.state

    with pytest.raises(ValueError, match=r"expected one of \['srrc', 'rectangular'\]"):
        generator.generate()

    assert generator.random_generator.bit_generator.state == state


@pytest.mark.parametrize(
    ("alpha_rolloff_min", "alpha_rolloff_max"),
    [
        (0.0, 0.3),
        (-0.1, 0.3),
        (0.3, 0.2),
        (0.2, 1.0),
        (0.5, 1.5),
        (1.0, 1.0),
        (0.0, 0.0),
        (float("nan"), 0.3),
        (0.2, float("nan")),
    ],
)
@pytest.mark.parametrize("pulse_shape_name", ["srrc", "rectangular", None])
def test_constellation_signal_generator_rejects_invalid_rolloff_range(alpha_rolloff_min, alpha_rolloff_max, pulse_shape_name):
    """A rolloff range outside 0 < min <= max < 1 raises ValueError before any draw, whatever the pulse shape."""
    metadata = {
        "constellation_name": "qpsk",
        **PINNING_METADATA,
        "alpha_rolloff_min": alpha_rolloff_min,
        "alpha_rolloff_max": alpha_rolloff_max,
    }
    if pulse_shape_name is not None:
        metadata["pulse_shape_name"] = pulse_shape_name
    generator = ConstellationSignalGenerator(metadata=metadata, seed=0)
    state = generator.random_generator.bit_generator.state

    with pytest.raises(ValueError, match="0 < alpha_rolloff_min <= alpha_rolloff_max < 1"):
        generator.generate()

    assert generator.random_generator.bit_generator.state == state


@pytest.mark.parametrize(
    "rolloff_keys",
    [
        {"alpha_rolloff_min": 0.2},
        {"alpha_rolloff_max": 0.3},
        {"alpha_rolloff_min": 0.2, "alpha_rolloff_max": None},
    ],
)
def test_constellation_signal_generator_requires_both_rolloff_bounds(rolloff_keys):
    """Setting only one rolloff bound raises ValueError."""
    generator = ConstellationSignalGenerator(
        metadata={"constellation_name": "qpsk", **PINNING_METADATA, "pulse_shape_name": "srrc", **rolloff_keys},
        seed=0,
    )

    with pytest.raises(ValueError, match="alpha_rolloff_min and alpha_rolloff_max must be set together"):
        generator.generate()


@pytest.mark.parametrize("constellation_name", ["bpsk", "qpsk", "16qam", "ook"])
def test_constellation_signal_generator_pinned_pulse_shape_sets_occupied_bandwidth(constellation_name):
    """For the same requested bandwidth, the rectangular pulse occupies more spectrum than SRRC.

    Both bounds follow from the pulse spectra, not from observed values. The
    modulator sets the symbol rate equal to the requested bandwidth, so:

    * SRRC with rolloff alpha is band-limited to (1 + alpha) times the
      bandwidth; its 99% occupied bandwidth cannot exceed that, apart from one
      Welch bin of estimator resolution.
    * The rectangular pulse keeps only about 93% of its power (about 96% for
      OOK, whose carrier adds power at DC) inside its main lobe, which is twice
      the bandwidth wide, so its 99% occupied bandwidth must exceed twice the
      bandwidth.

    Measured here: about 1.2 (SRRC) and 3.3 (rectangular; 3.1 for OOK) times
    the bandwidth.
    """
    sample_rate = 10_000_000
    bandwidth = 1_000_000
    alpha_rolloff = 0.35
    nperseg = 1024
    metadata = {
        "constellation_name": constellation_name,
        "sample_rate": sample_rate,
        "bandwidth_min": bandwidth,
        "bandwidth_max": bandwidth,
        "signal_duration_in_samples_min": 16_384,
        "signal_duration_in_samples_max": 16_384,
    }
    srrc = ConstellationSignalGenerator(
        metadata={**metadata, "pulse_shape_name": "srrc", "alpha_rolloff_min": alpha_rolloff, "alpha_rolloff_max": alpha_rolloff},
        seed=11,
    )()
    rectangular = ConstellationSignalGenerator(metadata={**metadata, "pulse_shape_name": "rectangular"}, seed=11)()

    srrc_occupied = _occupied_bandwidth(srrc.data, sample_rate, nperseg)
    rectangular_occupied = _occupied_bandwidth(rectangular.data, sample_rate, nperseg)

    assert srrc["bandwidth"] == rectangular["bandwidth"] == bandwidth
    assert srrc_occupied <= (1 + alpha_rolloff) * bandwidth + sample_rate / nperseg
    assert rectangular_occupied > 2 * bandwidth
    assert rectangular_occupied > srrc_occupied
