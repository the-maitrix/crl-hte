import pytest

from crl_hte.runner import (
    _axis_invariant_fit_key,
    _historical_draw_seed,
    _method_y_types,
)


def test_true_y_slice_does_not_request_h_models():
    assert _method_y_types("encoder_pred", ["true_y"]) == ["true_y"]
    assert _method_y_types("surrogate_index", ["true_y"]) == ["h_y"]


def test_axis_invariant_key_tracks_method_and_outcome_source():
    common = dict(
        method="raw_x",
        dgp_seed=42,
        trial_seed=1000,
        z_dim=10,
        z_s_fraction=0.5,
        alpha=0.0,
        delta=0.0,
        num_observational=None,
        n=100,
        learner="dml_learner",
        y_type="true_y",
        regressor="gbr",
    )
    first = _axis_invariant_fit_key(**common)
    second = _axis_invariant_fit_key(**common)
    assert first == second
    assert first != _axis_invariant_fit_key(**{**common, "y_type": "h_y"})
    assert first != _axis_invariant_fit_key(
        **{**common, "method": "surrogate_index", "y_type": "h_y"}
    )
    assert first == _axis_invariant_fit_key(
        **{**common, "num_observational": None}
    )
    h_y = {**common, "y_type": "h_y", "num_observational": 10000}
    assert _axis_invariant_fit_key(**h_y) != _axis_invariant_fit_key(
        **{**h_y, "num_observational": 5000}
    )
    assert len(first) == 12
    with pytest.raises(ValueError):
        _axis_invariant_fit_key(**{**common, "method": "encoder_pred"})


def test_historical_seed_modes_separate_paper_run_from_new_sweeps():
    assert _historical_draw_seed(
        "trial", dgp_seed=42, trial_seed=1003
    ) == 1003
    assert _historical_draw_seed(
        "dgp", dgp_seed=42, trial_seed=1003
    ) == 42
    with pytest.raises(ValueError):
        _historical_draw_seed("unknown", dgp_seed=42, trial_seed=1003)
