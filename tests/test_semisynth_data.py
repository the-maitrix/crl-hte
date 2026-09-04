from types import SimpleNamespace

import numpy as np
import pandas as pd

from crl_hte.semisynth import (
    diagnosis_prevalence_audit,
    generate_experimental,
    split_mhp_cohorts,
)
from scripts.build_semisynth import _select_x_columns
from scripts.run_semisynthetic import METHODS, parse_args


def test_pending_patients_join_historical_cohort_when_enabled():
    status = pd.Series([np.nan, 0.0, 1.0, 0.0, np.nan])

    historical, experimental, pending, no_mhp = split_mhp_cohorts(
        status, include_pending=True
    )

    assert historical.tolist() == [True, True, False, True, True]
    assert experimental.tolist() == [False, False, True, False, False]
    assert pending.tolist() == [False, True, False, True, False]
    assert no_mhp.tolist() == [True, False, False, False, True]


def test_pending_diagnosis_audit_detects_suppressed_export():
    status = pd.Series([np.nan] * 4 + [0.0] * 4 + [1.0] * 4)
    stale = pd.DataFrame(
        {
            "Anxiety_Prior": [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            "Depression_FirstTri": [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            "Diabetes_After_FirstPNV": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )

    audit = diagnosis_prevalence_audit(
        stale,
        status,
        prefixes=("Anxiety_", "Depression_", "Diabetes_"),
        min_reference_rate=0.01,
    )

    assert audit["ratio"] == 0.0
    assert set(audit["columns"]) == set(stale.columns)


def test_pending_diagnosis_audit_accepts_comparable_export():
    status = pd.Series([np.nan] * 4 + [0.0] * 4 + [1.0] * 4)
    corrected = pd.DataFrame(
        {
            "Anxiety_Prior": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "Depression_FirstTri": [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            "Diabetes_After_FirstPNV": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        }
    )

    audit = diagnosis_prevalence_audit(
        corrected,
        status,
        prefixes=("Anxiety_", "Depression_", "Diabetes_"),
        min_reference_rate=0.01,
    )

    assert audit["ratio"] == 1.0


def test_auxiliary_outcome_source_columns_are_excluded_from_x():
    frame = pd.DataFrame({
        "x1": [0.0, 1.0, 0.0, 1.0],
        "s_source": [1.0, 0.0, 1.0, 0.0],
        "outcome": [0.0, 1.0, 0.0, 1.0],
        "status": [0.0, 0.0, 1.0, 1.0],
    })
    config = {
        "cohort": {
            "id_cols": [],
            "drop_outcome_cols": ["outcome"],
            "mhp_status_col": "status",
            "outcome_col": "outcome",
        },
        "features": {
            "drop_missing_indicators": True,
            "drop_patterns": [],
            "tau_drop": 0.5,
            "tau_inform": 0.05,
        },
    }
    selected = _select_x_columns(
        frame,
        np.array([True, True, False, False]),
        np.array([False, False, True, True]),
        list(frame),
        list(frame),
        config,
        exclude_columns=["s_source"],
    )
    assert selected == ["x1"]


def test_experimental_sampling_respects_disjoint_pool_indices():
    cache = SimpleNamespace(
        X_E=np.arange(20, dtype=np.float32).reshape(10, 2),
        S_E=np.zeros((10, 1), dtype=np.float32),
        sel_idx=np.array([0]),
        muX_w=np.array([0.0]),
        muX_b=0.0,
        lambda_s=1.0,
        betaS=np.array([1.0]),
    )
    params = SimpleNamespace(
        g_form="linear",
        gamma_0=0.0,
        rel_idx=np.array([0]),
        gamma_x=np.array([0.0]),
        u_S=np.array([1.0]),
        sigma_S=np.array([0.0]),
        y_mode="binary",
    )
    sample = generate_experimental(
        3,
        cache,
        params,
        {},
        np.random.default_rng(7),
        pool_indices=np.array([5, 6, 7, 8]),
    )
    assert set(sample["pool_idx"]).issubset({5, 6, 7, 8})


def test_paper_semisynthetic_defaults():
    args = parse_args(["--cache-dir", "cache", "--output", "results.csv"])
    assert args.seeds == 10
    assert args.dgp_seed == 0
    assert args.sample_sizes == [100, 250, 500, 750, 1000]
    assert args.historical_n == 10000
    assert args.test_n == 5000
    assert args.phi_dim == 10
    assert args.methods == METHODS
    assert args.cate_learner == "econml_dml"
    assert args.outcome_source == "predicted"
    assert args.ate_calibration == "none"
    assert not args.no_standardize_representations
