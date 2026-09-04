from crl_hte.config import load_config


SMALL_EXPERIMENTAL_GRID = [50, 100, 250, 500, 750]


def test_requested_representation_dimensions_are_exact():
    cfg = load_config(
        "configs/default.yaml",
        "configs/slices/m_sweep.yaml",
    )
    assert list(cfg.sweep.phi_dim) == [2, 3, 5, 10, 20, 50]


def test_paper_slices_share_the_small_experimental_grid():
    for name in (
        "paper_main", "raw_x_comparison", "alpha_sweep_dml_varmatched",
        "delta_sweep_dml_varmatched", "m_sweep",
    ):
        cfg = load_config(
            "configs/default.yaml",
            f"configs/slices/{name}.yaml",
        )
        assert list(cfg.sweep.sample_sizes) == SMALL_EXPERIMENTAL_GRID
        assert max(cfg.sweep.sample_sizes) == 750
        assert int(cfg.dgp.num_experimental) == 5000
        assert cfg.cate.regressor == "histgbr"


def test_paper_main_matches_saved_paper_provenance():
    cfg = load_config(
        "configs/default.yaml",
        "configs/slices/paper_main.yaml",
    )
    assert cfg.slice.historical_seed_mode == "trial"
    assert int(cfg.slice.encoder_historical_seed) == 1000
    assert cfg.cate.regressor == "histgbr"
    assert list(cfg.slice.dgp_seeds) == [42]
    assert int(cfg.slice.shared_encoder_seed) == 42
    assert int(cfg.slice.n_trials) == 10
    assert list(cfg.sweep.learners) == ["x_learner", "dml_learner"]
    assert list(cfg.sweep.y_types) == ["h_y", "true_y"]


def test_paper_violation_sweeps_match_reported_configuration():
    for name in ("alpha_sweep_dml_varmatched", "delta_sweep_dml_varmatched"):
        cfg = load_config("configs/default.yaml", f"configs/slices/{name}.yaml")
        assert int(cfg.slice.n_trials) == 10
        assert list(cfg.slice.dgp_seeds) == [42]
        assert float(cfg.encoder.lambda_loss_s) == 2.0
        assert list(cfg.sweep.learners) == ["dml_learner"]
        assert list(cfg.sweep.y_types) == ["h_y", "true_y"]
