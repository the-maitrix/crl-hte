import numpy as np

from crl_hte import dgp
from crl_hte.metrics import pehe_norm


def _cfg(**overrides):
    cfg = {
        "x_dim": 8,
        "z_dim": 2,
        "s_dim": 2,
        "z_s_fraction": 1.0,
        "sigma_S": 0.0,
        "sigma_Y": 0.0,
        "x_to_z_nonlinearity": "square",
        "z_to_s_nonlinearity": "cubic",
        "s_to_y_nonlinearity": "cubic",
    }
    cfg.update(overrides)
    return cfg


def test_nonlinear_latent_map_is_applied_before_projection():
    cfg = _cfg()
    params = dgp.init_dgp_params(cfg, dgp_seed=7)
    sample = dgp.generate_experimental(
        32, params, cfg, np.random.default_rng(11)
    )
    expected = np.square(sample["X"]) @ params.W_xz.T
    np.testing.assert_allclose(sample["Z"], expected, rtol=1e-5, atol=1e-5)


def test_oracle_constant_effect_has_unit_normalized_pehe():
    tau = np.array([-2.0, -1.0, 1.0, 4.0])
    constant = np.full_like(tau, tau.mean())
    assert np.isclose(pehe_norm(constant, tau), 1.0)


def test_alpha_components_match_final_cate_variance():
    cfg = _cfg(
        x_dim=40,
        z_dim=4,
        s_dim=3,
        z_s_fraction=0.5,
        x_to_z_nonlinearity="linear",
        z_to_s_nonlinearity="linear",
        s_to_y_nonlinearity="linear",
    )
    params = dgp.init_dgp_params(cfg, dgp_seed=7)
    cov_z = params.W_xz @ params.W_xz.T
    coef_s = params.gamma_z.T @ params.w_sy
    coef_y = params.gamma_y_norm.T @ params.w_sy
    var_s = float(coef_s @ cov_z @ coef_s)
    var_y = float(coef_y @ cov_z @ coef_y)
    assert np.isclose(var_s, var_y, rtol=1e-10, atol=1e-12)


def test_delta_components_match_final_cate_variance():
    cfg = _cfg(
        x_dim=40,
        z_dim=4,
        s_dim=3,
        z_s_fraction=0.5,
        x_to_z_nonlinearity="linear",
        z_to_s_nonlinearity="linear",
        s_to_y_nonlinearity="linear",
    )
    params = dgp.init_dgp_params(cfg, dgp_seed=7)
    cov_z = params.W_xz @ params.W_xz.T
    coef_s = params.gamma_z.T @ params.w_sy
    var_s = float(coef_s @ cov_z @ coef_s)
    var_bypass = float(params.w_bypass @ cov_z @ params.w_bypass)
    assert np.isclose(var_s, var_bypass, rtol=1e-10, atol=1e-12)


def test_projection_silu_is_nonlinear_after_projection():
    cfg = _cfg(x_to_z_nonlinearity="projection_silu")
    params = dgp.init_dgp_params(cfg, dgp_seed=7)
    sample = dgp.generate_experimental(32, params, cfg, np.random.default_rng(11))
    scale = np.linalg.norm(params.W_xz, axis=1).clip(1e-6)
    projected = sample["X"] @ params.W_xz.T / scale
    expected = projected / (1.0 + np.exp(-projected)) * scale
    np.testing.assert_allclose(sample["Z"], expected, rtol=1e-5, atol=1e-5)


def test_smooth_interaction_retains_linear_and_interaction_signal():
    cfg = _cfg(x_to_z_nonlinearity="smooth_interaction")
    params = dgp.init_dgp_params(cfg, dgp_seed=7)
    sample = dgp.generate_experimental(32, params, cfg, np.random.default_rng(11))
    scale = np.linalg.norm(params.W_xz, axis=1).clip(1e-6)
    first = sample["X"] @ params.W_xz.T / scale
    second = sample["X"] @ np.roll(params.W_xz, 1, axis=1).T / scale
    expected = (first + 0.5 * first * second) * scale / np.sqrt(1.25)
    np.testing.assert_allclose(sample["Z"], expected, rtol=1e-5, atol=1e-5)
