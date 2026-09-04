"""
All encoder / representation methods used by the post-rebuttal experiments.

Methods implemented (each returns a fitted model exposing `.encode(X) -> phi`):

    Neural (PyTorch):
        encoder_pred       — CRL with MSE(Y) + lambda * MSE(S)         [main method]
        encoder_no_s     — MSE(Y) only                                [ablation]
        encoder_no_y     — MSE(S) only                                [ablation]
        mi_mine          — DV (MINE) lower bound on I(phi;Y) and I(phi;S)
        mi_infonce       — Symmetric InfoNCE bounds on I(phi;Y), I(phi;S)
        mi_mine_cond     — MINE on JOINT (Y,S), making the gradient track
                           I(phi; Y|S) + lambda * I(phi; S)            [conditional]
        mi_infonce_cond  — InfoNCE on JOINT (Y,S), same conditional objective
        mi_vib           — VIB: MSE decoders + beta * KL(q(phi|X)||N(0,I))
        autoencoder      — Reconstruct X (no S, no Y signal)

    The `*_cond` variants use the chain-rule identity
        I(phi; Y, S) = I(phi; Y | S) + I(phi; S)
    to estimate `I(Y; phi | S) + lambda * I(S; phi)` as
        I(phi; Y, S) + (lambda - 1) * I(phi; S)
    using two MI estimators (joint Y,S anchor + S-only anchor). At lambda=1
    the S-extra term drops to zero. At lambda=2 (default) it equals one
    extra unit of I(phi; S), matching the original lambda_loss_s scale.

    Classical (sklearn):
        pca, pls, ica

    Trivial (no compression):
        raw_x            — return X itself (phi = X)
        baseline_xs      — return X (S concatenated downstream)
        surrogate_index  — Athey-style h(X,S) imputation, CATE on X
        constant_ate     — constant-effect reference used by PEHE tables

Architectures are configurable via `arch in {"small", "medium", "large"}`:
    small:  64 -> 32 -> phi_dim
    medium: 128 -> 64 -> 32 -> phi_dim          [default]
    large:  256 -> 128 -> 64 -> 32 -> phi_dim

Heads (S, Y, projection, MINE statistics nets) are sized at hidden=64 for all
architectures by default.

Device handling
---------------
All neural training accepts a `device` argument and moves the model to that
device. Inference runs on CPU after training (`.to_cpu()`) so downstream
sklearn estimators don't have to worry about devices.

Reproducibility
---------------
Each `train_*` accepts a `seed` and seeds torch deterministically. Different
seeds produce different encoders even with the same data — used to match
trial-level seed handling in the runner.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FastICA


# ── Architecture builders ────────────────────────────────────────────────────

_ARCH_HIDDEN: Dict[str, List[int]] = {
    "small":  [64, 32],
    "medium": [128, 64, 32],
    "large":  [256, 128, 64, 32],
}


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Resolve a torch device. `prefer` may be 'cuda', 'mps', 'cpu', or None."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" or (prefer is None and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    if prefer == "mps" or (prefer is None and torch.backends.mps.is_available()):
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


# ── Modules ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Deterministic encoder X -> phi."""

    def __init__(self, x_dim: int, phi_dim: int, arch: str = "medium"):
        super().__init__()
        self.x_dim = x_dim
        self.phi_dim = phi_dim
        self.arch = arch
        hidden = _ARCH_HIDDEN[arch]
        self.net = _mlp(x_dim, hidden, phi_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StochasticEncoder(nn.Module):
    """Reparameterised Gaussian encoder for VIB."""

    def __init__(self, x_dim: int, phi_dim: int, arch: str = "medium"):
        super().__init__()
        hidden = _ARCH_HIDDEN[arch]
        # Backbone produces the last-hidden features
        backbone_layers: List[nn.Module] = []
        prev = x_dim
        for h in hidden:
            backbone_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.backbone = nn.Sequential(*backbone_layers)
        self.mu_head = nn.Linear(prev, phi_dim)
        self.lv_head = nn.Linear(prev, phi_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu_head(h)
        lv = self.lv_head(h).clamp(-8, 8)
        if self.training:
            phi = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        else:
            phi = mu
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(-1).mean()
        return phi, kl

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.mu_head(h)


class HeadS(nn.Module):
    def __init__(self, phi_dim: int, s_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(phi_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, s_dim))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi)


class HeadY(nn.Module):
    """Y head with input [phi, s_hat]. Matches the original CRL design."""

    def __init__(self, phi_dim: int, s_dim: int, hidden: int = 64, y_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(phi_dim + s_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, y_dim))

    def forward(self, s: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi, s], dim=1))


class HeadST(nn.Module):
    """S head with [phi, T] input — used during finetuning to expose τ_S residual."""

    def __init__(self, phi_dim: int, s_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(phi_dim + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, s_dim))

    def forward(self, phi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi, t], dim=1))


class HeadYT(nn.Module):
    """Y head with [phi, T] input — used during finetuning to expose bypass(Z_y)."""

    def __init__(self, phi_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(phi_dim + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, phi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([phi, t], dim=1))


class ProjHead(nn.Module):
    """L2-normalised projection head for InfoNCE."""

    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                                 nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class MINENet(nn.Module):
    """Statistics network T(a,b) -> ℝ for the DV bound."""

    def __init__(self, dim_a: int, dim_b: int, hidden: int = 64):
        super().__init__()
        self.T = nn.Sequential(nn.Linear(dim_a + dim_b, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.T(torch.cat([a, b], dim=-1))


class _AutoencoderModule(nn.Module):
    def __init__(self, x_dim: int, phi_dim: int, arch: str = "medium"):
        super().__init__()
        hidden = _ARCH_HIDDEN[arch]
        self.encoder_net = _mlp(x_dim, hidden, phi_dim)
        # Decoder mirrors the encoder
        self.decoder_net = _mlp(phi_dim, list(reversed(hidden)), x_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_net(self.encoder_net(x))


# ── Loss helpers ─────────────────────────────────────────────────────────────

def _infonce_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    B = z_a.size(0)
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(B, device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _mine_loss(T_net: nn.Module, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    B = a.size(0)
    perm = torch.randperm(B, device=a.device)
    t_joint = T_net(a, b)
    t_marginal = T_net(a, b[perm])
    mi_lb = t_joint.mean() - (t_marginal.logsumexp(0) - math.log(B))
    return -mi_lb  # minimise to maximise MI


def _early_stop(losses: List[float], patience: int, min_delta: float) -> bool:
    if len(losses) < patience + 1:
        return False
    return losses[-1] >= min(losses[-(patience + 1):-1]) - min_delta


# ── Train/val split + epoch helpers ─────────────────────────────────────────
# When cfg["val_frac"] > 0, the obs dataset is split deterministically (using
# encoder seed + a fixed offset) into train/val. Early-stopping then tracks
# *val* loss instead of train loss, which prevents the optimizer from
# terminating at a bad train-loss plateau on hard nonlinear DGPs.

def _split_obs(obs: Dict[str, np.ndarray], val_frac: float, split_seed: int):
    if val_frac <= 0.0:
        return obs, None
    n = len(obs["X"])
    n_val = int(round(n * val_frac))
    if n_val < 1 or n_val >= n:
        return obs, None
    perm = np.random.default_rng(split_seed).permutation(n)
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    obs_tr = {k: v[train_idx] for k, v in obs.items()}
    obs_va = {k: v[val_idx]   for k, v in obs.items()}
    return obs_tr, obs_va


def _get_train_val_loaders(obs: Dict, cfg: Dict, device: torch.device,
                           split_seed: int, use_y: bool = True, use_s: bool = True):
    """Returns (train_loader, val_loader, n_train, n_val). val_loader is None
    when cfg["val_frac"] == 0 (preserves the original behaviour)."""
    val_frac = float(cfg.get("val_frac", 0.0))
    obs_tr, obs_va = _split_obs(obs, val_frac, split_seed)
    train_loader, n_train = _build_loader(obs_tr, cfg["batch_size"], device,
                                          use_y=use_y, use_s=use_s)
    if obs_va is None:
        return train_loader, None, n_train, 0
    val_loader, n_val = _build_loader(obs_va, cfg["batch_size"], device,
                                      use_y=use_y, use_s=use_s, shuffle=False)
    return train_loader, val_loader, n_train, n_val


def _run_epoch(loader, batch_loss_fn, modules: List[nn.Module],
               optimizer: Optional[torch.optim.Optimizer], n_total: int) -> float:
    grad_enabled = optimizer is not None
    for m in modules:
        m.train() if grad_enabled else m.eval()
    total = 0.0
    for batch in loader:
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(grad_enabled):
            loss = batch_loss_fn(batch)
        if optimizer is not None:
            loss.backward(); optimizer.step()
        total += float(loss.item()) * batch[0].size(0)
    return total / max(1, n_total)


def _train_with_early_stop(train_loader, val_loader, batch_loss_fn,
                           modules: List[nn.Module],
                           optimizer: torch.optim.Optimizer, cfg: Dict,
                           n_train: int, n_val: int):
    """Run the standard train loop with early-stop on val loss when val_loader
    is provided, else on train loss.

    When cfg["select_best_epoch"] is true, snapshot every module's state_dict
    each time the stop-metric hits a new low and restore it before returning —
    so the caller gets the *best* model, not the last (potentially overfit) one.
    The legacy behavior (return the model at the loop-exit epoch, which can be
    `patience` epochs past the actual minimum) is preserved when the flag is
    false.

    Returns (epochs_run, train_losses, val_losses, best_epoch).
    """
    train_losses: List[float] = []; val_losses: List[float] = []
    epochs_run = cfg["epochs_obs"]
    patience = cfg["early_stopping_patience"]
    min_delta = cfg["early_stopping_min_delta"]
    select_best = bool(cfg.get("select_best_epoch", False))

    best_metric = float("inf")
    best_state: Optional[List[Dict[str, torch.Tensor]]] = None
    best_epoch = 0

    for epoch in range(cfg["epochs_obs"]):
        train_losses.append(_run_epoch(train_loader, batch_loss_fn,
                                       modules, optimizer, n_train))
        if val_loader is not None:
            val_losses.append(_run_epoch(val_loader, batch_loss_fn,
                                         modules, None, n_val))
            stop_metric = val_losses
        else:
            stop_metric = train_losses

        if select_best and stop_metric[-1] < best_metric - min_delta:
            best_metric = stop_metric[-1]
            best_state = [{k: v.detach().clone() for k, v in m.state_dict().items()}
                          for m in modules]
            best_epoch = epoch + 1

        if _early_stop(stop_metric, patience, min_delta):
            epochs_run = epoch + 1
            break

    if select_best and best_state is not None:
        for m, sd in zip(modules, best_state):
            m.load_state_dict(sd)

    return epochs_run, train_losses, val_losses, best_epoch


# ── Wrappers for uniform .encode(X) interface ───────────────────────────────

@dataclass
class FittedEncoder:
    """Uniform interface for all method outputs.

    Attributes
    ----------
    name        : str
    encode      : callable (np.ndarray [n, x_dim]) -> np.ndarray [n, phi_dim]
    is_neural   : bool
    epochs_run  : int (0 for non-neural)
    state       : dict — anything else (sklearn objects etc.) for caching
    """
    name: str
    encode: callable
    is_neural: bool = False
    epochs_run: int = 0
    state: Dict = None
    torch_module: Optional[nn.Module] = None  # set for neural methods


def _neural_encode_fn(model: nn.Module, kind: str = "deterministic"):
    """Build an `encode(X_np) -> np_phi` closure that runs on CPU."""
    model.eval().to("cpu")

    def fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            if kind == "stochastic":
                out = model.encode(X_t)
            else:
                out = model(X_t)
            return out.numpy()
    return fn


# ── Trainers ─────────────────────────────────────────────────────────────────

def _build_loader(obs: Dict[str, np.ndarray], batch_size: int, device: torch.device,
                  use_y: bool = True, use_s: bool = True,
                  shuffle: bool = True) -> Tuple[DataLoader, int]:
    X_t = torch.tensor(obs["X"], dtype=torch.float32).to(device)
    tensors = [X_t]
    if use_s:
        tensors.append(torch.tensor(obs["S"], dtype=torch.float32).to(device))
    if use_y:
        tensors.append(torch.tensor(obs["Y"], dtype=torch.float32).to(device))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                      shuffle=shuffle), len(X_t)


def train_crl_mse(obs: Dict, cfg: Dict, device: torch.device,
                  seed: int, lambda_s: Optional[float] = None,
                  use_y: bool = True, use_s: bool = True,
                  arch: Optional[str] = None) -> FittedEncoder:
    """Train CRL with MSE(Y) + lambda * MSE(S). Use lambda_s=0 for no_s, use_y=False for no_y."""
    torch.manual_seed(seed)
    arch = arch or cfg.get("arch", "medium")
    lam = cfg.get("lambda_loss_s", 2.0) if lambda_s is None else lambda_s

    enc = Encoder(cfg["x_dim"], cfg["phi_dim"], arch=arch).to(device)
    head_s = HeadS(cfg["phi_dim"], cfg["s_dim"]).to(device)
    head_y = HeadY(cfg["phi_dim"], cfg["s_dim"]).to(device) if use_y else None

    params = list(enc.parameters()) + list(head_s.parameters())
    if head_y is not None:
        params += list(head_y.parameters())
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    train_loader, val_loader, n_train, n_val = _get_train_val_loaders(
        obs, cfg, device, split_seed=seed + 99991, use_y=use_y, use_s=True)
    mse = nn.MSELoss()

    def batch_loss(batch):
        xb, sb, yb = batch[0], batch[1], (batch[2] if use_y else None)
        phi = enc(xb)
        s_hat = head_s(phi)
        loss = torch.zeros((), device=device)
        if use_s and lam > 0:
            loss = loss + lam * mse(s_hat, sb)
        if use_y and head_y is not None:
            y_hat = head_y(s_hat, phi)
            loss = loss + mse(y_hat, yb)
        return loss

    modules: List[nn.Module] = [enc, head_s] + ([head_y] if head_y is not None else [])
    epochs_run, train_losses, val_losses, best_epoch = _train_with_early_stop(
        train_loader, val_loader, batch_loss, modules, opt, cfg, n_train, n_val)

    name = "encoder_pred"
    if not use_s or lam == 0:
        name = "encoder_no_s"
    elif not use_y:
        name = "encoder_no_y"

    return FittedEncoder(
        name=name,
        encode=_neural_encode_fn(enc, "deterministic"),
        is_neural=True,
        epochs_run=epochs_run,
        state={"final_train_loss": train_losses[-1],
               "final_val_loss": (val_losses[-1] if val_losses else None),
               "best_epoch": best_epoch},
        torch_module=enc.cpu(),
    )


def train_mi_infonce(obs: Dict, cfg: Dict, device: torch.device, seed: int,
                     arch: Optional[str] = None) -> FittedEncoder:
    torch.manual_seed(seed)
    arch = arch or cfg.get("arch", "medium")
    phi_dim = cfg["phi_dim"]; s_dim = cfg["s_dim"]; y_dim = cfg.get("y_dim", 1)
    proj_d = cfg.get("proj_dim", 64)
    temp = cfg.get("infonce_temperature", 0.07)
    lam = cfg.get("lambda_loss_s", 2.0)

    enc = Encoder(cfg["x_dim"], phi_dim, arch=arch).to(device)
    p_phi_s = ProjHead(phi_dim, proj_d).to(device)
    p_phi_y = ProjHead(phi_dim, proj_d).to(device)
    p_s = ProjHead(s_dim, proj_d).to(device)
    p_y = ProjHead(y_dim, proj_d).to(device)
    all_params = (list(enc.parameters()) + list(p_phi_s.parameters())
                  + list(p_phi_y.parameters()) + list(p_s.parameters())
                  + list(p_y.parameters()))
    opt = torch.optim.Adam(all_params, lr=cfg["lr"])

    train_loader, val_loader, n_train, n_val = _get_train_val_loaders(
        obs, cfg, device, split_seed=seed + 99991)

    def batch_loss(batch):
        xb, sb, yb = batch
        phi = enc(xb)
        return (_infonce_loss(p_phi_y(phi), p_y(yb), temp)
                + lam * _infonce_loss(p_phi_s(phi), p_s(sb), temp))

    modules: List[nn.Module] = [enc, p_phi_s, p_phi_y, p_s, p_y]
    epochs_run, train_losses, val_losses, best_epoch = _train_with_early_stop(
        train_loader, val_loader, batch_loss, modules, opt, cfg, n_train, n_val)

    return FittedEncoder(
        name="mi_infonce",
        encode=_neural_encode_fn(enc),
        is_neural=True, epochs_run=epochs_run,
        state={"final_train_loss": train_losses[-1],
               "final_val_loss": (val_losses[-1] if val_losses else None),
               "best_epoch": best_epoch},
        torch_module=enc.cpu(),
    )


def train_mi_mine(obs: Dict, cfg: Dict, device: torch.device, seed: int,
                  arch: Optional[str] = None) -> FittedEncoder:
    torch.manual_seed(seed)
    arch = arch or cfg.get("arch", "medium")
    phi_dim = cfg["phi_dim"]; s_dim = cfg["s_dim"]; y_dim = cfg.get("y_dim", 1)
    lam = cfg.get("lambda_loss_s", 2.0)

    enc = Encoder(cfg["x_dim"], phi_dim, arch=arch).to(device)
    mine_s = MINENet(phi_dim, s_dim).to(device)
    mine_y = MINENet(phi_dim, y_dim).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(mine_s.parameters()) + list(mine_y.parameters()),
        lr=cfg["lr"])

    train_loader, val_loader, n_train, n_val = _get_train_val_loaders(
        obs, cfg, device, split_seed=seed + 99991)

    def batch_loss(batch):
        xb, sb, yb = batch
        phi = enc(xb)
        return _mine_loss(mine_y, phi, yb) + lam * _mine_loss(mine_s, phi, sb)

    modules: List[nn.Module] = [enc, mine_s, mine_y]
    epochs_run, train_losses, val_losses, best_epoch = _train_with_early_stop(
        train_loader, val_loader, batch_loss, modules, opt, cfg, n_train, n_val)

    return FittedEncoder(
        name="mi_mine",
        encode=_neural_encode_fn(enc),
        is_neural=True, epochs_run=epochs_run,
        state={"final_train_loss": train_losses[-1],
               "final_val_loss": (val_losses[-1] if val_losses else None),
               "best_epoch": best_epoch},
        torch_module=enc.cpu(),
    )


def train_mi_infonce_cond(obs: Dict, cfg: Dict, device: torch.device, seed: int,
                          arch: Optional[str] = None) -> FittedEncoder:
    """InfoNCE on the JOINT (Y, S) target — gradient tracks I(phi; Y|S) + λ·I(phi; S).

    Decomposes the conditional objective via I(phi; Y, S) = I(phi; Y|S) + I(phi; S),
    giving total weighted MI of I(phi; Y, S) + (λ-1)·I(phi; S).
    """
    torch.manual_seed(seed)
    arch = arch or cfg.get("arch", "medium")
    phi_dim = cfg["phi_dim"]; s_dim = cfg["s_dim"]; y_dim = cfg.get("y_dim", 1)
    proj_d = cfg.get("proj_dim", 64)
    temp = cfg.get("infonce_temperature", 0.07)
    lam = cfg.get("lambda_loss_s", 2.0)
    lam_extra_s = max(0.0, lam - 1.0)  # negative would penalise I(phi;S); clamp at 0.

    enc = Encoder(cfg["x_dim"], phi_dim, arch=arch).to(device)
    p_phi_ys = ProjHead(phi_dim, proj_d).to(device)         # joint (Y,S) anchor for phi
    p_ys     = ProjHead(y_dim + s_dim, proj_d).to(device)   # joint (Y,S) anchor for label
    p_phi_s  = ProjHead(phi_dim, proj_d).to(device)
    p_s      = ProjHead(s_dim, proj_d).to(device)
    all_params = (list(enc.parameters()) + list(p_phi_ys.parameters())
                  + list(p_ys.parameters()) + list(p_phi_s.parameters())
                  + list(p_s.parameters()))
    opt = torch.optim.Adam(all_params, lr=cfg["lr"])

    train_loader, val_loader, n_train, n_val = _get_train_val_loaders(
        obs, cfg, device, split_seed=seed + 99991)

    def batch_loss(batch):
        xb, sb, yb = batch
        phi = enc(xb)
        ys = torch.cat([yb, sb], dim=-1)
        loss = _infonce_loss(p_phi_ys(phi), p_ys(ys), temp)
        if lam_extra_s > 0:
            loss = loss + lam_extra_s * _infonce_loss(p_phi_s(phi), p_s(sb), temp)
        return loss

    modules: List[nn.Module] = [enc, p_phi_ys, p_ys, p_phi_s, p_s]
    epochs_run, train_losses, val_losses, best_epoch = _train_with_early_stop(
        train_loader, val_loader, batch_loss, modules, opt, cfg, n_train, n_val)

    return FittedEncoder(
        name="mi_infonce_cond",
        encode=_neural_encode_fn(enc),
        is_neural=True, epochs_run=epochs_run,
        state={"final_train_loss": train_losses[-1],
               "final_val_loss": (val_losses[-1] if val_losses else None),
               "best_epoch": best_epoch},
        torch_module=enc.cpu(),
    )


def train_mi_mine_cond(obs: Dict, cfg: Dict, device: torch.device, seed: int,
                       arch: Optional[str] = None) -> FittedEncoder:
    """MINE on the JOINT (Y, S) target — same conditional objective as mi_infonce_cond."""
    torch.manual_seed(seed)
    arch = arch or cfg.get("arch", "medium")
    phi_dim = cfg["phi_dim"]; s_dim = cfg["s_dim"]; y_dim = cfg.get("y_dim", 1)
    lam = cfg.get("lambda_loss_s", 2.0)
    lam_extra_s = max(0.0, lam - 1.0)

    enc = Encoder(cfg["x_dim"], phi_dim, arch=arch).to(device)
    mine_ys = MINENet(phi_dim, y_dim + s_dim).to(device)    # joint (Y,S)
    mine_s  = MINENet(phi_dim, s_dim).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(mine_ys.parameters()) + list(mine_s.parameters()),
        lr=cfg["lr"])

    train_loader, val_loader, n_train, n_val = _get_train_val_loaders(
        obs, cfg, device, split_seed=seed + 99991)

    def batch_loss(batch):
        xb, sb, yb = batch
        phi = enc(xb)
        ys = torch.cat([yb, sb], dim=-1)
        loss = _mine_loss(mine_ys, phi, ys)
        if lam_extra_s > 0:
            loss = loss + lam_extra_s * _mine_loss(mine_s, phi, sb)
        return loss

    modules: List[nn.Module] = [enc, mine_ys, mine_s]
    epochs_run, train_losses, val_losses, best_epoch = _train_with_early_stop(
        train_loader, val_loader, batch_loss, modules, opt, cfg, n_train, n_val)

    return FittedEncoder(
        name="mi_mine_cond",
        encode=_neural_encode_fn(enc),
        is_neural=True, epochs_run=epochs_run,
        state={"final_train_loss": train_losses[-1],
               "final_val_loss": (val_losses[-1] if val_losses else None),
               "best_epoch": best_epoch},
        torch_module=enc.cpu(),
    )


def train_mi_vib(obs: Dict, cfg: Dict, device: torch.device, seed: int,
                 arch: Optional[str] = None) -> FittedEncoder:
    torch.manual_seed(seed)
    arch = arch or cfg.get("arch", "medium")
    lam = cfg.get("lambda_loss_s", 2.0)
    beta = cfg.get("vib_beta", 1e-3)

    enc = StochasticEncoder(cfg["x_dim"], cfg["phi_dim"], arch=arch).to(device)
    head_s = HeadS(cfg["phi_dim"], cfg["s_dim"]).to(device)
    head_y = HeadY(cfg["phi_dim"], cfg["s_dim"]).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(head_s.parameters()) + list(head_y.parameters()),
        lr=cfg["lr"])

    train_loader, val_loader, n_train, n_val = _get_train_val_loaders(
        obs, cfg, device, split_seed=seed + 99991)
    mse = nn.MSELoss()

    def batch_loss(batch):
        xb, sb, yb = batch
        phi, kl = enc(xb)
        s_hat = head_s(phi)
        y_hat = head_y(s_hat, phi)
        return mse(y_hat, yb) + lam * mse(s_hat, sb) + beta * kl

    modules: List[nn.Module] = [enc, head_s, head_y]
    epochs_run, train_losses, val_losses, best_epoch = _train_with_early_stop(
        train_loader, val_loader, batch_loss, modules, opt, cfg, n_train, n_val)

    return FittedEncoder(
        name="mi_vib",
        encode=_neural_encode_fn(enc, "stochastic"),
        is_neural=True, epochs_run=epochs_run,
        state={"final_train_loss": train_losses[-1],
               "final_val_loss": (val_losses[-1] if val_losses else None),
               "best_epoch": best_epoch},
        torch_module=enc.cpu(),
    )


def train_autoencoder(obs: Dict, cfg: Dict, device: torch.device, seed: int,
                      arch: Optional[str] = None) -> FittedEncoder:
    torch.manual_seed(seed)
    arch = arch or cfg.get("arch", "medium")
    ae = _AutoencoderModule(cfg["x_dim"], cfg["phi_dim"], arch=arch).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg["lr"])
    train_loader, val_loader, n_train, n_val = _get_train_val_loaders(
        obs, cfg, device, split_seed=seed + 99991, use_y=False, use_s=False)
    mse = nn.MSELoss()

    def batch_loss(batch):
        (xb,) = batch
        return mse(ae(xb), xb)

    modules: List[nn.Module] = [ae]
    epochs_run, train_losses, val_losses, best_epoch = _train_with_early_stop(
        train_loader, val_loader, batch_loss, modules, opt, cfg, n_train, n_val)

    # AE encode = ae.encoder_net forward
    ae.eval().to("cpu")
    enc_net = ae.encoder_net

    def encode(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return enc_net(X_t).numpy()

    return FittedEncoder(
        name="autoencoder", encode=encode, is_neural=True,
        epochs_run=epochs_run,
        state={"final_train_loss": train_losses[-1],
               "final_val_loss": (val_losses[-1] if val_losses else None),
               "best_epoch": best_epoch},
        torch_module=ae,
    )


# ── Sklearn-based methods ────────────────────────────────────────────────────

def fit_pca(obs: Dict, cfg: Dict, seed: int) -> FittedEncoder:
    pca = PCA(n_components=cfg["phi_dim"], random_state=seed).fit(obs["X"])
    return FittedEncoder(
        name="pca",
        encode=lambda X: pca.transform(X).astype(np.float32),
        state={"sklearn": pca},
    )


def fit_pls(obs: Dict, cfg: Dict, seed: int) -> FittedEncoder:
    pls = PLSRegression(n_components=cfg["phi_dim"]).fit(obs["X"], obs["S"])
    return FittedEncoder(
        name="pls",
        encode=lambda X: pls.transform(X).astype(np.float32),
        state={"sklearn": pls},
    )


def fit_ica(obs: Dict, cfg: Dict, seed: int) -> FittedEncoder:
    ica = FastICA(n_components=cfg["phi_dim"], random_state=seed,
                  max_iter=500, tol=0.01).fit(obs["X"])
    return FittedEncoder(
        name="ica",
        encode=lambda X: ica.transform(X).astype(np.float32),
        state={"sklearn": ica},
    )


def fit_raw_x(obs: Dict, cfg: Dict, seed: int) -> FittedEncoder:
    """Identity encoder: phi(X) = X. Used to evaluate downstream learners on raw covariates."""
    return FittedEncoder(name="raw_x", encode=lambda X: X.astype(np.float32))


def fit_baseline_xs(obs: Dict, cfg: Dict, seed: int) -> FittedEncoder:
    """Same as raw_x; the runner appends S downstream."""
    return FittedEncoder(name="baseline_xs", encode=lambda X: X.astype(np.float32))


def fit_identity(name: str) -> FittedEncoder:
    """Identity representation for baselines whose behavior lives downstream."""
    return FittedEncoder(name=name, encode=lambda X: X.astype(np.float32))


# ── Method registry ──────────────────────────────────────────────────────────

# Maps method name -> trainer signature (obs, cfg, device, seed, arch?) -> FittedEncoder
METHOD_REGISTRY = {
    "encoder_pred":  lambda obs, cfg, device, seed, arch=None:
        train_crl_mse(obs, cfg, device, seed, arch=arch),
    "encoder_no_s": lambda obs, cfg, device, seed, arch=None:
        train_crl_mse(obs, cfg, device, seed, lambda_s=0.0, use_y=True, arch=arch),
    "encoder_no_y": lambda obs, cfg, device, seed, arch=None:
        train_crl_mse(obs, cfg, device, seed, use_y=False, arch=arch),
    "mi_mine":         train_mi_mine,
    "mi_infonce":      train_mi_infonce,
    "mi_mine_cond":    train_mi_mine_cond,
    "mi_infonce_cond": train_mi_infonce_cond,
    "mi_vib":          train_mi_vib,
    "autoencoder":     train_autoencoder,
    # Non-neural
    "pca":         lambda obs, cfg, device, seed, arch=None: fit_pca(obs, cfg, seed),
    "pls":         lambda obs, cfg, device, seed, arch=None: fit_pls(obs, cfg, seed),
    "ica":         lambda obs, cfg, device, seed, arch=None: fit_ica(obs, cfg, seed),
    "raw_x":       lambda obs, cfg, device, seed, arch=None: fit_raw_x(obs, cfg, seed),
    "baseline_xs": lambda obs, cfg, device, seed, arch=None: fit_baseline_xs(obs, cfg, seed),
    "surrogate_index": lambda obs, cfg, device, seed, arch=None:
        fit_identity("surrogate_index"),
    "constant_ate": lambda obs, cfg, device, seed, arch=None:
        fit_identity("constant_ate"),
}

NEURAL_METHODS = {"encoder_pred", "encoder_no_s", "encoder_no_y",
                  "mi_mine", "mi_infonce",
                  "mi_mine_cond", "mi_infonce_cond",
                  "mi_vib", "autoencoder"}


def fit_method(name: str, obs: Dict, cfg: Dict, device: torch.device,
               seed: int, arch: Optional[str] = None) -> FittedEncoder:
    if name not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method: {name}. Available: {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name](obs, cfg, device, seed, arch)


# ── Finetuning ───────────────────────────────────────────────────────────────

def _freeze_backbone_keep_last_linear(enc: nn.Module,
                                      n_layers: int = 1) -> List[nn.Parameter]:
    """Freeze all encoder params except the LAST `n_layers` nn.Linear layers.

    n_layers=1 → readout only (most conservative; preserves backbone exactly).
    n_layers=2 → last hidden + readout (standard transfer-learning sweet spot).
    n_layers=k → last k Linears unfrozen.

    Returns the list of trainable parameters.
    """
    for p in enc.parameters():
        p.requires_grad = False
    container = getattr(enc, "net", None)
    if container is None:
        # StochasticEncoder shape — finetune the mu_head only (n_layers ignored).
        last_linear = getattr(enc, "mu_head", None)
        if last_linear is not None:
            for p in last_linear.parameters():
                p.requires_grad = True
    else:
        # Pick the last `n_layers` Linear layers in enc.net
        linears = [m for m in container if isinstance(m, nn.Linear)]
        keep = linears[-max(1, n_layers):] if linears else []
        for layer in keep:
            for p in layer.parameters():
                p.requires_grad = True
    return [p for p in enc.parameters() if p.requires_grad]


def finetune_ftS(enc_pretrained: nn.Module, exp: Dict, cfg: Dict,
                 device: torch.device, seed: int,
                 use_y_loss: bool = False,
                 freeze_backbone: bool = False,
                 unfreeze_layers: int = 1) -> nn.Module:
    """Finetune encoder on (X_exp, T, S_exp). Optionally also Y_exp.

    When `freeze_backbone=True`, only the encoder's last Linear layer + the
    head(s) get gradient updates — the backbone (pretrained representation)
    is preserved. Standard transfer-learning recipe to avoid overwriting
    Z_s when finetuning on small T=1 samples.

    Returns a fresh encoder module (deepcopied) that has been finetuned.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    enc = deepcopy(enc_pretrained).to(device)
    hs_t = HeadST(cfg["phi_dim"], cfg["s_dim"]).to(device)
    if freeze_backbone:
        enc_params = _freeze_backbone_keep_last_linear(enc, n_layers=unfreeze_layers)
    else:
        enc_params = list(enc.parameters())
    params = enc_params + list(hs_t.parameters())
    hy_ft = None
    if use_y_loss:
        hy_ft = HeadY(cfg["phi_dim"], cfg["s_dim"]).to(device)
        params += list(hy_ft.parameters())
    opt = torch.optim.Adam(params, lr=cfg.get("finetune_lr", 1e-3))

    bs = min(cfg.get("finetune_batch_size", 64), len(exp["X"]))
    Xt = torch.tensor(exp["X"], dtype=torch.float32).to(device)
    St = torch.tensor(exp["S"], dtype=torch.float32).to(device)
    Tt = torch.tensor(exp["T"], dtype=torch.float32).to(device)
    if use_y_loss:
        Yt = torch.tensor(exp["Y"], dtype=torch.float32).to(device)
        loader = DataLoader(TensorDataset(Xt, St, Tt, Yt), batch_size=bs, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(Xt, St, Tt), batch_size=bs, shuffle=True)
    mse = nn.MSELoss()
    lam_y = cfg.get("lambda_finetune_y", 1.0)
    losses: List[float] = []
    n_train = len(exp["X"])
    for _ in range(cfg.get("finetune_epochs", 100)):
        enc.train(); hs_t.train()
        if hy_ft is not None: hy_ft.train()
        ep_loss = 0.0
        for batch in loader:
            opt.zero_grad()
            xb, sb, tb = batch[0], batch[1], batch[2]
            phi = enc(xb)
            s_hat = hs_t(phi, tb)
            loss = mse(s_hat, sb)
            if use_y_loss and hy_ft is not None:
                yb = batch[3]
                loss = loss + lam_y * mse(hy_ft(s_hat, phi), yb)
            loss.backward(); opt.step()
            ep_loss += float(loss.item()) * xb.size(0)
        losses.append(ep_loss / max(1, n_train))
    enc_cpu = enc.cpu().eval()
    enc_cpu._finetune_losses = losses   # diagnostic — read by convergence checker
    return enc_cpu


def finetune_ftY(enc_pretrained: nn.Module, exp: Dict, cfg: Dict,
                 device: torch.device, seed: int,
                 freeze_backbone: bool = False,
                 unfreeze_layers: int = 1) -> nn.Module:
    """Finetune encoder on (X_exp, T, Y_exp) — exposes bypass(Z_y) via [phi, T] head."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    enc = deepcopy(enc_pretrained).to(device)
    hy_ft = HeadYT(cfg["phi_dim"]).to(device)
    if freeze_backbone:
        enc_params = _freeze_backbone_keep_last_linear(enc, n_layers=unfreeze_layers)
    else:
        enc_params = list(enc.parameters())
    opt = torch.optim.Adam(enc_params + list(hy_ft.parameters()),
                           lr=cfg.get("finetune_lr", 1e-3))
    bs = min(cfg.get("finetune_batch_size", 64), len(exp["X"]))
    Xt = torch.tensor(exp["X"], dtype=torch.float32).to(device)
    Tt = torch.tensor(exp["T"], dtype=torch.float32).to(device)
    Yt = torch.tensor(exp["Y"], dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(Xt, Tt, Yt), batch_size=bs, shuffle=True)
    mse = nn.MSELoss()
    losses: List[float] = []
    n_train = len(exp["X"])
    for _ in range(cfg.get("finetune_epochs", 100)):
        enc.train(); hy_ft.train()
        ep_loss = 0.0
        for xb, tb, yb in loader:
            opt.zero_grad()
            phi = enc(xb)
            loss = mse(hy_ft(phi, tb), yb)
            loss.backward(); opt.step()
            ep_loss += float(loss.item()) * xb.size(0)
        losses.append(ep_loss / max(1, n_train))
    enc_cpu = enc.cpu().eval()
    enc_cpu._finetune_losses = losses
    return enc_cpu
