"""Sugeno-style fuzzy calibration for the frozen quantile LSTM.

Design (Tasks F):

* Two small differentiable Sugeno systems. Gaussian membership functions with
  learnable centres and widths, product t-norm, weighted-average
  defuzzification, a 3x3 rule base, constant or linear consequents.
    - ``FIS_width``: (cov_gap, vol) -> log width multiplier ``m`` (tanh-bounded).
    - ``FIS_loc``  : (bias_z, vol) -> location shift ``s``.
* Inputs are recomputed **per forecast date** from information available then.
  Forecast residuals are *matured*: a forecast issued at ``t`` for horizon ``h``
  matures at ``t + h``, so the rolling window for a forecast at ``t`` may only
  contain forecasts issued at or before ``t - h``.
    - ``cov_gap`` = P(y < P10) + P(y > P90) - 0.20 over the last N matured rows
      (positive = intervals too narrow). Only used at h=1 and h=7.
    - ``vol``     = trailing std of daily log-value changes, z-scored against its
      expanding history.
    - ``bias_z``  = mean(y - P50) / std(y - P50) over the last N matured rows.
* The fuzzy parameters are trained (base model frozen) by minimising the summed
  pinball loss over {0.1, 0.5, 0.9}; consequents start at zero so the untrained
  layer is the identity.
* Inference (:func:`calibrate`) is numpy-only and reads a plain JSON parameter
  dict. It never imports torch.

Application::

    p50' = p50 + s
    p10' = p50' - (p50 - p10) * exp(m)
    p90' = p50' + (p90 - p50) * exp(m)

so ordering is preserved by construction.
"""
from __future__ import annotations

import itertools
import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
FIG_DIR: Final[Path] = REPO_ROOT / "docs" / "figures"
PARAM_DIR: Final[Path] = REPO_ROOT / "src" / "data" / "fuzzy_params"

#: Bound on the log width multiplier: exp(m) in [exp(-M), exp(M)]. The width
#: multiplier is dimensionless (a log-multiplier on the base interval), so it
#: needs no horizon scaling.
M_SCALE: Final[float] = 0.7
#: Location bound is a fixed multiple of the horizon's target standard
#: deviation, so it is comparable across horizons. ``S_SCALE_PER_SD`` is set
#: from the pre-registered h=1 bound (0.15 ~= 2.2 x the h=1 target SD of
#: 0.068); a constant 0.15 was 2.2 x the h=1 SD but only 0.28 x the h=90 SD.
S_SCALE_PER_SD: Final[float] = 2.20
#: Rolling window length (in matured forecasts) for cov_gap / bias_z at h=1.
N_MATURED: Final[int] = 60
#: Minimum matured rows before cov_gap / bias_z are considered usable.
MIN_MATURED: Final[int] = 5
#: Number of non-overlapping (independent-ish) residuals the matured window
#: should span. At h=90 a 60-row window holds only ~1 such observation, so the
#: window is widened to ``N_MATURED_INDEPENDENT * h``.
N_MATURED_INDEPENDENT: Final[int] = 5
#: Trailing window (days) for the volatility input.
VOL_WINDOW: Final[int] = 14


def matured_window(h: int) -> int:
    """Rolling matured-residual window for horizon ``h``.

    Overlapping h-step residuals carry roughly one independent observation per
    ``h`` rows, so a fixed 60-row window spans ~60/h independent observations:
    ~5 at h=7 but only ~0.7 at h=90. Widening the window to ``5 * h`` keeps
    the effective sample roughly constant across horizons.
    """
    return max(N_MATURED, N_MATURED_INDEPENDENT * h)


def location_shift_bound(scale: float) -> float:
    """Location-shift bound for a horizon whose target SD is ``scale``."""
    bound = S_SCALE_PER_SD * float(scale)
    return bound if math.isfinite(bound) and bound > 0 else 0.15

FuzzyParams = dict[str, object]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def compute_fuzzy_inputs(
    frame: pl.DataFrame,
    h: int,
    n_matured: int | None = None,
    vol_window: int = VOL_WINDOW,
    min_matured: int = MIN_MATURED,
    matured_from: date | None = None,
) -> pl.DataFrame:
    """Per-row causal fuzzy inputs: ``vol_z``, ``cov_gap``, ``bias_z``.

    ``frame`` must contain ``date, target_class, log_value, p_0.1, p_0.5,
    p_0.9, y``. ``y`` is the realised h-step level for the issue date on that
    row, i.e. the outcome that matures ``h`` days later. A row at date ``t``
    only sees rows with ``date <= t - h``.

    ``n_matured`` defaults to :func:`matured_window` for ``h``. Pass
    ``matured_from`` to add a burn-in: matured residuals from before that date
    (e.g. in-sample train residuals when calibrating on valid) are ignored, so
    the first rows with too few out-of-sample outcomes fall back to the
    identity window and contribute no cov_gap / bias_z signal.
    """
    n_matured = matured_window(h) if n_matured is None else n_matured
    rows: list[dict[str, object]] = []
    for cls_frame in frame.sort(["target_class", "date"]).partition_by("target_class", maintain_order=True):
        cls = str(cls_frame["target_class"][0])
        g = cls_frame.sort("date")
        n = g.height
        if n == 0:
            continue
        logv = g["log_value"].to_numpy().astype(float)
        p10 = g["p_0.1"].to_numpy().astype(float)
        p50 = g["p_0.5"].to_numpy().astype(float)
        p90 = g["p_0.9"].to_numpy().astype(float)
        yv = g["y"].to_numpy().astype(float)
        dates = g["date"].to_numpy().astype("datetime64[D]")

        # Trailing volatility and its expanding z-score (stats from t' < t only).
        d = np.diff(logv, prepend=np.nan)
        vol = np.full(n, np.nan)
        if n >= vol_window:
            for i in range(vol_window - 1, n):
                seg = d[i - vol_window + 1 : i + 1]
                if np.sum(~np.isnan(seg)) >= max(3, vol_window // 2):
                    vol[i] = np.nanstd(seg)
        vol_z = np.zeros(n)
        csum = csum2 = 0.0
        cnt = 0
        for i in range(n):
            if cnt > 1:
                mu = csum / cnt
                var = max(csum2 / cnt - mu * mu, 0.0)
                sd = math.sqrt(var)
            else:
                mu, sd = 0.0, 1.0
            if not np.isnan(vol[i]):
                vol_z[i] = (vol[i] - mu) / (sd + 1e-9)
                csum += vol[i]
                csum2 += vol[i] * vol[i]
                cnt += 1
        vol_z = np.nan_to_num(vol_z, nan=0.0, posinf=0.0, neginf=0.0)

        below = np.where(np.isnan(yv), np.nan, (yv < p10).astype(float))
        above = np.where(np.isnan(yv), np.nan, (yv > p90).astype(float))
        resid = yv - p50
        floor = (
            int(np.searchsorted(dates, np.datetime64(matured_from), side="left"))
            if matured_from is not None
            else 0
        )

        for i in range(n):
            cutoff = dates[i] - np.timedelta64(h, "D")
            j = int(np.searchsorted(dates, cutoff, side="right"))
            lo = max(floor, j - n_matured)
            count = j - lo
            cov_gap = 0.0
            bias_z = 0.0
            if count >= min_matured:
                b = below[lo:j]
                a = above[lo:j]
                if not np.all(np.isnan(b)):
                    cov_gap = float(np.nanmean(b) + np.nanmean(a) - 0.20)
                r = resid[lo:j]
                r = r[~np.isnan(r)]
                if r.size > 1 and r.std() > 1e-9:
                    bias_z = float(r.mean() / r.std())
            rows.append({
                "date": g["date"][i],
                "target_class": cls,
                "vol_z": float(vol_z[i]),
                "cov_gap": cov_gap,
                "bias_z": bias_z,
                "n_matured": count,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# numpy inference (no torch)
# ---------------------------------------------------------------------------

def _fis_numpy(spec: dict[str, object], x: np.ndarray) -> np.ndarray:
    """Evaluate a Sugeno FIS stored in ``spec`` on ``x`` (n, n_inputs)."""
    centers = np.asarray(spec["centers"], dtype=float)
    widths = np.asarray(spec["widths"], dtype=float)
    n_inputs = centers.shape[0]
    n_mfs = centers.shape[1]
    x = np.atleast_2d(x)
    if x.shape[1] != n_inputs:
        raise ValueError(f"expected {n_inputs} inputs, got {x.shape[1]}")
    d = (x[:, :, None] - centers[None, :, :]) / (widths[None, :, :] + 1e-9)
    mu = np.exp(-0.5 * d * d)  # (n, n_inputs, n_mfs)
    rule_idx = list(itertools.product(range(n_mfs), repeat=n_inputs))
    w = np.ones((x.shape[0], len(rule_idx)))
    for j, idx in enumerate(rule_idx):
        for i, mi in enumerate(idx):
            w[:, j] *= mu[:, i, mi]
    cons = np.asarray(spec["consequents"], dtype=float).reshape(-1)
    f = np.tile(cons[None, :], (x.shape[0], 1))
    lin = spec.get("linear_coefs")
    if lin is not None:
        lin = np.asarray(lin, dtype=float)  # (R, n_inputs)
        f = f + (lin[None, :, :] * x[:, None, :]).sum(axis=2)
    num = (w * f).sum(axis=1)
    den = w.sum(axis=1) + 1e-12
    out = num / den
    scale = float(spec.get("output_scale", 1.0))
    if spec.get("bounded", True):
        out = scale * np.tanh(out)
    return out


def _adjust_numpy(base: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Apply (m, s) to base columns [p10, p50, p90] (n, 3); numpy."""
    p10, p50, p90 = base[:, 0], base[:, 1], base[:, 2]
    p50_new = p50 + s
    mult = np.exp(m)
    p10_new = p50_new - (p50 - p10) * mult
    p90_new = p50_new + (p90 - p50) * mult
    out = np.stack([p10_new, p50_new, p90_new], axis=1)
    # Numerical safety: enforce ordering.
    out[:, 0] = np.minimum(out[:, 0], out[:, 1])
    out[:, 2] = np.maximum(out[:, 2], out[:, 1])
    return out


def _norm_inputs(x: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    mean = np.asarray(spec["norm_mean"], dtype=float)
    std = np.asarray(spec["norm_std"], dtype=float)
    return (x - mean) / (std + 1e-9)


def calibrate(
    pred_df: pl.DataFrame,
    history_df: pl.DataFrame,
    h: int,
    params: FuzzyParams,
    matured_from: date | None = None,
) -> pl.DataFrame:
    """Apply the trained Sugeno FIS to raw quantiles. numpy-only.

    Same schema and row order as ``pred_df``. ``history_df`` must contain the
    base quantiles and realised outcomes used to recompute causal inputs.
    """
    inputs = compute_fuzzy_inputs(history_df, h, matured_from=matured_from)
    joined = pred_df.join(
        inputs.select("date", "target_class", "vol_z", "cov_gap", "bias_z"),
        on=["date", "target_class"],
        how="left",
    )
    n = joined.height
    vol = np.nan_to_num(joined["vol_z"].to_numpy().astype(float))
    cov = np.nan_to_num(joined["cov_gap"].to_numpy().astype(float))
    bias = np.nan_to_num(joined["bias_z"].to_numpy().astype(float))

    width_spec = params["width"]  # type: ignore[index]
    loc_spec = params["loc"]  # type: ignore[index]
    width_cols = list(width_spec["inputs"])  # type: ignore[index]
    loc_cols = list(loc_spec["inputs"])  # type: ignore[index]

    def _build(cols: list[str]) -> np.ndarray:
        mapping = {"vol_z": vol, "cov_gap": cov, "bias_z": bias}
        if not cols:
            return np.zeros((n, 0))
        return np.stack([mapping[c] for c in cols], axis=1)

    xw = _build(width_cols)
    xl = _build(loc_cols)
    m = _fis_numpy(width_spec, _norm_inputs(xw, width_spec)) if width_cols else np.zeros(n)
    s = _fis_numpy(loc_spec, _norm_inputs(xl, loc_spec)) if loc_cols else np.zeros(n)

    base = np.stack([
        joined["p_0.1"].to_numpy().astype(float),
        joined["p_0.5"].to_numpy().astype(float),
        joined["p_0.9"].to_numpy().astype(float),
    ], axis=1)
    adj = _adjust_numpy(base, m, s)
    return pred_df.with_columns(
        pl.Series("p_0.1", adj[:, 0]),
        pl.Series("p_0.5", adj[:, 1]),
        pl.Series("p_0.9", adj[:, 2]),
    )


# ---------------------------------------------------------------------------
# torch training (base model frozen)
# ---------------------------------------------------------------------------

def _sugeno_torch(n_inputs: int, n_mfs: int, linear: bool, output_scale: float, bounded: bool):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SugenoFIS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_inputs = n_inputs
            self.n_mfs = n_mfs
            self.linear = linear
            self.output_scale = output_scale
            self.bounded = bounded
            centers = torch.linspace(-1.0, 1.0, n_mfs).repeat(n_inputs, 1)
            self.centers = nn.Parameter(centers)
            self.log_widths = nn.Parameter(torch.full((n_inputs, n_mfs), math.log(0.7)))
            rule_idx = list(itertools.product(range(n_mfs), repeat=n_inputs))
            self.register_buffer("rule_idx", torch.tensor(rule_idx, dtype=torch.long))
            self.consequents = nn.Parameter(torch.zeros(len(rule_idx), 1))
            if linear:
                self.lin = nn.Parameter(torch.zeros(len(rule_idx), n_inputs, 1))
            else:
                self.lin = None

        def forward(self, x):
            widths = F.softplus(self.log_widths) + 1e-3
            d = (x.unsqueeze(2) - self.centers.unsqueeze(0)) / widths.unsqueeze(0)
            mu = torch.exp(-0.5 * d * d)
            w = torch.ones(x.shape[0], self.rule_idx.shape[0], device=x.device)
            for i in range(self.n_inputs):
                idx = self.rule_idx[:, i]
                w = w * mu[:, i, :].index_select(1, idx)
            f = self.consequents.unsqueeze(0)
            if self.lin is not None:
                f = f + (self.lin.unsqueeze(0) * x.unsqueeze(1).unsqueeze(-1)).sum(dim=2)
            num = (w.unsqueeze(-1) * f).sum(dim=1)
            den = w.sum(dim=1, keepdim=True) + 1e-9
            out = (num / den).squeeze(-1)
            if self.bounded:
                out = self.output_scale * torch.tanh(out)
            return out

    return SugenoFIS()


def _spec_from_torch(fis, inputs: list[str], norm_mean: np.ndarray, norm_std: np.ndarray) -> dict[str, object]:
    import torch.nn.functional as F

    widths = (F.softplus(fis.log_widths) + 1e-3).detach().cpu().numpy()
    linear = fis.linear
    spec: dict[str, object] = {
        "inputs": inputs,
        "centers": fis.centers.detach().cpu().numpy().tolist(),
        "widths": widths.tolist(),
        "consequents": fis.consequents.detach().cpu().numpy().reshape(-1).tolist(),
        "linear_coefs": None if not linear else fis.lin.detach().cpu().numpy().reshape(len(fis.rule_idx), -1).tolist(),
        "output_scale": float(fis.output_scale),
        "bounded": bool(fis.bounded),
        "n_mfs": int(fis.n_mfs),
        "norm_mean": np.asarray(norm_mean).tolist(),
        "norm_std": np.asarray(norm_std).tolist(),
    }
    return spec


def fit_fuzzy(
    context_frame: pl.DataFrame,
    train_frame: pl.DataFrame,
    h: int,
    use_width: bool = True,
    use_loc: bool = True,
    with_vol: bool = True,
    linear: bool = False,
    epochs: int = 600,
    lr: float = 0.05,
    seed: int = 0,
    matured_from: date | None = None,
) -> tuple[FuzzyParams, dict[str, object]]:
    """Train the fuzzy parameters on ``train_frame`` (base quantiles frozen).

    ``context_frame`` is the history+split frame (base quantiles and matured
    outcomes) from which causal inputs are computed. ``train_frame`` selects
    the rows to fit on. ``with_vol=False`` drops the volatility input from both
    systems (ablation); at h >= 30 the width system is volatility-only, so that
    arm then receives no width input at all. ``matured_from`` forwards a
    burn-in to :func:`compute_fuzzy_inputs`.
    """
    import torch

    from ml.models.lstm import pinball_loss, set_seed

    set_seed(seed)

    inputs_all = compute_fuzzy_inputs(context_frame, h, matured_from=matured_from)
    train = train_frame.join(
        inputs_all.select("date", "target_class", "vol_z", "cov_gap", "bias_z"),
        on=["date", "target_class"],
        how="left",
    ).drop_nulls(["y", "p_0.1", "p_0.5", "p_0.9", "log_value"])

    n = train.height
    if n == 0:
        raise ValueError("no training rows for fuzzy calibration")

    vol = np.nan_to_num(train["vol_z"].to_numpy().astype(float))
    cov = np.nan_to_num(train["cov_gap"].to_numpy().astype(float))
    bias = np.nan_to_num(train["bias_z"].to_numpy().astype(float))
    mapping = {"vol_z": vol, "cov_gap": cov, "bias_z": bias}

    width_cols = []
    if use_width:
        # cov_gap needs enough matured forecasts inside the split, so it is only
        # used at h < 30; at h >= 30 the width system is volatility-only. The
        # no-volatility arm then removes the width input entirely rather than
        # silently keeping it (which made ``novol`` identical to ``both`` there).
        width_cols = ["cov_gap"] if h < 30 else []
        if with_vol:
            width_cols = width_cols + ["vol_z"]
    loc_cols = (["bias_z"] + (["vol_z"] if with_vol else [])) if use_loc else []

    base = np.stack([
        train["p_0.1"].to_numpy().astype(float) - train["log_value"].to_numpy().astype(float),
        train["p_0.5"].to_numpy().astype(float) - train["log_value"].to_numpy().astype(float),
        train["p_0.9"].to_numpy().astype(float) - train["log_value"].to_numpy().astype(float),
    ], axis=1)
    y_ret = (train["y"].to_numpy().astype(float) - train["log_value"].to_numpy().astype(float))

    def _norm(cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack([mapping[c] for c in cols], axis=1) if cols else np.zeros((n, 0))
        mean = x.mean(axis=0) if x.size else np.zeros(0)
        std = x.std(axis=0) if x.size else np.zeros(0)
        std = np.where(std < 1e-9, 1.0, std)
        return x, mean, std

    xw, w_mean, w_std = _norm(width_cols)
    xl, l_mean, l_std = _norm(loc_cols)

    # Location bound is horizon-scaled to the calibration frame's target SD so
    # that the h=1 and h=90 systems are allowed comparable moves relative to
    # the quantity they are shifting.
    s_scale = location_shift_bound(float(np.nanstd(y_ret)))

    device = torch.device("cpu")
    width_fis = _sugeno_torch(len(width_cols), 3, linear, M_SCALE, True).to(device) if width_cols else None
    loc_fis = _sugeno_torch(len(loc_cols), 3, linear, s_scale, True).to(device) if loc_cols else None

    params = []
    if width_fis is not None:
        params += list(width_fis.parameters())
    if loc_fis is not None:
        params += list(loc_fis.parameters())
    if params:
        opt = torch.optim.Adam(params, lr=lr)
    else:
        opt = None

    tw = torch.tensor((xw - w_mean) / w_std, dtype=torch.float32) if width_cols else None
    tl = torch.tensor((xl - l_mean) / l_std, dtype=torch.float32) if loc_cols else None
    tbase = torch.tensor(base, dtype=torch.float32)
    ty = torch.tensor(y_ret, dtype=torch.float32)

    for _epoch in range(epochs):
        if opt is not None:
            opt.zero_grad()
        m = width_fis(tw) if width_fis is not None else torch.zeros(n)
        s = loc_fis(tl) if loc_fis is not None else torch.zeros(n)
        p50 = tbase[:, 1] + s
        mult = torch.exp(m)
        p10 = p50 - (tbase[:, 1] - tbase[:, 0]) * mult
        p90 = p50 + (tbase[:, 2] - tbase[:, 1]) * mult
        pred = torch.stack([p10, p50, p90], dim=1)
        loss = pinball_loss(pred, ty)
        if opt is not None:
            loss.backward()
            opt.step()

    width_spec = _spec_from_torch(width_fis, width_cols, w_mean, w_std) if width_fis is not None else {
        "inputs": [], "centers": [], "widths": [], "consequents": [],
        "linear_coefs": None, "output_scale": M_SCALE, "bounded": True, "n_mfs": 3,
        "norm_mean": [], "norm_std": [],
    }
    loc_spec = _spec_from_torch(loc_fis, loc_cols, l_mean, l_std) if loc_fis is not None else {
        "inputs": [], "centers": [], "widths": [], "consequents": [],
        "linear_coefs": None, "output_scale": s_scale, "bounded": True, "n_mfs": 3,
        "norm_mean": [], "norm_std": [],
    }

    n_matured = train.join(inputs_all.select("date", "target_class", "n_matured"), on=["date", "target_class"], how="left")
    stats = {
        "h": h,
        "n_train": n,
        "n_matured_window": matured_window(h),
        "n_matured_median": float(n_matured["n_matured"].median()),
        "n_matured_mean": float(n_matured["n_matured"].mean()),
        "frac_rows_with_matured": float((n_matured["n_matured"] >= MIN_MATURED).mean()),
        "s_scale": s_scale,
        "final_train_pinball": float(loss.item()),
        "width_inputs": width_cols,
        "loc_inputs": loc_cols,
    }
    return {"h": h, "width": width_spec, "loc": loc_spec, "stats": stats}, stats


def save_params(params: FuzzyParams, name: str) -> Path:
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    path = PARAM_DIR / f"{name}.json"
    path.write_text(json.dumps(params, indent=2))
    return path


def load_params(name: str) -> FuzzyParams:
    return json.loads((PARAM_DIR / f"{name}.json").read_text())


def describe_params(params: FuzzyParams) -> str:
    """Human-readable dump of membership functions and rule consequents."""
    lines: list[str] = [f"FIS for h={params.get('h')}"]
    for system in ("width", "loc"):
        spec = params[system]  # type: ignore[index]
        lines.append(f"\n[{system}] inputs={spec['inputs']} output_scale={spec['output_scale']}")
        centers = np.asarray(spec["centers"], dtype=float)
        widths = np.asarray(spec["widths"], dtype=float)
        for i, name in enumerate(spec["inputs"]):  # type: ignore[union-attr]
            lines.append(f"  {name}: " + " ".join(
                f"MF{k}(c={centers[i, k]:+.3f}, w={widths[i, k]:.3f})" for k in range(centers.shape[1])
            ))
        cons = np.asarray(spec["consequents"], dtype=float)
        lines.append(f"  consequents: {np.round(cons, 4).tolist()}")
    return "\n".join(lines)


def plot_fuzzy(params: FuzzyParams, name: str) -> list[Path]:
    """Plot membership functions and the width surface to ``docs/figures``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, system in zip(axes, ("width", "loc")):
        spec = params[system]  # type: ignore[index]
        centers = np.asarray(spec["centers"], dtype=float)
        widths = np.asarray(spec["widths"], dtype=float)
        for i, iname in enumerate(spec["inputs"]):  # type: ignore[union-attr]
            grid = np.linspace(-3, 3, 200)
            for k in range(centers.shape[1]):
                mu = np.exp(-0.5 * ((grid - centers[i, k]) / (widths[i, k] + 1e-9)) ** 2)
                ax.plot(grid, mu, label=f"{iname} MF{k}")
        ax.set_title(f"{system} membership functions")
        ax.set_xlabel("standardised input")
        ax.set_ylabel("membership")
        ax.legend(fontsize=6)
    fig.tight_layout()
    p = FIG_DIR / f"fuzzy_v2_membership_{name}.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    spec = params["width"]  # type: ignore[index]
    if len(spec["inputs"]) == 2:  # type: ignore[arg-type]
        grid = np.linspace(-2.5, 2.5, 60)
        xx, yy = np.meshgrid(grid, grid)
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)
        z = _fis_numpy(spec, _norm_inputs(x, spec)).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(5, 4))
        cs = ax.contourf(xx, yy, z, levels=20, cmap="RdBu_r")
        fig.colorbar(cs, ax=ax, label="log width multiplier m")
        ax.set_xlabel(spec["inputs"][0]); ax.set_ylabel(spec["inputs"][1])
        ax.set_title("FIS_width surface")
        fig.tight_layout()
        p = FIG_DIR / f"fuzzy_v2_surface_{name}.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths
