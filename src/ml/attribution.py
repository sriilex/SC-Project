"""Attribution baselines for the Sugeno calibration layer.

These baselines ask whether the fuzzy *structure* matters, or whether any
state-dependent correction with the same output bounds does just as well. All
of them are fit on the validation split with the same pinball objective and the
same bounded location/width maps as the fuzzy layer.

Supported kinds
---------------
``constant_width``
    One constant log width multiplier. No location shift.
``constant_width_location``
    One constant width multiplier plus one constant location shift.
``constant_width_rollbias``
    Constant width plus a causal rolling-bias location correction,
    ``s_t = bound * tanh(k * z_t)``, where ``z_t`` is the standardised matured
    bias signal.
``linear``
    A linear state-dependent calibrator on the same inputs as the fuzzy layer,
    with the same ``tanh`` output bounds for width and location. A small MLP is
    left as an optional extension because the linear arm already answers the
    "ordinary state-dependent regression" question at this sample size.

The functions mirror :mod:`ml.fuzzy_calibration`: inputs are recomputed
causally from matured forecasts, and the apply function preserves the input
frame's schema and row order.
"""
from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np
import polars as pl

from ml.fuzzy_calibration import (
    M_SCALE,
    compute_fuzzy_inputs,
    location_shift_bound,
)

KINDS: Final[tuple[str, ...]] = (
    "constant_width",
    "constant_width_location",
    "constant_width_rollbias",
    "linear",
)


def _training_arrays(
    context_frame: pl.DataFrame,
    train_frame: pl.DataFrame,
    h: int,
    matured_from: date | None,
) -> dict[str, object]:
    """Join causal inputs onto the fit rows and normalise them."""
    inputs = compute_fuzzy_inputs(context_frame, h, matured_from=matured_from)
    train = train_frame.join(
        inputs.select("date", "target_class", "vol_z", "cov_gap", "bias_z"),
        on=["date", "target_class"],
        how="left",
    ).drop_nulls(["y", "p_0.1", "p_0.5", "p_0.9", "log_value"])
    if train.height == 0:
        raise ValueError("no training rows for attribution calibrator")

    vol = np.nan_to_num(train["vol_z"].to_numpy().astype(float))
    cov = np.nan_to_num(train["cov_gap"].to_numpy().astype(float))
    bias = np.nan_to_num(train["bias_z"].to_numpy().astype(float))
    mapping = {"vol_z": vol, "cov_gap": cov, "bias_z": bias}

    width_cols = (["cov_gap"] if h < 30 else []) + ["vol_z"]
    loc_cols = ["bias_z", "vol_z"]

    def _norm(cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.stack([mapping[c] for c in cols], axis=1)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        return x, mean, std

    xw, w_mean, w_std = _norm(width_cols)
    xl, l_mean, l_std = _norm(loc_cols)
    logv = train["log_value"].to_numpy().astype(float)
    base = np.stack([
        train["p_0.1"].to_numpy().astype(float) - logv,
        train["p_0.5"].to_numpy().astype(float) - logv,
        train["p_0.9"].to_numpy().astype(float) - logv,
    ], axis=1)
    y_ret = train["y"].to_numpy().astype(float) - logv
    return {
        "n": train.height,
        "width_cols": width_cols,
        "loc_cols": loc_cols,
        "xw": xw,
        "xl": xl,
        "w_mean": w_mean,
        "w_std": w_std,
        "l_mean": l_mean,
        "l_std": l_std,
        "bias_raw": bias,
        "bias_mean": float(bias.mean()),
        "bias_std": float(bias.std()) if bias.std() > 1e-9 else 1.0,
        "base": base,
        "y_ret": y_ret,
        "s_scale": location_shift_bound(float(np.nanstd(y_ret))),
    }


def fit_attribution(
    context_frame: pl.DataFrame,
    train_frame: pl.DataFrame,
    h: int,
    kind: str,
    epochs: int = 600,
    lr: float = 0.05,
    seed: int = 0,
    matured_from: date | None = None,
) -> dict[str, object]:
    """Fit one attribution baseline on ``train_frame`` (base frozen)."""
    if kind not in KINDS:
        raise ValueError(f"unknown attribution kind {kind!r}")
    import torch

    from ml.models.lstm import pinball_loss, set_seed

    set_seed(seed)
    d = _training_arrays(context_frame, train_frame, h, matured_from)
    n = int(d["n"])
    s_scale = float(d["s_scale"])
    tbase = torch.tensor(d["base"], dtype=torch.float32)
    ty = torch.tensor(d["y_ret"], dtype=torch.float32)
    tw = torch.tensor((d["xw"] - d["w_mean"]) / d["w_std"], dtype=torch.float32)
    tl = torch.tensor((d["xl"] - d["l_mean"]) / d["l_std"], dtype=torch.float32)
    zb = torch.tensor(
        (np.asarray(d["bias_raw"]) - float(d["bias_mean"])) / float(d["bias_std"]),
        dtype=torch.float32,
    )

    raw_m = torch.zeros(1, requires_grad=True)
    params: list[torch.Tensor] = [raw_m]
    raw_s = None
    k_bias = None
    w_coef = w_bias = l_coef = l_bias = None

    if kind in ("constant_width_location",):
        raw_s = torch.zeros(1, requires_grad=True)
        params.append(raw_s)
    elif kind == "constant_width_rollbias":
        k_bias = torch.zeros(1, requires_grad=True)
        params.append(k_bias)
    elif kind == "linear":
        w_coef = torch.zeros(tw.shape[1], requires_grad=True)
        w_bias = torch.zeros(1, requires_grad=True)
        l_coef = torch.zeros(tl.shape[1], requires_grad=True)
        l_bias = torch.zeros(1, requires_grad=True)
        params += [w_coef, w_bias, l_coef, l_bias]

    opt = torch.optim.Adam(params, lr=lr)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        m = M_SCALE * torch.tanh(raw_m.squeeze())
        if kind == "constant_width":
            s = torch.zeros(n)
        elif kind == "constant_width_location":
            s = s_scale * torch.tanh(raw_s.squeeze())
        elif kind == "constant_width_rollbias":
            s = s_scale * torch.tanh(k_bias.squeeze() * zb)
        else:
            m = M_SCALE * torch.tanh(tw @ w_coef + w_bias.squeeze())
            s = s_scale * torch.tanh(tl @ l_coef + l_bias.squeeze())
        p50 = tbase[:, 1] + s
        mult = torch.exp(m)
        p10 = p50 - (tbase[:, 1] - tbase[:, 0]) * mult
        p90 = p50 + (tbase[:, 2] - tbase[:, 1]) * mult
        loss = pinball_loss(torch.stack([p10, p50, p90], dim=1), ty)
        loss.backward()
        opt.step()

    out: dict[str, object] = {
        "kind": kind,
        "h": h,
        "n_train": n,
        "epochs": epochs,
        "lr": lr,
        "objective": "pinball_0.1_0.5_0.9",
        "matured_from": None if matured_from is None else str(matured_from),
        "width_inputs": d["width_cols"],
        "loc_inputs": d["loc_cols"],
        "width_scale": M_SCALE,
        "loc_scale": s_scale,
        "w_mean": np.asarray(d["w_mean"]).tolist(),
        "w_std": np.asarray(d["w_std"]).tolist(),
        "l_mean": np.asarray(d["l_mean"]).tolist(),
        "l_std": np.asarray(d["l_std"]).tolist(),
        "bias_mean": float(d["bias_mean"]),
        "bias_std": float(d["bias_std"]),
        "raw_m": float(raw_m.detach()),
        "m": float(M_SCALE * np.tanh(float(raw_m.detach()))),
        "raw_s": None if raw_s is None else float(raw_s.detach()),
        "s": None if raw_s is None else float(s_scale * np.tanh(float(raw_s.detach()))),
        "rollbias_coef": None if k_bias is None else float(k_bias.detach()),
        "width_coefs": None if w_coef is None else w_coef.detach().numpy().tolist(),
        "width_bias": None if w_bias is None else float(w_bias.detach()),
        "loc_coefs": None if l_coef is None else l_coef.detach().numpy().tolist(),
        "loc_bias": None if l_bias is None else float(l_bias.detach()),
        "final_train_pinball": float(loss.item()),
    }
    return out


def _adjust(base: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Apply bounded log-width ``m`` and location ``s``; keep ordering."""
    p10, p50, p90 = base[:, 0], base[:, 1], base[:, 2]
    p50_new = p50 + s
    mult = np.exp(m)
    p10_new = p50_new - (p50 - p10) * mult
    p90_new = p50_new + (p90 - p50) * mult
    out = np.stack([p10_new, p50_new, p90_new], axis=1)
    out[:, 0] = np.minimum(out[:, 0], out[:, 1])
    out[:, 2] = np.maximum(out[:, 2], out[:, 1])
    return out


def apply_attribution(
    pred_df: pl.DataFrame,
    history_df: pl.DataFrame,
    h: int,
    params: dict[str, object],
    matured_from: date | None = None,
) -> pl.DataFrame:
    """Apply a fitted attribution calibrator, preserving schema and row order."""
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
    mapping = {"vol_z": vol, "cov_gap": cov, "bias_z": bias}

    def _build(cols: list[str]) -> np.ndarray:
        return np.stack([mapping[c] for c in cols], axis=1) if cols else np.zeros((n, 0))

    kind = str(params["kind"])
    s_scale = float(params["loc_scale"])
    m_scale = float(params["width_scale"])
    if kind in ("constant_width", "constant_width_location", "constant_width_rollbias"):
        m = np.full(n, m_scale * np.tanh(float(params["raw_m"])))
        if kind == "constant_width":
            s = np.zeros(n)
        elif kind == "constant_width_location":
            s = np.full(n, s_scale * np.tanh(float(params["raw_s"])))
        else:
            z = (bias - float(params["bias_mean"])) / float(params["bias_std"])
            s = s_scale * np.tanh(float(params["rollbias_coef"]) * z)
    else:
        xw = (_build(list(params["width_inputs"])) - np.asarray(params["w_mean"], dtype=float)) / np.asarray(params["w_std"], dtype=float)
        xl = (_build(list(params["loc_inputs"])) - np.asarray(params["l_mean"], dtype=float)) / np.asarray(params["l_std"], dtype=float)
        m = m_scale * np.tanh(xw @ np.asarray(params["width_coefs"], dtype=float) + float(params["width_bias"]))
        s = s_scale * np.tanh(xl @ np.asarray(params["loc_coefs"], dtype=float) + float(params["loc_bias"]))

    base = np.stack([
        joined["p_0.1"].to_numpy().astype(float),
        joined["p_0.5"].to_numpy().astype(float),
        joined["p_0.9"].to_numpy().astype(float),
    ], axis=1)
    adj = _adjust(base, np.asarray(m, dtype=float), np.asarray(s, dtype=float))
    return pred_df.with_columns(
        pl.Series("p_0.1", adj[:, 0]),
        pl.Series("p_0.5", adj[:, 1]),
        pl.Series("p_0.9", adj[:, 2]),
    )
