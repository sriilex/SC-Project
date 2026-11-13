"""Small fixed-membership Sugeno calibration, fitted by exact 1-D search.

Gaussian rules on ONE diagnostic input blend per-rule positive width
multipliers; the lower and upper tails may carry different multipliers
(asymmetric). Quantiles are affine in the consequents, so the (optionally
recency-weighted) pinball objective plus L1 rule-difference shrinkage is a
piecewise-linear convex function. Every coordinate is minimized EXACTLY by a
sorted breakpoint sweep: no scipy, no linear program, no gradient descent.

The one-rule symmetric model is the constant W+L comparator, fitted with the
identical objective, bounds, and procedure. Missing diagnostics map to the
neutral membership input; this is not a promised identity correction.
"""

from __future__ import annotations

from datetime import date
from typing import Final

import numpy as np
import polars as pl

from ml.fuzzy_calibration import M_SCALE, location_shift_bound, matured_window

QUANTILES: Final[np.ndarray] = np.array([0.1, 0.5, 0.9])
FEATURES: Final[tuple[str, ...]] = ("vol_ratio", "error_ratio", "median_bias")
_MONTH_DAYS: Final[float] = 30.44


def compute_compact_inputs(
    forecasts: pl.DataFrame,
    daily_levels: pl.DataFrame,
    h: int,
    *,
    min_matured: int = 5,
    n_matured: int | None = None,
) -> pl.DataFrame:
    """Causal inputs using daily LOCF levels and ONLY supplied OOS residuals.

    ``daily_levels`` contains date/target_class/log_value, not forecast rows.
    Missing dates are inserted and forward-filled, never backfilled. Volatility
    uses changes on the continuous calendar, including issuance-day information.
    Residuals issued at t' are accessible at t iff t'+h <= t. Nonfinite outcomes
    do not count toward warm-up. Missing diagnostics use the neutral membership
    input at inference; this is NOT a promised identity correction.
    """
    if h < 1 or min_matured < 1:
        raise ValueError("h and min_matured must be positive")
    window = matured_window(h) if n_matured is None else n_matured
    if window < min_matured:
        raise ValueError("n_matured must be at least min_matured")
    _unique_keys(forecasts)
    _unique_keys(daily_levels)
    # Truncate before densifying so future levels cannot enter expanding history.
    daily = (
        daily_levels.filter(pl.col("date") <= forecasts["date"].max())
        .sort(["target_class", "date"])
        .upsample(time_column="date", every="1d", group_by="target_class")
        .sort(["target_class", "date"])
        .with_columns(pl.col("log_value").forward_fill().over("target_class"))
        .with_columns(pl.col("log_value").diff().over("target_class").alias("change"))
        .with_columns(
            pl.col("change")
            .rolling_std(14, min_samples=14, ddof=0)
            .over("target_class")
            .alias("v14"),
            pl.col("change")
            .rolling_std(90, min_samples=90, ddof=0)
            .over("target_class")
            .alias("v90"),
        )
        .with_columns(
            ((pl.col("v14") + 1e-9) / (pl.col("v90") + 1e-9)).log().alias("vol_ratio")
        )
        .select("date", "target_class", "vol_ratio")
    )
    result: list[dict[str, object]] = []
    for group in forecasts.sort(["target_class", "date"]).partition_by("target_class"):
        dates = group["date"].to_numpy().astype("datetime64[D]")
        y = group["y"].to_numpy().astype(float)
        q = group.select("p_0.1", "p_0.5", "p_0.9").to_numpy().astype(float)
        half = (q[:, 2] - q[:, 0]) / 2
        resid = y - q[:, 1]
        usable = np.isfinite(y) & np.isfinite(q).all(axis=1) & (half > 1e-9)
        for i, t in enumerate(dates):
            stop = np.searchsorted(dates, t - np.timedelta64(h, "D"), side="right")
            idx = np.flatnonzero(usable[:stop])[-window:]
            severity = bias = float("nan")
            if idx.size >= min_matured:
                severity = float(
                    np.log(max(np.quantile(np.abs(resid[idx]) / half[idx], 0.8), 1e-6))
                )
                bias = float(np.median(resid[idx]) / max(np.median(half[idx]), 1e-9))
            result.append(
                {
                    "date": group["date"][i],
                    "target_class": group["target_class"][i],
                    "error_ratio": severity,
                    "median_bias": bias,
                    "n_matured": int(idx.size),
                }
            )
    return (
        pl.DataFrame(result)
        .join(daily, on=["date", "target_class"], how="left")
        .with_columns(pl.col("vol_ratio").fill_null(float("nan")))
    )


def _unique_keys(frame: pl.DataFrame) -> None:
    if frame.select("date", "target_class").unique().height != frame.height:
        raise ValueError("duplicate date/target_class keys")


def membership_spec(values: np.ndarray, n_rules: int) -> dict[str, object]:
    """Fit only robust input normalization; Gaussian geometry stays fixed."""
    if n_rules not in (1, 2, 3):
        raise ValueError("n_rules must be 1, 2, or 3")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    center = float(np.median(finite)) if finite.size else 0.0
    scale = (
        float(np.subtract(*np.quantile(finite, [0.75, 0.25]))) if finite.size else 1.0
    )
    if scale < 1e-6:
        scale = 1.0
    centers = {1: [0.0], 2: [-0.75, 0.75], 3: [-1.0, 0.0, 1.0]}[n_rules]
    return {"center": center, "scale": scale, "centers": centers, "width": 1.0}


def rule_weights(values: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    """Stable normalized Gaussian firing, even far outside the fit support."""
    values = np.asarray(values, dtype=float)
    z = (values - float(spec["center"])) / float(spec["scale"])
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
    centers = np.asarray(spec["centers"], dtype=float)
    log_mu = -0.5 * ((z[:, None] - centers) / float(spec["width"])) ** 2
    log_mu -= log_mu.max(axis=1, keepdims=True)
    mu = np.exp(log_mu)
    return mu / mu.sum(axis=1, keepdims=True)


def _join_inputs(frame: pl.DataFrame, inputs: pl.DataFrame) -> pl.DataFrame:
    _unique_keys(frame)
    _unique_keys(inputs)
    joined = frame.join(
        inputs.select("date", "target_class", *FEATURES),
        on=["date", "target_class"],
        how="left",
        maintain_order="left",
    )
    if any(joined[f].null_count() for f in FEATURES):
        # NaN explicitly represents unavailable diagnostics; null means missing keys.
        raise ValueError("missing input keys; compute inputs for every forecast row")
    return joined


def _pl_argmin(
    base_slope: float,
    breakpoints: np.ndarray,
    jumps: np.ndarray,
    lo: float,
    hi: float,
    previous: float,
) -> float:
    """Exact minimizer of a convex piecewise-linear function on [lo, hi].

    The slope just right of ``lo`` is ``base_slope``; crossing breakpoint k
    changes the slope by ``jumps[k]`` (all > 0). Returns the minimizer closest
    to ``previous`` (the objective is flat on intervals between kinks).
    """
    if base_slope >= 0:
        return float(lo)
    if base_slope + float(jumps.sum()) <= 0:
        return float(hi)
    cum = np.cumsum(jumps)
    k = int(np.searchsorted(cum, -base_slope, side="left"))
    a = float(breakpoints[k])
    # Strictly increasing cum: a zero-slope interval starts exactly at k.
    b = float(breakpoints[k + 1]) if k + 1 < len(breakpoints) else float(hi)
    if base_slope + float(cum[k]) > 0:
        b = a  # single crossing point, not a flat interval
    return float(np.clip(previous, a, b))


def _recency_weights(dates: np.ndarray, half_life_months: float | None) -> np.ndarray:
    if half_life_months is None:
        return np.ones(len(dates))
    if not np.isfinite(half_life_months) or half_life_months <= 0:
        raise ValueError("recency_half_life_months must be positive")
    age_days = (dates.max() - dates) / np.timedelta64(1, "D")
    return 0.5 ** (age_days / (float(half_life_months) * _MONTH_DAYS))


def _fit_piecewise_linear(
    q: np.ndarray,
    y: np.ndarray,
    level: np.ndarray,
    w: np.ndarray,
    aw: np.ndarray,
    n_rules: int,
    asymmetric: bool,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    """Exact coordinate descent on the piecewise-linear convex objective.

    Quantile slopes per coordinate (per row i, weight w_i):
      k_lo_r: -0.9 c -> +0.1 c at p_lo = y, c = w_i a_i alpha_ir
      k_hi_r: -0.1 c -> +0.9 c at p_hi = y, c = w_i b_i alpha_ir
      s:      lo term -0.1w -> +0.9w at s = y-q5+a K_lo;
              hi term -0.1w -> +0.9w at s = y-q5-b K_hi;
              median  -0.5w -> +0.5w at s = y-q5
    Shrinkage (2*lambda/R) * |k_r - mean(k)| contributes a kink at the mean of
    the OTHER rules with slope change 2 * (2*lambda/R) * (R-1)/R.
    """
    q1, q5, q9 = q[:, 0], q[:, 1], q[:, 2]
    a = q5 - q1
    b = q9 - q5
    k_lo = np.ones(n_rules)
    k_hi = np.ones(n_rules) if asymmetric else None
    s = 0.0
    kb_lo = float(np.exp(-M_SCALE))
    kb_hi = float(np.exp(M_SCALE))
    s_bound = location_shift_bound(float(np.std(y - level)))
    sb_lo, sb_hi = -s_bound, s_bound
    pc = (
        (2.0 * shrinkage / n_rules) * (n_rules - 1) / n_rules if n_rules > 1 else 0.0
    )
    # k_r - mean(k) = (R-1)/R * (k_r - mean of the others)
    pen_kink_scale = (2.0 * shrinkage / n_rules) * (n_rules - 1) / n_rules
    n_loss_scale = float(w.sum())

    def predict() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K_lo = aw @ k_lo
        K_hi = aw @ k_hi if asymmetric else K_lo
        med = q5 + s
        return med - a * K_lo, med, med + b * K_hi

    def objective() -> float:
        p_lo, med, p_hi = predict()
        err = y[:, None] - np.stack([p_lo, med, p_hi], axis=1)
        pin = np.maximum(QUANTILES * err, (QUANTILES - 1) * err).mean(axis=1)
        loss = float((w * pin).sum() / (3.0 * n_loss_scale * float(np.std(y - level) or 1e-6)))
        pen = 0.0
        if pc:
            pen += float(np.abs(k_lo - k_lo.mean()).sum()) * (
                (2.0 * shrinkage / n_rules) * (n_rules - 1) / n_rules
            )
            if asymmetric:
                pen += float(np.abs(k_hi - k_hi.mean()).sum()) * (
                    (2.0 * shrinkage / n_rules) * (n_rules - 1) / n_rules
                )
        return loss + pen

    def kink_mean(k: np.ndarray, r: int) -> float:
        return float(k.sum() - k[r]) / (n_rules - 1) if n_rules > 1 else float("nan")

    sweeps = 0
    prev = objective()
    for sweeps in range(1, 101):
        med = q5 + s
        # --- lower-tail rule multipliers -----------------------------------
        for r in range(n_rules):
            rest = aw @ np.delete(k_lo, r) if n_rules > 1 else np.zeros(len(y))
            c = w * a * aw[:, r]
            bp = (med - y) / a - rest
            base = float(
                np.where(bp <= kb_lo, 0.1 * c, -0.9 * c).sum()
                + (-pc if kb_lo < kink_mean(k_lo, r) else pc) if pc else 0.0
            )
            jumps = c
            bps = bp
            if pc:
                km = kink_mean(k_lo, r)
                inside = (kb_lo < km < kb_hi)
                if inside:
                    bps = np.append(bp, km)
                    jumps = np.append(c, 2.0 * pc)
            keep = (bps > kb_lo) & (bps < kb_hi) & (jumps > 0)
            order = np.argsort(bps[keep])
            k_lo[r] = _pl_argmin(
                base, bps[keep][order], jumps[keep][order], kb_lo, kb_hi, k_lo[r]
            )
        # --- upper-tail rule multipliers -----------------------------------
        if asymmetric:
            for r in range(n_rules):
                rest = aw @ np.delete(k_hi, r) if n_rules > 1 else np.zeros(len(y))
                c = w * b * aw[:, r]
                bp = (y - med) / b - rest
                km = kink_mean(k_hi, r)
                base = float(np.where(bp <= kb_lo, 0.9 * c, -0.1 * c).sum())
                if pc:
                    base += -pc if kb_lo < km else pc
                bps = np.append(bp, km) if pc else bp
                jumps = np.append(c, 2.0 * pc) if pc else c
                keep = (bps > kb_lo) & (bps < kb_hi) & (jumps > 0)
                order = np.argsort(bps[keep])
                k_hi[r] = _pl_argmin(
                    base, bps[keep][order], jumps[keep][order], kb_lo, kb_hi, k_hi[r]
                )
        # --- global location shift ------------------------------------------
        K_lo = aw @ k_lo
        K_hi = aw @ k_hi if asymmetric else K_lo
        bps = np.concatenate(
            [(y - q5) + a * K_lo, (y - q5) - b * K_hi, y - q5]
        )
        jumps = np.concatenate([w, w, w])
        base = float(
            np.where(((y - q5) + a * K_lo) <= sb_lo, 0.9 * w, -0.1 * w).sum()
            + np.where(((y - q5) - b * K_hi) <= sb_lo, 0.9 * w, -0.1 * w).sum()
            + np.where((y - q5) <= sb_lo, 0.5 * w, -0.5 * w).sum()
        )
        keep = (bps > sb_lo) & (bps < sb_hi) & (jumps > 0)
        order = np.argsort(bps[keep])
        s = _pl_argmin(base, bps[keep][order], jumps[keep][order], sb_lo, sb_hi, s)
        cur = objective()
        if abs(prev - cur) <= 1e-12 * max(1.0, abs(cur)):
            prev = cur
            break
        prev = cur
    p_lo, med, p_hi = predict()
    err = y[:, None] - np.stack([p_lo, med, p_hi], axis=1)
    fit_pinball = float(np.maximum(QUANTILES * err, (QUANTILES - 1) * err).mean())
    return k_lo, (k_hi if asymmetric else k_lo.copy()), s, objective(), sweeps, fit_pinball


def fit_compact(
    fit_frame: pl.DataFrame,
    inputs: pl.DataFrame,
    h: int,
    *,
    n_rules: int = 3,
    asymmetric: bool = False,
    width_feature: str = "vol_ratio",
    shrinkage: float = 0.01,
    recency_half_life_months: float | None = None,
) -> dict[str, object]:
    """Fit a bounded Sugeno FIS by exact 1-D piecewise-linear coordinate search.

    The objective is the weighted mean pinball of the three calibrated
    quantiles (same terms scored by ``row_loss``), scaled by the fit-window
    target-return SD, plus L1 shrinkage of each rule multiplier toward the mean
    of the OTHER multipliers in its tail. The one-rule symmetric model IS the
    constant W+L comparator: identical objective, bounds, and procedure.
    Finite base quantiles, positive base widths, and outcomes are required;
    reversed base quantiles are rejected.
    """
    if width_feature not in FEATURES:
        raise ValueError("unknown feature")
    if not (1 <= n_rules <= 3):
        raise ValueError("n_rules must be 1, 2, or 3")
    if not np.isfinite(shrinkage) or shrinkage < 0:
        raise ValueError("shrinkage must be finite and nonnegative")
    train = _join_inputs(fit_frame, inputs)
    q = train.select("p_0.1", "p_0.5", "p_0.9").to_numpy().astype(float)
    y = train["y"].to_numpy().astype(float)
    level = train["log_value"].to_numpy().astype(float)
    dates = train["date"].to_numpy()
    if not train.height or not (
        np.isfinite(q).all() and np.isfinite(y).all() and np.isfinite(level).all()
    ):
        raise ValueError(
            "fit rows must have finite base quantiles, levels, and outcomes"
        )
    if np.any(np.diff(q, axis=1) <= 0):
        raise ValueError("base quantiles must be strictly ordered")
    w = _recency_weights(dates, recency_half_life_months)
    wspec = membership_spec(train[width_feature].to_numpy(), n_rules)
    aw = rule_weights(train[width_feature].to_numpy(), wspec)
    k_lo, k_hi, s, objective, sweeps, fit_pinball = _fit_piecewise_linear(
        q, y, level, w, aw, n_rules, asymmetric, shrinkage
    )
    s_bound = location_shift_bound(float(np.std(y - level)))
    return {
        "version": "compact-sugeno-v2",
        "h": h,
        "n_rules": n_rules,
        "asymmetric": asymmetric,
        "width_feature": width_feature,
        "width_memberships": wspec,
        "width_consequents_lo": k_lo.tolist(),
        "width_consequents_hi": k_hi.tolist(),
        "location_shift": float(s),
        "width_bounds": [float(np.exp(-M_SCALE)), float(np.exp(M_SCALE))],
        "location_bound": s_bound,
        "shrinkage": shrinkage,
        "recency_half_life_months": (
            float(recency_half_life_months)
            if recency_half_life_months is not None
            else None
        ),
        "n_parameters": n_rules + (n_rules if asymmetric else 0) + 1,
        "stats": {
            "n_fit": train.height,
            "loss_scale": float(np.std(y - level)),
            "fit_pinball": fit_pinball,
            "objective": objective,
            "sweeps": sweeps,
            "solver": "exact-1d-piecewise-linear",
        },
    }


def compact_effects(
    pred: pl.DataFrame, inputs: pl.DataFrame, params: dict[str, object]
) -> pl.DataFrame:
    """Expose firing strengths and actual correction variability for audits."""
    frame = _join_inputs(pred, inputs)
    aw = rule_weights(
        frame[str(params["width_feature"])].to_numpy(), params["width_memberships"]
    )
    k_lo = np.asarray(params["width_consequents_lo"], dtype=float)
    k_hi = np.asarray(params["width_consequents_hi"], dtype=float)
    K_lo = aw @ k_lo
    K_hi = aw @ k_hi
    shift = float(params["location_shift"])
    cols = [
        pl.Series("width_multiplier_lo", K_lo),
        pl.Series("width_multiplier_hi", K_hi),
        pl.Series("width_multiplier", (K_lo + K_hi) / 2),
        pl.Series("location_shift", np.full(len(K_lo), shift)),
    ]
    cols += [pl.Series(f"width_rule_{r}", aw[:, r]) for r in range(aw.shape[1])]
    return frame.select("date", "target_class").with_columns(cols)


def apply_compact(
    pred: pl.DataFrame, inputs: pl.DataFrame, params: dict[str, object]
) -> pl.DataFrame:
    """Apply frozen parameters without reading outcomes or changing row order."""
    effects = compact_effects(pred, inputs, params)
    q = pred.select("p_0.1", "p_0.5", "p_0.9").to_numpy().astype(float)
    k_lo = effects["width_multiplier_lo"].to_numpy()
    k_hi = effects["width_multiplier_hi"].to_numpy()
    s = effects["location_shift"].to_numpy()
    if not np.isfinite(q).all() or np.any(np.diff(q, axis=1) < 0):
        raise ValueError("base quantiles must be finite and ordered")
    median = q[:, 1] + s
    return pred.with_columns(
        pl.Series("p_0.1", median - (q[:, 1] - q[:, 0]) * k_lo),
        pl.Series("p_0.5", median),
        pl.Series("p_0.9", median + (q[:, 2] - q[:, 1]) * k_hi),
    )


def matured_fit_rows(frame: pl.DataFrame, score_start: date, h: int) -> pl.DataFrame:
    """Purge labels unavailable immediately BEFORE a scoring window starts."""
    return frame.filter(
        pl.col("date") + pl.duration(days=h) < pl.lit(score_start)
    ).filter(pl.col("y").is_not_null() & pl.col("y").is_finite())
