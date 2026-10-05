"""Sugeno-style fuzzy inference system (FIS) for forecast calibration.

Inputs
------
  mean_pct_error  -- recent percentage bias of the model (negative = under-pred).
  recent_volatility -- rolling standard deviation of log-returns (uncertainty proxy).

Output
------
  shift_factor    -- additive correction applied symmetrically to all three
                     quantile predictions (P10, P50, P90) in log space.

Defuzzification is weighted-average (Sugeno style): each rule fires with a
strength (min of its input MF degrees), and the output is the strength-weighted
mean of the crisp rule consequents. No external fuzzy library is required.

Usage
-----
    from ml.fuzzy_calibration import calibrate
    adjusted_df = calibrate(pred_df, history_df, h)
"""
from __future__ import annotations

from typing import Final

import polars as pl

# ---------------------------------------------------------------------------
# Membership functions (plain Python, no library required)
# ---------------------------------------------------------------------------

def _trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangle MF: rises linearly from a to b, falls linearly from b to c."""
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoid MF: flat-top from b to c, slopes on [a,b] and [c,d]."""
    if x <= a or x >= d:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if d != c else 1.0


# ---------------------------------------------------------------------------
# Linguistic variable definitions
# ---------------------------------------------------------------------------
# mean_pct_error: negative = model under-predicts, positive = over-predicts
#   range roughly [-0.15, +0.15] for freight rates in log space
# recent_volatility: rolling std of log returns over last 14 days
#   range roughly [0.0, 0.12]

def _mpe_negative(e: float) -> float:
    """Bias is significantly negative (model under-predicts)."""
    return _trapmf(e, -0.20, -0.15, -0.06, -0.02)


def _mpe_neutral(e: float) -> float:
    """Bias is near zero."""
    return _trimf(e, -0.06, 0.0, 0.06)


def _mpe_positive(e: float) -> float:
    """Bias is significantly positive (model over-predicts)."""
    return _trapmf(e, 0.02, 0.06, 0.15, 0.20)


def _vol_low(v: float) -> float:
    """Market is calm."""
    return _trapmf(v, 0.0, 0.0, 0.02, 0.05)


def _vol_medium(v: float) -> float:
    """Normal volatility."""
    return _trimf(v, 0.02, 0.05, 0.09)


def _vol_high(v: float) -> float:
    """High uncertainty."""
    return _trapmf(v, 0.06, 0.09, 0.20, 0.20)


# ---------------------------------------------------------------------------
# Sugeno FIS: rules & defuzzification
# ---------------------------------------------------------------------------
# Consequents are crisp scalars (log-space additive shift to P50/P10/P90).
# Positive shift => nudge forecasts up; negative => nudge down.

# Rule consequent singletons (additive log-space corrections)
_RULES: Final[list[tuple[float, float, float]]] = [
    # (mpe_strength,  vol_strength,  consequent)  -- filled dynamically per call
    # Rule 1: negative bias + low vol  => push up moderately
    # Rule 2: negative bias + high vol => push up aggressively
    # Rule 3: neutral bias + any vol   => no shift
    # Rule 4: positive bias + low vol  => push down moderately
    # Rule 5: positive bias + high vol => push down aggressively
    # Rule 6: any bias + very high vol => small push in bias direction (dampened)
]  # (populated in sugeno_shift below for clarity)


def sugeno_shift(mean_pct_error: float, recent_volatility: float) -> float:
    """Return the Sugeno defuzzified shift (scalar, log-space additive)."""
    e = float(mean_pct_error)
    v = float(recent_volatility)

    # Clamp inputs to defined universe
    e = max(-0.20, min(0.20, e))
    v = max(0.00, min(0.20, v))

    # Fire degrees (min-operator conjunction)
    rules: list[tuple[float, float]] = [
        # (firing_strength, consequent)
        (min(_mpe_negative(e), _vol_low(v)),    +0.030),   # R1
        (min(_mpe_negative(e), _vol_medium(v)), +0.050),   # R2
        (min(_mpe_negative(e), _vol_high(v)),   +0.080),   # R3
        (min(_mpe_neutral(e),  _vol_low(v)),    +0.000),   # R4
        (min(_mpe_neutral(e),  _vol_medium(v)), +0.000),   # R5
        (min(_mpe_neutral(e),  _vol_high(v)),   +0.010),   # R6 (vol squeeze)
        (min(_mpe_positive(e), _vol_low(v)),    -0.030),   # R7
        (min(_mpe_positive(e), _vol_medium(v)), -0.050),   # R8
        (min(_mpe_positive(e), _vol_high(v)),   -0.080),   # R9
    ]

    total_strength = sum(s for s, _ in rules)
    if total_strength < 1e-9:
        return 0.0
    return sum(s * c for s, c in rules) / total_strength


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _compute_inputs(
    pred_df: pl.DataFrame,
    history_df: pl.DataFrame,
    h: int,
    lookback: int = 60,
) -> tuple[float, float]:
    """Derive mean_pct_error and recent_volatility from history.

    Parameters
    ----------
    pred_df:     Raw quantile predictions (columns: date, target_class,
                 h, p_0.1, p_0.5, p_0.9).
    history_df:  Training-or-validation split with true targets
                 (columns: date, target_class, log_value, y_step_h{h}).
    h:           Forecast horizon.
    lookback:    Number of most-recent rows used to estimate bias/volatility.
    """
    target_col = f"y_step_h{h}"

    # --- Bias: compare past p_0.5 predictions vs realized log_value --------
    if target_col in history_df.columns:
        tail = (
            history_df.drop_nulls([target_col, "log_value"])
            .sort("date")
            .tail(lookback)
        )
        if tail.height > 0:
            # Proxy for model bias using RW as a stand-in if pred history is
            # unavailable; real bias = mean(y_hat - y) / mean(|y|)
            errs = tail[target_col] - tail["log_value"]
            y_bar = tail["log_value"].abs().mean()
            mean_pct_error = float(errs.mean()) / (float(y_bar) if y_bar else 1.0)
        else:
            mean_pct_error = 0.0
    else:
        mean_pct_error = 0.0

    # --- Volatility: rolling std of 1-day log returns ----------------------
    if "log_value" in history_df.columns and history_df.height > 1:
        tail_vol = (
            history_df.sort("date")
            .tail(lookback + 1)
            .select(pl.col("log_value").diff().drop_nulls())
        )
        recent_volatility = float(tail_vol["log_value"].std()) if tail_vol.height > 1 else 0.0
    else:
        recent_volatility = 0.0

    return mean_pct_error, recent_volatility


def calibrate(
    pred_df: pl.DataFrame,
    history_df: pl.DataFrame,
    h: int,
    lookback: int = 60,
) -> pl.DataFrame:
    """Apply Sugeno FIS calibration to raw P10/P50/P90 quantile forecasts.

    Parameters
    ----------
    pred_df:     Raw forecast DataFrame with columns:
                 date, target_class, h, p_0.1, p_0.5, p_0.9
    history_df:  Reference data (train/validation split) used to estimate
                 recent bias and volatility.  Must contain log_value and,
                 if available, the target column y_step_h{h}.
    h:           Forecast horizon (must match pred_df's ``h`` column values).
    lookback:    How many trailing rows of history to use for input estimation.

    Returns
    -------
    pl.DataFrame with the same schema as pred_df but with calibrated quantiles.
    """
    # Compute per-class inputs and apply a per-class shift for precision,
    # falling back to a pooled shift if a class has no history.
    classes = pred_df["target_class"].unique().to_list()
    shift_by_class: dict[str, float] = {}

    for cls in classes:
        cls_history = history_df.filter(pl.col("target_class") == cls)
        if cls_history.is_empty():
            cls_history = history_df  # fallback: pooled
        mpe, vol = _compute_inputs(pred_df, cls_history, h, lookback)
        shift_by_class[cls] = sugeno_shift(mpe, vol)

    # Build a mapping Series for vectorised apply
    shift_expr = pl.col("target_class").replace_strict(
        shift_by_class, default=0.0
    )

    return pred_df.with_columns(
        (pl.col("p_0.1") + shift_expr).alias("p_0.1"),
        (pl.col("p_0.5") + shift_expr).alias("p_0.5"),
        (pl.col("p_0.9") + shift_expr).alias("p_0.9"),
    )
