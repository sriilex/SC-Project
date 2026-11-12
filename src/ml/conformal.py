"""Conformal / adaptive calibration baselines for the frozen quantile LSTM.

All three methods take the *frozen* base quantiles and adjust them:

  const_shift : additive shift equal to the median residual on a calibration split
  cqr         : split conformalized quantile regression (Romano et al. 2019)
  aci         : adaptive conformal inference / online quantile tracking
                (Gibbs and Candes 2021)

Every function returns a frame with the same schema and row order as the input
prediction frame. Ordering is enforced by construction so interval metrics are
well defined.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Final

import numpy as np
import polars as pl

QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)

#: ACI learning rate, warm-up length and score-buffer size. Exposed so they can
#: be written into run metadata instead of living as magic numbers.
ACI_GAMMA: Final[float] = 0.01
ACI_WARMUP: Final[int] = 20
ACI_SCORE_BUFFER: Final[int] = 1000


def attach_truth(pred: pl.DataFrame, source: pl.DataFrame, h: int) -> pl.DataFrame:
    """Join the realised h-step level (and current level) onto a prediction frame."""
    truth = source.select(
        "date", "target_class", "log_value",
        pl.col(f"y_step_h{h}").alias("y"),
    )
    return pred.join(truth, on=["date", "target_class"], how="inner").drop_nulls("y")


def _order(frame: pl.DataFrame) -> pl.DataFrame:
    """Enforce P10 <= P50 <= P90 (conformal adjustments can otherwise cross)."""
    lo = pl.min_horizontal("p_0.1", "p_0.5", "p_0.9")
    hi = pl.max_horizontal("p_0.1", "p_0.5", "p_0.9")
    mid = pl.min_horizontal(hi, pl.max_horizontal(lo, pl.col("p_0.5")))
    return frame.with_columns(
        lo.alias("p_0.1"), mid.alias("p_0.5"), hi.alias("p_0.9")
    )


# ---------------------------------------------------------------------------
# const_shift
# ---------------------------------------------------------------------------

def fit_const_shift(calib: pl.DataFrame, per_class: bool = True) -> dict[str, float]:
    """Median residual ``median(y - p50)`` on the calibration rows."""
    d = calib.with_columns((pl.col("y") - pl.col("p_0.5")).alias("r"))
    if not per_class:
        return {"POOLED": float(d["r"].median())}
    out = {
        row["target_class"]: float(row["med"])
        for row in d.group_by("target_class").agg(pl.col("r").median().alias("med")).iter_rows(named=True)
    }
    out["POOLED"] = float(d["r"].median())
    return out


def apply_const_shift(pred: pl.DataFrame, shifts: dict[str, float]) -> pl.DataFrame:
    per_class = {k: v for k, v in shifts.items() if k != "POOLED"}
    s = pl.col("target_class").replace_strict(per_class, default=shifts["POOLED"])
    return pred.with_columns(
        (pl.col("p_0.1") + s).alias("p_0.1"),
        (pl.col("p_0.5") + s).alias("p_0.5"),
        (pl.col("p_0.9") + s).alias("p_0.9"),
    ).pipe(_order)


# ---------------------------------------------------------------------------
# CQR
# ---------------------------------------------------------------------------

def fit_cqr(calib: pl.DataFrame, alpha: float = 0.2) -> dict[str, float]:
    """Split-conformal CQR scores from the calibration rows.

    ``e`` is the standard CQR conformity score for the central
    ``1 - alpha`` interval; ``med`` is the median residual used to shift P50.
    """
    d = calib.with_columns(
        pl.max_horizontal(
            pl.col("p_0.1") - pl.col("y"),
            pl.col("y") - pl.col("p_0.9"),
        ).alias("e"),
        (pl.col("p_0.1") - pl.col("y")).alias("lo"),
        (pl.col("y") - pl.col("p_0.9")).alias("hi"),
        (pl.col("y") - pl.col("p_0.5")).alias("med"),
    )
    return {
        "e_q": float(d["e"].quantile(1.0 - alpha)),
        "lo_q": float(d["lo"].quantile(1.0 - alpha)),
        "hi_q": float(d["hi"].quantile(1.0 - alpha)),
        "med_q": float(d["med"].quantile(0.5)),
    }


def apply_cqr(pred: pl.DataFrame, params: dict[str, float], symmetric: bool = True) -> pl.DataFrame:
    """Apply CQR. ``symmetric`` uses the canonical interval adjustment."""
    if symmetric:
        w = params["e_q"]
        lo_adj, hi_adj = w, w
    else:
        lo_adj, hi_adj = params["lo_q"], params["hi_q"]
    out = pred.with_columns(
        (pl.col("p_0.1") - lo_adj).alias("p_0.1"),
        (pl.col("p_0.5") + params["med_q"]).alias("p_0.5"),
        (pl.col("p_0.9") + hi_adj).alias("p_0.9"),
    )
    return _order(out)


# ---------------------------------------------------------------------------
# ACI
# ---------------------------------------------------------------------------

def aci_adjust(
    pred: pl.DataFrame,
    source: pl.DataFrame,
    h: int,
    gamma: float = ACI_GAMMA,
    alpha: float = 0.2,
    warmup: int = ACI_WARMUP,
    score_buffer: int = ACI_SCORE_BUFFER,
    state: dict[str, object] | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Adaptive conformal inference with delayed (matured) feedback.

    Maintains a per-class target miscoverage level ``alpha_t`` for the central
    ``1 - alpha`` interval and a rolling buffer of base conformity scores
    ``E = max(p10 - y, y - p90)``. The adjusted interval is
    ``[p10 - w_t, p90 + w_t]`` where ``w_t`` is the empirical ``1 - alpha_t``
    quantile of the *past, matured* scores, so the correction is bounded by the
    observed score distribution rather than drifting.

    A forecast issued at ``t`` for horizon ``h`` resolves only at ``t + h``.
    The previous implementation updated ``alpha_t`` on the row's own outcome
    immediately, so an outcome that was still unknown leaked into forecasts
    issued before it matured. Here a small maturity queue holds pending
    forecasts; at issue date ``t`` we first resolve every pending forecast with
    maturity ``<= t``, then issue the current interval. ``alpha_t`` is updated
    with ``alpha_t + gamma * (alpha - err)`` only when a forecast matures.
    P50 is left unchanged. The returned state warm-starts the next split
    causally and carries any still-unmatured forecasts across the boundary.
    """
    pred_idx = pred.with_row_index("_row")
    j = attach_truth(pred_idx, source, h)
    state = dict(state) if state else {}
    adj: dict[int, tuple[float, float, float]] = {}

    d_sorted = j.sort(["target_class", "date"])
    for cls_frame in d_sorted.partition_by("target_class", maintain_order=True):
        cls = str(cls_frame["target_class"][0])
        st = state.get(cls)
        if isinstance(st, dict):
            a = float(st["alpha"])
            scores = list(st["scores"])
            pending = list(st.get("pending", []))
        else:
            a, scores, pending = alpha, [], []
        for r in cls_frame.iter_rows(named=True):
            issue_date = r["date"]

            # Resolve forecasts whose outcome is observable at this issue date.
            still_pending: list[dict[str, object]] = []
            for item in pending:
                maturity = item["maturity"]
                if maturity is not None and maturity <= issue_date:
                    y_i = float(item["y"])  # type: ignore[arg-type]
                    err = 1.0 if (y_i < float(item["adj_p10"]) or y_i > float(item["adj_p90"])) else 0.0
                    scores.append(
                        max(float(item["base_p10"]) - y_i, y_i - float(item["base_p90"]))
                    )
                    if len(scores) > score_buffer:
                        scores.pop(0)
                    a = min(max(a + gamma * (alpha - err), 0.001), 0.5)
                else:
                    still_pending.append(item)
            pending = still_pending

            y = float(r["y"])
            p10, p50, p90 = float(r["p_0.1"]), float(r["p_0.5"]), float(r["p_0.9"])
            w = float(np.quantile(scores, 1.0 - a)) if len(scores) >= warmup else 0.0
            new_p10 = p10 - w
            new_p90 = p90 + w
            adj[int(r["_row"])] = (new_p10, p50, new_p90)
            pending.append({
                "maturity": issue_date + timedelta(days=h),
                "base_p10": p10,
                "base_p90": p90,
                "adj_p10": new_p10,
                "adj_p90": new_p90,
                "y": y,
            })
        state[cls] = {
            "alpha": a,
            "scores": scores,
            "pending": pending,
            "gamma": gamma,
            "warmup": warmup,
            "score_buffer": score_buffer,
        }

    out = pred_idx.with_columns(
        pl.col("_row").replace_strict({k: v[0] for k, v in adj.items()}, default=None).alias("p_0.1"),
        pl.col("_row").replace_strict({k: v[1] for k, v in adj.items()}, default=None).alias("p_0.5"),
        pl.col("_row").replace_strict({k: v[2] for k, v in adj.items()}, default=None).alias("p_0.9"),
    ).drop("_row")
    return _order(out), state
