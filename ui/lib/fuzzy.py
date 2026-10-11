"""Sugeno FIS inspection and editing for the UI.

Thin wrappers around ``ml.fuzzy_calibration`` so that whatever the UI shows is
exactly what the pipeline computes. The evaluation core stays numpy-only.
"""

from __future__ import annotations

import copy
import itertools

import numpy as np
import polars as pl

from ml.fuzzy_calibration import _adjust_numpy, _fis_numpy, _norm_inputs

LINGUISTIC: tuple[str, ...] = ("low", "medium", "high")

INPUT_LABELS: dict[str, str] = {
    "vol_z": "volatility (z)",
    "cov_gap": "coverage gap",
    "bias_z": "median bias (z)",
}

SYSTEM_LABELS: dict[str, str] = {
    "width": "Width system  f_w -> log-multiplier m",
    "loc": "Location system  f_l -> shift s",
}


def n_inputs(spec: dict) -> int:
    return len(spec.get("inputs") or [])


def rule_index(spec: dict) -> list[tuple[int, ...]]:
    n = n_inputs(spec)
    return list(itertools.product(range(int(spec.get("n_mfs", 3))), repeat=n))


def evaluate(spec: dict, x_raw: np.ndarray) -> np.ndarray:
    """Evaluate a FIS on raw (unstandardised) inputs."""
    if not spec.get("inputs"):
        return np.zeros(np.atleast_2d(x_raw).shape[0])
    return _fis_numpy(spec, _norm_inputs(np.atleast_2d(x_raw), spec))


def rule_weights(spec: dict, x_raw: np.ndarray) -> np.ndarray:
    """Firing strength of every rule for a single raw input vector."""
    centers = np.asarray(spec["centers"], dtype=float)
    widths = np.asarray(spec["widths"], dtype=float)
    x = np.atleast_2d(_norm_inputs(np.atleast_2d(x_raw), spec))
    d = (x[:, :, None] - centers[None, :, :]) / (widths[None, :, :] + 1e-9)
    mu = np.exp(-0.5 * d * d)
    weights = np.ones((x.shape[0], len(rule_index(spec))))
    for i, idx in enumerate(rule_index(spec)):
        for axis, mf in enumerate(idx):
            weights[:, i] *= mu[:, axis, mf]
    return weights


def memberships(spec: dict, x_raw: np.ndarray) -> np.ndarray:
    """Membership degree per input/term, shape ``(n, n_inputs, n_mfs)``."""
    centers = np.asarray(spec["centers"], dtype=float)
    widths = np.asarray(spec["widths"], dtype=float)
    x = np.atleast_2d(_norm_inputs(np.atleast_2d(x_raw), spec))
    d = (x[:, :, None] - centers[None, :, :]) / (widths[None, :, :] + 1e-9)
    return np.exp(-0.5 * d * d)


def membership_curve(spec: dict, axis: int, grid: np.ndarray) -> np.ndarray:
    """Membership curves for one input over a *standardised* grid."""
    centers = np.asarray(spec["centers"], dtype=float)[axis]
    widths = np.asarray(spec["widths"], dtype=float)[axis]
    d = (grid[:, None] - centers[None, :]) / (widths[None, :] + 1e-9)
    return np.exp(-0.5 * d * d)


def input_grid(
    spec: dict, axis: int, lo: float = -3.0, hi: float = 3.0, n: int = 200
) -> np.ndarray:
    """Standardised grid for plotting; optionally cover the observed range."""
    return np.linspace(lo, hi, n)


def describe_rules(spec: dict) -> pl.DataFrame:
    """Human-readable antecedent/consequent table for the rule editor."""
    rows = []
    cons = list(np.asarray(spec.get("consequents", []), dtype=float).reshape(-1))
    for i, idx in enumerate(rule_index(spec)):
        antecedents = [
            f"{INPUT_LABELS.get(name, name)} = {LINGUISTIC[mi]}"
            for name, mi in zip(spec.get("inputs") or [], idx)
        ]
        rows.append(
            {
                "rule": f"R{i + 1}",
                "if": " and ".join(antecedents) if antecedents else "(no input)",
                "consequent": cons[i] if i < len(cons) else 0.0,
            }
        )
    return pl.DataFrame(rows)


def set_consequents(spec: dict, values: list[float]) -> dict:
    out = copy.deepcopy(spec)
    out["consequents"] = [float(v) for v in values]
    return out


def set_terms(spec: dict, axis: int, centers: list[float], widths: list[float]) -> dict:
    out = copy.deepcopy(spec)
    centers_all = [list(map(float, row)) for row in out["centers"]]
    widths_all = [list(map(float, row)) for row in out["widths"]]
    centers_all[axis] = [float(c) for c in centers]
    widths_all[axis] = [max(float(w), 1e-3) for w in widths]
    out["centers"] = centers_all
    out["widths"] = widths_all
    return out


def surface(
    spec: dict, axis_x: int, axis_y: int | None = None, n: int = 61
) -> tuple[np.ndarray, ...]:
    """Output surface over standardised grids (2-D) or a curve (1-D)."""
    if not spec.get("inputs"):
        grid = np.linspace(-3, 3, n)
        return grid, np.zeros(n)
    if n_inputs(spec) == 1 or axis_y is None:
        grid_std = np.linspace(-3, 3, n)
        # _fis_numpy expects raw; invert the stored normalisation.
        mean = float(np.asarray(spec["norm_mean"], dtype=float)[0])
        std = float(np.asarray(spec["norm_std"], dtype=float)[0])
        x = (grid_std * std + mean).reshape(-1, 1)
        z = _fis_numpy(spec, _norm_inputs(x, spec))
        return grid_std, z
    grid = np.linspace(-3, 3, n)
    xx, yy = np.meshgrid(grid, grid)
    mean = np.asarray(spec["norm_mean"], dtype=float)
    std = np.asarray(spec["norm_std"], dtype=float)
    raw = np.stack(
        [
            xx.ravel() * std[axis_x] + mean[axis_x],
            yy.ravel() * std[axis_y] + mean[axis_y],
        ],
        axis=1,
    )
    z = _fis_numpy(spec, _norm_inputs(raw, spec)).reshape(xx.shape)
    return xx, yy, z


def apply_adjustment(base: pl.DataFrame, m: np.ndarray, s: np.ndarray) -> pl.DataFrame:
    """Apply (m, s) to a prediction frame and return ordered quantiles."""
    arr = np.stack(
        [
            base["p_0.1"].to_numpy().astype(float),
            base["p_0.5"].to_numpy().astype(float),
            base["p_0.9"].to_numpy().astype(float),
        ],
        axis=1,
    )
    adj = _adjust_numpy(arr, np.asarray(m, dtype=float), np.asarray(s, dtype=float))
    return base.with_columns(
        pl.Series("p_0.1", adj[:, 0]),
        pl.Series("p_0.5", adj[:, 1]),
        pl.Series("p_0.9", adj[:, 2]),
    )


def outputs_for_frame(
    params: dict, h: int, inputs: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Width multiplier ``m`` and location shift ``s`` for every row of ``inputs``."""
    width_spec = params.get("width") or {}
    loc_spec = params.get("loc") or {}
    mapping = {
        "vol_z": inputs["vol_z"].to_numpy().astype(float),
        "cov_gap": inputs["cov_gap"].to_numpy().astype(float),
        "bias_z": inputs["bias_z"].to_numpy().astype(float),
    }

    def _x(spec: dict) -> np.ndarray:
        cols = list(spec.get("inputs") or [])
        if not cols:
            return np.zeros((inputs.height, 0))
        return np.stack([mapping[c] for c in cols], axis=1)

    m = (
        evaluate(width_spec, _x(width_spec))
        if width_spec.get("inputs")
        else np.zeros(inputs.height)
    )
    s = (
        evaluate(loc_spec, _x(loc_spec))
        if loc_spec.get("inputs")
        else np.zeros(inputs.height)
    )
    return m, s


def summarise(params: dict) -> pl.DataFrame:
    """One-row-per-system summary of learned parameters."""
    rows = []
    for system in ("width", "loc"):
        spec = params.get(system, {})
        if not spec:
            continue
        rows.append(
            {
                "system": system,
                "inputs": ", ".join(spec.get("inputs") or []) or "(none)",
                "n_rules": len(spec.get("consequents") or []),
                "output_scale": float(spec.get("output_scale", 0.0)),
                "max_abs_consequent": float(
                    np.max(
                        np.abs(np.asarray(spec.get("consequents", [0.0]), dtype=float))
                    )
                ),
            }
        )
    return pl.DataFrame(rows)
