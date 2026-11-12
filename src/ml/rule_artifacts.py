"""Export learned Sugeno memberships and consequents as paper artifacts.

Writes ``src/data/rule_artifacts.md`` and regenerates the membership/surface
figures in ``docs/figures`` for each saved horizon. If the learned terms are not
distinguishable (nearly identical centres), the table makes that visible rather
than hiding it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np

from ml.fuzzy_calibration import load_params, plot_fuzzy

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA: Final[Path] = REPO_ROOT / "src" / "data"
HORIZONS: Final[tuple[int, ...]] = (1, 7, 30, 90)


def _fmt(values: object) -> str:
    return ", ".join(f"{float(v):+.3f}" for v in values)  # type: ignore[union-attr]


def _mfs(spec: dict[str, object]) -> list[str]:
    centres = np.asarray(spec["centers"], dtype=float)
    widths = np.asarray(spec["widths"], dtype=float)
    return [
        f"({centres[i, k]:+.3f}, {widths[i, k]:.3f})"
        for i in range(centres.shape[0])
        for k in range(centres.shape[1])
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    lines = [
        "# Learned Sugeno rule artifacts",
        "",
        "Generated from the saved parameter JSON. Each row lists one input's",
        "Gaussian membership functions as (centre, width) pairs, then the constant",
        "rule consequents in row-major order.",
        "",
        "| h | system | inputs | membership functions | consequents |",
        "|---|---|---|---|---|",
    ]
    for h in HORIZONS:
        try:
            params = load_params(f"lstm_fuzzy_h{h}")
        except FileNotFoundError:
            LOGGER.warning("missing params for h=%d", h)
            continue
        for system in ("width", "loc"):
            spec = params[system]  # type: ignore[index]
            inputs = ", ".join(spec["inputs"])  # type: ignore[arg-type]
            mfs = " ".join(_mfs(spec))
            cons = _fmt(spec["consequents"])
            lines.append(f"| {h} | {system} | {inputs} | {mfs} | {cons} |")
        try:
            paths = plot_fuzzy(params, f"h{h}")
            LOGGER.info("h=%d figures: %s", h, [p.name for p in paths])
        except Exception as exc:  # noqa: BLE001 - plotting is best-effort
            LOGGER.warning("h=%d plot failed: %s", h, exc)

    lines += [
        "",
        "## Honest reading",
        "",
        "Overlapping Gaussian memberships mean individual rules should not be read as",
        "isolated if-then statements. At long horizons the width memberships can be",
        "nearly identical; when they are, the layer is effectively a constant width",
        "adjustment and the table says so.",
        "",
    ]
    out = DATA / "rule_artifacts.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("wrote %s", out)


if __name__ == "__main__":
    main()
