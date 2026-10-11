"""Subprocess runner for the project's command-line pipeline steps.

Only a small allow-list of modules can be launched, and logs stream to the UI
line by line so long runs stay observable. Nothing here re-implements training;
it shells out to the same entry points used on the command line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from lib.paths import DATA, REPO_ROOT


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    args: tuple[str, ...]
    description: str
    heavy: bool = False
    outputs: tuple[Path, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)


def _out(*names: str) -> tuple[Path, ...]:
    return tuple(DATA / n for n in names)


STEPS: dict[str, Step] = {
    "main_table": Step(
        key="main_table",
        label="Rebuild main results table",
        args=("-m", "ml.main_table"),
        description=(
            "Reads baseline_metrics_common.csv and rewrites main_table.csv/md "
            "(common-support table + h=90 per-class breakdown)."
        ),
        outputs=_out("main_table.csv", "main_table.md"),
    ),
    "diagnostics": Step(
        key="diagnostics",
        label="Rebuild interval diagnostics",
        args=("-m", "ml.interval_diagnostics"),
        description=(
            "Coverage, width, quantile crossings and per-quantile pinball for the "
            "base LSTM and random-walk baselines on train/valid."
        ),
        outputs=_out("interval_diagnostics.csv", "interval_diagnostics_common.csv"),
    ),
    "robustness": Step(
        key="robustness",
        label="Rebuild robustness comparison",
        args=("-m", "ml.robustness_compare"),
        description="Compares the main run against the 7-day congestion-lag rebuild on valid.",
        outputs=_out("robustness_metrics.csv", "robustness_summary.md"),
    ),
    "export": Step(
        key="export",
        label="Export XGBoost inference models",
        args=("-m", "ml.export"),
        description="Writes src/data/models/xgb_h*.ubj plus feature lists used by the playground.",
        outputs=_out("models"),
    ),
    "baselines_lstm": Step(
        key="baselines_lstm",
        label="Re-run baselines: LSTM + calibrators only",
        args=("-m", "ml.baselines"),
        description=(
            "Trains/caches the LSTM ensemble and scores lstm, lstm_fuzzy, cqr, aci, "
            "const_shift, rw, rw_emp, ar1. Skips XGBoost/LightGBM. Writes "
            "baseline_metrics.csv and baseline_dm.csv. This is the slow step."
        ),
        heavy=True,
        outputs=_out("baseline_metrics.csv", "baseline_dm.csv", "baseline_summary.md"),
        env={"BASELINES_MODELS": "lstm,lstm_fuzzy,cqr,aci,const_shift,rw,rw_emp,ar1"},
    ),
    "baselines_test": Step(
        key="baselines_test",
        label="Re-run baselines INCLUDING the frozen test split",
        args=("-m", "ml.baselines"),
        description=(
            "Same as above but also scores the frozen test split. Enabling this reads "
            "samples_test.parquet through the project guard; it is the one-and-only "
            "final evaluation and should not be used for tuning."
        ),
        heavy=True,
        outputs=_out(
            "baseline_metrics.csv", "baseline_metrics_common.csv", "baseline_dm.csv"
        ),
        env={
            "BASELINES_MODELS": "lstm,lstm_fuzzy,cqr,aci,const_shift,rw,rw_emp,ar1",
            "BASELINES_INCLUDE_TEST": "1",
        },
    ),
}


def build_env(step: Step, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"
    env.update(step.env)
    if extra:
        env.update(extra)
    return env


def stream(
    step: Step, extra_env: dict[str, str] | None = None
) -> Iterator[tuple[str, int | None]]:
    """Yield ``(line, None)`` then a final ``("", returncode)``."""
    cmd = [sys.executable, *step.args]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=build_env(step, extra_env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    yield f"$ {' '.join(cmd)}", None
    for line in proc.stdout:
        yield line.rstrip("\n"), None
    proc.wait()
    yield "", proc.returncode


def tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return f"(missing) {path}"
    if path.is_dir():
        listing = sorted(p.name for p in path.iterdir())
        return f"{path}\n" + "\n".join(listing[:n])
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:  # noqa: BLE001
        return f"(could not read {path}: {exc})"
    return "\n".join(lines[-n:])
