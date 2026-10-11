"""Tests for the frozen-test access guard (Task G)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml import baselines
from ml.frozen_test import (
    FrozenTestAccessError,
    allow_test_set_access,
    load_frozen_test,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_guard_blocks_and_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(baselines, "load_split", lambda name: pl.DataFrame({"name": [name]}))
    with pytest.raises(FrozenTestAccessError):
        load_frozen_test()
    with allow_test_set_access("guard test"):
        df = load_frozen_test()
    assert df["name"][0] == "test"
    with pytest.raises(FrozenTestAccessError):
        load_frozen_test()


def test_frozen_guard_requires_reason() -> None:
    with pytest.raises(ValueError), allow_test_set_access("   "):
        pass


def test_no_direct_test_load_outside_guard() -> None:
    """No module may call ``load_split("test")`` outside the guard module."""
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if path.name == "frozen_test.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if 'load_split("test")' in text or "load_split('test')" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"frozen test split read outside the guard: {offenders}"
