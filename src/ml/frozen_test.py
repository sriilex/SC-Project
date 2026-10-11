"""Guard against untracked reads of the frozen test split.

Why this exists
----------------
``samples_test.parquet`` exists to answer one question once: how well the
finished pipeline actually performs. Every time it is read for anything else --
comparing a candidate model, tuning a hyperparameter, eyeballing "does this look
right" -- that answer gets less honest, because the choice made in response to
what was seen becomes implicitly fitted to the test data.

The project rule is "touch test once, at the end, ever". Nothing enforced it.
This module provides the one sanctioned entry point, :func:`load_frozen_test`,
which raises :class:`FrozenTestAccessError` unless the call happens inside an
explicit, reasoned :func:`allow_test_set_access` block.

This is a runtime speed bump, not a cryptographic lock -- the goal is to make
"touching test outside the final report" require a conscious, greppable,
reviewable decision.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import polars as pl

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

_state = threading.local()


class FrozenTestAccessError(RuntimeError):
    """Raised when test-split access is attempted outside an allowed block."""


def _is_allowed() -> bool:
    return getattr(_state, "allowed", False)


@contextmanager
def allow_test_set_access(reason: str) -> Iterator[None]:
    """The only sanctioned way to read the frozen test split.

    Nests safely: an inner block does not revoke access for an outer caller that
    already granted it. Each grant is logged at WARNING so it shows up in
    ordinary run output.
    """
    if not reason or not reason.strip():
        raise ValueError(
            "allow_test_set_access requires a non-empty reason describing why the "
            "frozen test split is being read right now."
        )
    already_allowed = _is_allowed()
    _state.allowed = True
    if not already_allowed:
        LOGGER.warning(f"frozen test set access granted: {reason!r}")
    try:
        yield
    finally:
        _state.allowed = already_allowed


def load_frozen_test() -> pl.DataFrame:
    """Load the test split. Only callable inside an ``allow_test_set_access`` block."""
    if not _is_allowed():
        raise FrozenTestAccessError(
            "Reading the frozen test split outside an allow_test_set_access(...) "
            "block. Tune hyperparameters and select models on `valid`; the test "
            "split is read once, at the very end, for the final report. Wrap "
            "that one call site in:\n"
            "    with allow_test_set_access(\"why\"):\n"
            "        test = load_frozen_test()"
        )
    from ml.baselines import load_split

    return load_split("test")
