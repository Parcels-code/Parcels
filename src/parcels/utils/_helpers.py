"""Internal helpers for Parcels."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

PACKAGE = "Parcels"


def timedelta_to_float(dt: float | timedelta | np.timedelta64) -> float:
    """Convert a timedelta to a float in seconds."""
    if isinstance(dt, timedelta):
        return dt.total_seconds()
    if isinstance(dt, np.timedelta64):
        return float(dt / np.timedelta64(1, "s"))
    return float(dt)
