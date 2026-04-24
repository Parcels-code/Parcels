from __future__ import annotations

from datetime import datetime

import numpy as np
from cftime import datetime as cftime_datetime
from hypothesis import strategies as st

from parcels._core.utils.time import (
    TimeInterval,
)

calendar_strategy = st.sampled_from(
    [
        "gregorian",
        "proleptic_gregorian",
        "365_day",
        "360_day",
        "julian",
        "366_day",
        np.datetime64,
        datetime,
        np.timedelta64,
    ]
)


@st.composite
def np_timedelta64_strategy(draw):
    """Strategy for generating np.timedelta64 objects."""
    return np.timedelta64(draw(st.integers(1, 60 * 60 * 24 * 100 * 365)), "s")


@st.composite
def datetime_strategy(draw, calendar=None):
    if calendar is None:
        calendar = draw(calendar_strategy)
    if calendar is np.timedelta64:
        return draw(np_timedelta64_strategy())

    year = draw(st.integers(1900, 2100))
    month = draw(st.integers(1, 12))
    day = draw(st.integers(1, 28))
    if calendar is datetime:
        return datetime(year, month, day)
    if calendar is np.datetime64:
        return np.datetime64(datetime(year, month, day))

    return cftime_datetime(year, month, day, calendar=calendar)


@st.composite
def time_interval_strategy(draw, left=None, calendar=None):
    if left is None:
        left = draw(datetime_strategy(calendar=calendar))
    right = left + draw(np_timedelta64_strategy())

    return TimeInterval(left, right)
