from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import parcels

CF_CALENDARS = [
    ("360_day", "2000-01-01"),
    ("360_day", "2000-02-30"),
    ("noleap", "2000-01-01"),
    ("365_day", "2000-01-01"),
    ("all_leap", "2000-01-01"),
    ("366_day", "2000-01-01"),
    ("julian", "2000-01-01"),
    ("NoLeap", "2000-01-01"),
]


def _write_particlefile(path, attrs, values):
    schema = pa.schema([pa.field("t", pa.float64(), metadata=attrs), pa.field("particle_id", pa.int64())])
    table = pa.table(
        {"t": pa.array(values, type=pa.float64()), "particle_id": pa.array(range(len(values)), type=pa.int64())},
        schema=schema,
    )
    pq.write_table(table, path)


@pytest.mark.parametrize(("calendar", "origin"), CF_CALENDARS)
@pytest.mark.parametrize("values", [[0.0, 86400.0], []])
def test_read_particlefile_unsupported_calendar(tmp_parquet, calendar, origin, values):
    _write_particlefile(tmp_parquet, {"units": f"seconds since {origin}", "calendar": calendar}, values)
    with pytest.raises(NotImplementedError, match=calendar):
        parcels.read_particlefile(tmp_parquet)


@pytest.mark.parametrize(
    "attrs",
    [
        {"units": "seconds since 2000-02-30", "calendar": "360_day"},
        {"units": "seconds since 2000-01-01", "calendar": "noleap"},
        {"units": "seconds since 2000-01-01", "calendar": "unknown_calendar"},
        {"units": "seconds since NOT_A_DATE", "calendar": "standard"},
    ],
)
@pytest.mark.parametrize("values", [[0.0, 86400.0], []])
def test_read_particlefile_raw_times(tmp_parquet, attrs, values):
    _write_particlefile(tmp_parquet, attrs, values)
    df = parcels.read_particlefile(tmp_parquet, decode_times=False)
    assert isinstance(df, pl.DataFrame)
    assert df["t"].dtype == pl.Float64
    assert df["t"].to_list() == values
    assert df["particle_id"].to_list() == list(range(len(values)))


@pytest.mark.parametrize("calendar", ["standard", "gregorian", "proleptic_gregorian", "Standard", None])
@pytest.mark.parametrize("values", [[0.0, 86400.0], []])
def test_read_particlefile_supported_calendar(tmp_parquet, calendar, values):
    attrs = {"units": "seconds since 2000-01-01"}
    if calendar is not None:
        attrs["calendar"] = calendar
    _write_particlefile(tmp_parquet, attrs, values)
    df = parcels.read_particlefile(tmp_parquet)
    assert isinstance(df, pl.DataFrame)
    assert df["t"].dtype == pl.Datetime("ns")
    expected = [datetime(2000, 1, 1) + timedelta(seconds=value) for value in values]
    assert df["t"].to_list() == expected
    assert df["particle_id"].to_list() == list(range(len(values)))


@pytest.mark.parametrize("calendar", [None, "360_day"])
@pytest.mark.parametrize("decode_times", [True, False])
def test_read_particlefile_elapsed_seconds(tmp_parquet, calendar, decode_times):
    attrs = {"units": "seconds"}
    if calendar is not None:
        attrs["calendar"] = calendar
    _write_particlefile(tmp_parquet, attrs, [0.0, 2.0])
    df = parcels.read_particlefile(tmp_parquet, decode_times=decode_times)
    if decode_times:
        assert df["t"].dtype == pl.Duration("ns")
        np.testing.assert_array_equal(df["t"].to_numpy(), np.array([0, 2], dtype="timedelta64[s]"))
    else:
        assert df["t"].dtype == pl.Float64
        assert df["t"].to_list() == [0.0, 2.0]


@pytest.mark.parametrize(
    "attrs",
    [
        {"units": "seconds since 2000-01-01", "calendar": "unknown_calendar"},
        {"units": "seconds since NOT_A_DATE", "calendar": "standard"},
    ],
)
def test_read_particlefile_invalid_time_metadata(tmp_parquet, attrs):
    _write_particlefile(tmp_parquet, attrs, [0.0])
    with pytest.raises(ValueError, match="unable to decode time units"):
        parcels.read_particlefile(tmp_parquet)


@pytest.mark.parametrize("decode_times", [True, False])
def test_read_particlefile_missing_units(tmp_parquet, decode_times):
    _write_particlefile(tmp_parquet, {"calendar": "standard"}, [0.0])
    with pytest.raises(ValueError, match="Could not find 'units'"):
        parcels.read_particlefile(tmp_parquet, decode_times=decode_times)
