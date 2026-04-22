import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests import utils


def test_round_and_hash_float_array():
    decimals = 7
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    h = utils.round_and_hash_float_array(arr, decimals=decimals)
    assert h == 1068792616613

    delta = 10**-decimals
    arr_test = arr + 0.49 * delta
    h2 = utils.round_and_hash_float_array(arr_test, decimals=decimals)
    assert h2 == h

    arr_test = arr + 0.51 * delta
    h3 = utils.round_and_hash_float_array(arr_test, decimals=decimals)
    assert h3 != h


@pytest.mark.parametrize("cal", ["julian", "proleptic_gregorian", "365_day", "366_day", "360_day"])
def test_assert_cftime_like_particlefile(tmp_path, cal):
    path = tmp_path / "test.parquet"
    attrs = {"units": "seconds since 2000-01-01 17:00:00", "calendar": cal}
    field = pa.field("time", pa.float64(), metadata={"attrs": json.dumps(attrs)})
    schema = pa.schema([field])
    table = pa.table({"time": pa.array([-20.0, 1.0])}, schema=schema)
    pq.write_table(table, path)

    utils.assert_cftime_like_particlefile(path)


def test_assert_cftime_like_particlefile_broken_parquet(tmp_path):
    path = tmp_path / "test.parquet"
    attrs = {"units": "broken-units", "calendar": "365_day"}
    field = pa.field("time", pa.float64(), metadata={"attrs": json.dumps(attrs)})
    schema = pa.schema([field])
    table = pa.table({"time": pa.array([-20.0, 1.0])}, schema=schema)
    pq.write_table(table, path)

    with pytest.raises(Exception, match="CF-time values in Parquet did not get properly decoded"):
        utils.assert_cftime_like_particlefile(path)
