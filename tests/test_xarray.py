import pytest
import xarray as xr
import zarr

from parcels import open_raw_zarr
from parcels._datasets.structured.generic import datasets


@pytest.mark.parametrize("ds", [pytest.param(v, id=k) for k, v in datasets.items()])
def test_open_raw_zarr(ds, tmp_path):
    path = tmp_path / "ds.zarr"
    ds.to_zarr(path)

    result = open_raw_zarr(path)

    for k in result.data_vars:
        # tests that the internal representation within Xarray isn't coerced into a numpy array
        assert isinstance(result[k]._variable._data, zarr.Array)

    xr.testing.assert_identical(result.load(), ds)
