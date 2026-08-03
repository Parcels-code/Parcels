import pytest
import xarray as xr
import zarr

from parcels import open_raw_zarr
from parcels._datasets.structured.generic import datasets


@pytest.mark.filterwarnings("ignore:Consolidated metadata is currently not part in the Zarr format 3 specification")
@pytest.mark.parametrize("ds", [pytest.param(v, id=k) for k, v in datasets.items()])
def test_open_raw_zarr(ds: xr.Dataset, tmp_path):
    path = tmp_path / "ds.zarr"
    ds.to_zarr(path)

    result = open_raw_zarr(path)

    for k in result.data_vars:
        # tests that the internal representation within Xarray isn't coerced into a numpy array
        assert isinstance(result[k]._variable._data, zarr.Array)

    xr.testing.assert_identical(result.load(), ds)
