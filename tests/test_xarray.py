import pytest
import xarray as xr

from parcels import open_raw_zarr
from parcels._datasets.structured.generic import datasets


@pytest.mark.parametrize("ds", [pytest.param(v, id=k) for k, v in datasets.items()])
def test_open_raw_zarr_roundtrip(ds, tmp_path):
    path = tmp_path / "ds.zarr"
    ds.to_zarr(path)

    result = open_raw_zarr(path)

    xr.testing.assert_identical(result.load(), ds)
