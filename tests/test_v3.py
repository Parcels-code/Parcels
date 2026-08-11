import tempfile
from pathlib import Path

import xarray as xr
from hypothesis import given

import parcels._strategies as pst
from parcels._v3 import particlefile_to_v3_zarr


def assert_valid_v3_particlefile_structure(ds: xr.Dataset):
    for var in ["lat", "lon", "depth", "time"]:
        assert var in ds.variables

    assert set(ds.dims) == {"obs", "trajectory"}


@given(df=pst.particlefile_output())
def test_particlefile_to_v3_zarr(df):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_parquet = Path(tmpdir) / "tmp.parquet"
        tmp_zarr = Path(tmpdir) / "output.zarr"

        df.to_parquet(tmp_parquet)

        particlefile_to_v3_zarr(tmp_parquet, tmp_zarr)
        ds = xr.open_zarr(tmp_zarr)
        assert_valid_v3_particlefile_structure(ds)
