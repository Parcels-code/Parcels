import tempfile
from pathlib import Path

import xarray as xr
from hypothesis import given

import parcels._strategies as pst
from parcels._compat_v3 import particlefile_to_v3_zarr


def assert_valid_v3_particlefile_structure(ds: xr.Dataset):
    for var in ["lat", "lon", "z", "time"]:
        assert var in ds.variables

    assert set(ds.dims) == {"obs", "trajectory"}
    assert set(ds.coords) == {"obs", "trajectory"}

    assert ds["lat"].attrs["axis"] == "Y"  # attrs are copied accross correctly


@given(buf=pst.particlefile_output())
def test_particlefile_to_v3_zarr(buf):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zarr = Path(tmpdir) / "output.zarr"

        particlefile_to_v3_zarr(from_parquet=buf, to_zarr=tmp_zarr)
        ds = xr.open_zarr(tmp_zarr)
        assert_valid_v3_particlefile_structure(ds)
