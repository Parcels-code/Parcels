"""Tests for the transparent rolling time-window cache (WindowedArray)."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from parcels import FieldSet, ParticleSet
from parcels._core._windowed_array import WindowedArray, maybe_windowed
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.kernels import AdvectionRK2


def test_to_windowed_arrays_is_idempotent_and_forwards_max_levels():
    ds = simple_UV_dataset(mesh="flat")
    fs = FieldSet.from_sgrid_conventions(ds.chunk({"time": 1}), mesh="flat")

    fs.to_windowed_arrays(max_levels=3)
    first = fs.U.data
    assert isinstance(first, WindowedArray)
    assert first._max == 3

    # re-wrapping returns the same object (idempotent), not a nested wrapper
    fs.to_windowed_arrays(max_levels=3)
    assert fs.U.data is first

@pytest.mark.parametrize("mesh", ["flat", "spherical"])
def test_dask_advection_matches_numpy(mesh):
    """An identical advection must give identical trajectories whether the field
    is numpy-backed or dask-backed (windowed)."""
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0  # steady zonal flow -> in-bounds, deterministic

    def run(windowed):
        d = ds.chunk({"time": 1}) if chunked else ds
        fs = FieldSet.from_sgrid_conventions(d, mesh=mesh)
        if windowed:
            fs.to_windowed_arrays()
        pset = ParticleSet(fs, lon=np.zeros(10), lat=np.linspace(-10, 10, 10))
        pset.execute(AdvectionRK2, runtime=7200, dt=np.timedelta64(15, "m"))
        return np.array(pset.lon), np.array(pset.lat)

    lon_np, lat_np = run(False)
    lon_dk, lat_dk = run(True)
    np.testing.assert_allclose(lon_dk, lon_np, atol=1e-9)
    np.testing.assert_allclose(lat_dk, lat_np, atol=1e-9)
