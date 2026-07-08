"""Tests for the transparent rolling time-window cache (WindowedArray)."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from parcels import FieldSet, ParticleSet
from parcels._core._windowed_array import WindowedArray, maybe_windowed
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.kernels import AdvectionRK2


def test_windowed_isel_matches_dask_loads_once_and_evicts():
    """WindowedArray.isel must equal dask isel, load each level once, keep <=2 resident."""
    ntime, n, npart = 20, 64, 200
    rng = np.random.default_rng(0)
    base = rng.standard_normal((ntime, 3, n, n))
    lazy = xr.DataArray(da.from_array(base, chunks=(1, 3, n, n)), dims=("time", "depth", "lat", "lon"))
    win = WindowedArray(lazy)

    worst, max_cache = 0.0, 0
    for step in range(40):
        ti = min(step // 2, ntime - 2)  # advancing clock, 2 sub-steps per level
        yi, xi = rng.integers(0, n, npart), rng.integers(0, n, npart)
        zi = np.zeros(npart, dtype=int)
        sel = dict(
            time=xr.DataArray(np.r_[np.full(npart, ti), np.full(npart, ti + 1)], dims="p"),
            depth=xr.DataArray(np.r_[zi, zi], dims="p"),
            lat=xr.DataArray(np.r_[yi, yi], dims="p"),
            lon=xr.DataArray(np.r_[xi, xi], dims="p"),
        )
        got = win.isel(sel).data
        ref = lazy.isel(sel).data.compute()
        worst = max(worst, float(np.abs(got - ref).max()))
        max_cache = max(max_cache, len(win._cache))

    assert worst == 0.0  # byte-identical to dask
    assert win.loads == ntime  # each time level read exactly once
    assert max_cache <= 2  # only the bracketing levels resident


def test_windowed_isel_backward_clock_loads_once_and_evicts():
    """A backward-running clock (dt < 0) must also load each level once and keep <=2
    resident: eviction is symmetric, so no integration-direction flag is needed.
    """
    ntime, n, npart = 20, 64, 200
    rng = np.random.default_rng(0)
    base = rng.standard_normal((ntime, 3, n, n))
    lazy = xr.DataArray(da.from_array(base, chunks=(1, 3, n, n)), dims=("time", "depth", "lat", "lon"))
    win = WindowedArray(lazy)

    worst, max_cache = 0.0, 0
    for step in range(40):
        ti = max(ntime - 2 - step // 2, 0)  # receding clock, 2 sub-steps per level
        yi, xi = rng.integers(0, n, npart), rng.integers(0, n, npart)
        zi = np.zeros(npart, dtype=int)
        sel = dict(
            time=xr.DataArray(np.r_[np.full(npart, ti), np.full(npart, ti + 1)], dims="p"),
            depth=xr.DataArray(np.r_[zi, zi], dims="p"),
            lat=xr.DataArray(np.r_[yi, yi], dims="p"),
            lon=xr.DataArray(np.r_[xi, xi], dims="p"),
        )
        got = win.isel(sel).data
        ref = lazy.isel(sel).data.compute()
        worst = max(worst, float(np.abs(got - ref).max()))
        max_cache = max(max_cache, len(win._cache))

    assert worst == 0.0  # byte-identical to dask
    assert win.loads == ntime  # each time level read exactly once, going backward
    assert max_cache <= 2  # only the bracketing levels resident


def test_to_windowed_arrays_wraps_dask_but_not_numpy():
    ds = simple_UV_dataset(mesh="flat")
    fs_np = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    fs_dk = FieldSet.from_sgrid_conventions(ds.chunk({"time": 1}), mesh="flat")

    # construction is never windowing -- it is opt-in via the fieldset method
    assert not isinstance(fs_np.U.data, WindowedArray)
    assert not isinstance(fs_dk.U.data, WindowedArray)

    assert fs_np.to_windowed_arrays() is fs_np  # chainable
    fs_dk.to_windowed_arrays()

    # numpy-backed field is left eager; dask-backed field gets wrapped
    assert not isinstance(fs_np.U.data, WindowedArray)
    assert isinstance(fs_dk.U.data, WindowedArray)
    # transparency: forwarded attributes still behave like the DataArray
    assert fs_dk.U.data.dims == fs_np.U.data.dims
    assert fs_dk.U.data.shape == fs_np.U.data.shape


def test_to_windowed_arrays_is_idempotent_and_forwards_max_levels():
    ds = simple_UV_dataset(mesh="flat")
    fs = FieldSet.from_sgrid_conventions(ds.chunk({"time": 1}), mesh="flat")

    fs.to_windowed_arrays(max_levels=3)
    first = fs.U.data
    assert isinstance(first, WindowedArray)
    assert first._max == 3

    # re-wrapping returns the same object (idempotent, warm cache preserved)
    fs.to_windowed_arrays(max_levels=3)
    assert fs.U.data is first


def test_maybe_windowed_passthrough_for_non_time_leading():
    da_no_time = xr.DataArray(da.zeros((3, 4), chunks=(3, 4)), dims=("lat", "lon"))
    assert maybe_windowed(da_no_time) is da_no_time  # not wrapped (no leading time dim)


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
@pytest.mark.parametrize("dt_minutes", [15, -15], ids=["forward", "backward"])
def test_dask_advection_matches_numpy(mesh, dt_minutes):
    """An identical advection must give identical trajectories whether the field
    is numpy-backed or dask-backed (windowed) -- for both forward (dt > 0) and
    backward (dt < 0) integration.
    """
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0  # steady zonal flow -> in-bounds, deterministic

    def run(chunked):
        d = ds.chunk({"time": 1}) if chunked else ds
        fs = FieldSet.from_sgrid_conventions(d, mesh=mesh)
        if chunked:
            fs.to_windowed_arrays()
        pset = ParticleSet(fs, lon=np.zeros(10), lat=np.linspace(-10, 10, 10))
        pset.execute(AdvectionRK2, runtime=7200, dt=np.timedelta64(dt_minutes, "m"))
        return np.array(pset.lon), np.array(pset.lat)

    lon_np, lat_np = run(False)
    lon_dk, lat_dk = run(True)
    np.testing.assert_allclose(lon_dk, lon_np, atol=1e-9)
    np.testing.assert_allclose(lat_dk, lat_np, atol=1e-9)
