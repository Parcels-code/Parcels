from __future__ import annotations

import xarray as xr
import zarr
import zarr.storage
from zarr.abc.store import Store


def _not_implemented(*args, **kwargs):
    raise NotImplementedError("This function it not implemented")


def open_raw_zarr(store: Store):
    """Open a Zarr dataset in an Xarray dataset, bypassing Dask."""
    with xr.open_zarr(store) as ds:
        var_to_dims = {name: var.dims for name, var in ds.variables.items()}
        coord_names = list(ds.coords)

    group = zarr.open(store, mode="r")
    assert isinstance(group, zarr.Group)

    data_vars = {}
    coords = {}
    for name, array in group.members():
        if not isinstance(array, zarr.Array):
            raise ValueError("Discovered a zarr.Group in the root group. open_raw_zarr doesn't work with nested groups")
        is_coord = name in coord_names

        if not is_coord:
            array.__array_function__ = _not_implemented  # trick xarray to prevent coersion to a numpy array

        var = xr.Variable(var_to_dims[name], array)

        if is_coord:
            coords[name] = var
        else:  # name is a data var
            data_vars[name] = var

    return xr.Dataset(data_vars, coords)
