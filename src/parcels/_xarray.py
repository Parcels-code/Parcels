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
        var_to_attrs = {name: var.attrs for name, var in ds.variables.items()}
        coords = {name: ds[name].variable.load() for name in ds.coords}
        ds_attrs = ds.attrs

    group = zarr.open(store, mode="r")
    assert isinstance(group, zarr.Group)

    data_vars = {}
    for name, array in group.members():
        if not isinstance(array, zarr.Array):
            raise ValueError("Discovered a zarr.Group in the root group. open_raw_zarr doesn't work with nested groups")
        if name in coords:
            continue

        # trick xarray to prevent coersion to a numpy array
        array.__array_function__ = _not_implemented  # type: ignore[attr-defined]
        data_vars[name] = xr.Variable(var_to_dims[name], array, attrs=var_to_attrs[name])

    return xr.Dataset(data_vars, coords, attrs=ds_attrs)
