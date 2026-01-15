"""
This module provide a series of functions model outputs (which might be following their
own conventions) to metadata rich data (i.e., data following SGRID/UGRID as well as CF
conventions). Parcels needs this metadata rich data to discover grid geometries among other
things.

These functions use knowledge about the model to attach any missing metadata. The functions
emit verbose messaging so that the user is kept in the loop. The returned output is an
Xarray dataset so that users can further provide any missing metadata that was unable to
be determined before they pass it to the FieldSet constructor.
"""

from __future__ import annotations

import typing

import numpy as np
import xarray as xr

from parcels._core.utils import sgrid
from parcels._logger import logger

if typing.TYPE_CHECKING:
    import uxarray as ux

_NEMO_DIMENSION_COORD_NAMES = ["x", "y", "time", "x", "x_center", "y", "y_center", "depth", "glamf", "gphif"]

_NEMO_AXIS_VARNAMES = {
    "x": "X",
    "x_center": "X",
    "y": "Y",
    "y_center": "Y",
    "depth": "Z",
    "time": "T",
}

_NEMO_VARNAMES_MAPPING = {
    "time_counter": "time",
    "depthw": "depth",
    "uo": "U",
    "vo": "V",
    "wo": "W",
}

_COPERNICUS_MARINE_AXIS_VARNAMES = {
    "X": "lon",
    "Y": "lat",
    "Z": "depth",
    "T": "time",
}


def _maybe_bring_UV_depths_to_depth(ds):
    if "U" in ds.variables and "depthu" in ds.U.coords and "depth" in ds.coords:
        ds["U"] = ds["U"].assign_coords(depthu=ds["depth"].values).rename({"depthu": "depth"})
    if "V" in ds.variables and "depthv" in ds.V.coords and "depth" in ds.coords:
        ds["V"] = ds["V"].assign_coords(depthv=ds["depth"].values).rename({"depthv": "depth"})
    return ds


def _maybe_create_depth_dim(ds):
    if "depth" not in ds.dims:
        ds = ds.expand_dims({"depth": [0]})
        ds["depth"] = xr.DataArray([0], dims=["depth"])
    return ds


def _maybe_rename_coords(ds, axis_varnames):
    try:
        for axis, [coord] in ds.cf.axes.items():
            ds = ds.rename({coord: axis_varnames[axis]})
    except ValueError as e:
        raise ValueError(f"Multiple coordinates found on axis '{axis}'. Check your DataSet.") from e
    return ds


def _maybe_rename_variables(ds, varnames_mapping):
    rename_dict = {old: new for old, new in varnames_mapping.items() if (old in ds.data_vars) or (old in ds.coords)}
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds


def _assign_dims_as_coords(ds, dimension_names):
    for axis in dimension_names:
        if axis in ds.dims and axis not in ds.coords:
            ds = ds.assign_coords({axis: np.arange(ds.sizes[axis])})
    return ds


def _drop_unused_dimensions_and_coords(ds, dimension_and_coord_names):
    for dim in ds.dims:
        if dim not in dimension_and_coord_names:
            ds = ds.drop_dims(dim, errors="ignore")
    for coord in ds.coords:
        if coord not in dimension_and_coord_names:
            ds = ds.drop_vars(coord, errors="ignore")
    return ds


def _set_coords(ds, dimension_names):
    for varname in dimension_names:
        if varname in ds and varname not in ds.coords:
            ds = ds.set_coords([varname])
    return ds


def _maybe_remove_depth_from_lonlat(ds):
    for coord in ["glamf", "gphif"]:
        if coord in ds.coords and "depth" in ds[coord].dims:
            ds[coord] = ds[coord].squeeze("depth", drop=True)
    return ds


def _set_axis_attrs(ds, dim_axis):
    for dim, axis in dim_axis.items():
        ds[dim].attrs["axis"] = axis
    return ds


def _ds_rename_using_standard_names(ds: xr.Dataset | ux.UxDataset, name_dict: dict[str, str]) -> xr.Dataset:
    for standard_name, rename_to in name_dict.items():
        name = ds.cf[standard_name].name
        ds = ds.rename({name: rename_to})
        logger.info(
            f"cf_xarray found variable {name!r} with CF standard name {standard_name!r} in dataset, renamed it to {rename_to!r} for Parcels simulation."
        )
    return ds


# TODO is this function still needed, now that we require users to provide field names explicitly?
def _discover_U_and_V(ds: xr.Dataset, cf_standard_names_fallbacks) -> xr.Dataset:
    # Assumes that the dataset has U and V data

    if "W" not in ds:
        for cf_standard_name_W in cf_standard_names_fallbacks["W"]:
            if cf_standard_name_W in ds.cf.standard_names:
                ds = _ds_rename_using_standard_names(ds, {cf_standard_name_W: "W"})
                break

    if "U" in ds and "V" in ds:
        return ds  # U and V already present
    elif "U" in ds or "V" in ds:
        raise ValueError(
            "Dataset has only one of the two variables 'U' and 'V'. Please rename the appropriate variable in your dataset to have both 'U' and 'V' for Parcels simulation."
        )

    for cf_standard_name_U, cf_standard_name_V in cf_standard_names_fallbacks["UV"]:
        if cf_standard_name_U in ds.cf.standard_names:
            if cf_standard_name_V not in ds.cf.standard_names:
                raise ValueError(
                    f"Dataset has variable with CF standard name {cf_standard_name_U!r}, "
                    f"but not the matching variable with CF standard name {cf_standard_name_V!r}. "
                    "Please rename the appropriate variables in your dataset to have both 'U' and 'V' for Parcels simulation."
                )
        else:
            continue

        ds = _ds_rename_using_standard_names(ds, {cf_standard_name_U: "U", cf_standard_name_V: "V"})
        break
    return ds


def nemo_to_sgrid(*, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset):
    # TODO: Update docstring
    """Create a FieldSet from a xarray.Dataset from NEMO netcdf files.

    Parameters
    ----------
    ds : xarray.Dataset
        xarray.Dataset as obtained from a set of NEMO netcdf files.

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    The NEMO model (https://www.nemo-ocean.eu/) is used by a variety of oceanographic institutions around the world.
    Output from these models may differ subtly in terms of variable names and metadata conventions.
    This function attempts to standardize these differences to create a Parcels FieldSet.
    If you encounter issues with your specific NEMO dataset, please open an issue on the Parcels GitHub repository with details about your dataset.

    """
    fields = fields.copy()
    coords = coords[["gphif", "glamf"]]

    for name, field_da in fields.items():
        if isinstance(field_da, xr.Dataset):
            field_da = field_da[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            field_da = field_da.rename(name)

        match name:
            case "U":
                field_da = field_da.rename({"y": "y_center"})
            case "V":
                field_da = field_da.rename({"x": "x_center"})
            case _:
                pass
        field_da = field_da.reset_coords(drop=True)
        fields[name] = field_da

    if "time" in coords.dims:
        if coords.sizes["time"] != 1:
            raise ValueError("Time dimension in coords must be length 1 (i.e., no time-varying grid).")
        coords = coords.isel(time=0).drop("time")
    if len(coords.dims) == 3:
        for dim, len_ in coords.sizes.items():
            if len_ == 1:
                # TODO: log statement about selecting along z dim of 1
                coords = coords.isel({dim: 0})
    if len(coords.dims) != 2:
        raise ValueError("Expected coordsinates to be 2 dimensional")

    ds = xr.merge(list(fields.values()) + [coords])
    ds = _maybe_rename_variables(ds, _NEMO_VARNAMES_MAPPING)
    ds = _maybe_create_depth_dim(ds)
    ds = _maybe_bring_UV_depths_to_depth(ds)
    ds = _drop_unused_dimensions_and_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _assign_dims_as_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _set_coords(ds, _NEMO_DIMENSION_COORD_NAMES)
    ds = _maybe_remove_depth_from_lonlat(ds)
    ds = _set_axis_attrs(ds, _NEMO_AXIS_VARNAMES)

    expected_axes = set("XYZT")  # TODO: Update after we have support for 2D spatial fields
    if missing_axes := (expected_axes - set(ds.cf.axes)):
        raise ValueError(
            f"Dataset missing CF compliant metadata for axes "
            f"{missing_axes}. Expected 'axis' attribute to be set "
            f"on all dimension axes {expected_axes}. "
            "HINT: Add xarray metadata attribute 'axis' to dimension - e.g., ds['lat'].attrs['axis'] = 'Y'"
        )

    if "W" in ds.data_vars:
        # Negate W to convert from up positive to down positive (as that's the direction of positive z)
        ds["W"].data *= -1
    if "grid" in ds.cf.cf_roles:
        raise ValueError(
            "Dataset already has a 'grid' variable (according to cf_roles). Didn't expect there to be grid metadata on copernicusmarine datasets - please open an issue with more information about your dataset."
        )

    ds["grid"] = xr.DataArray(
        0,
        attrs=sgrid.Grid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("x", "y"),
            node_coordinates=("glamf", "gphif"),
            face_dimensions=(
                sgrid.DimDimPadding("x_center", "x", sgrid.Padding.LOW),
                sgrid.DimDimPadding("y_center", "y", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.DimDimPadding("z_center", "depth", sgrid.Padding.HIGH),),
        ).to_attrs(),
    )

    # NEMO models are always in degrees
    ds["glamf"].attrs["units"] = "degrees"
    ds["gphif"].attrs["units"] = "degrees"

    # Update to use lon and lat for internal naming
    ds = sgrid.rename(ds, {"gphif": "lat", "glamf": "lon"})  # TODO: Logging message about rename
    return ds


def copernicusmarine_to_sgrid(
    *, fields: dict[str, xr.Dataset | xr.DataArray], coords: xr.Dataset | None = None
) -> xr.Dataset:
    """Create an sgrid-compliant xarray.Dataset from a dataset of Copernicus Marine netcdf files.

    Parameters
    ----------
    fields : dict[str, xr.Dataset | xr.DataArray]
        Dictionary of xarray.DataArray objects as obtained from a set of Copernicus Marine netcdf files.
    coords : xarray.Dataset, optional
        xarray.Dataset containing coordinate variables. By default these are time, depth, latitude, longitude

    Returns
    -------
    xarray.Dataset
        Dataset object following SGRID conventions to be (optionally) modified and passed to a FieldSet constructor.

    Notes
    -----
    See https://help.marine.copernicus.eu/en/collections/9080063-copernicus-marine-toolbox for more information on the copernicusmarine toolbox.
    The toolbox to ingest data from most of the products on the Copernicus Marine Service (https://data.marine.copernicus.eu/products) into an xarray.Dataset.
    You can use indexing and slicing to select a subset of the data before passing it to this function.

    """
    fields = fields.copy()

    for name, field_da in fields.items():
        if isinstance(field_da, xr.Dataset):
            field_da = field_da[name]
            # TODO: logging message, warn if multiple fields are in this dataset
        else:
            field_da = field_da.rename(name)
        fields[name] = field_da

    ds = xr.merge(list(fields.values()) + ([coords] if coords is not None else []))
    ds.attrs.clear()  # Clear global attributes from the merging

    ds = _maybe_rename_coords(ds, _COPERNICUS_MARINE_AXIS_VARNAMES)
    if "W" in ds.data_vars:
        # Negate W to convert from up positive to down positive (as that's the direction of positive z)
        ds["W"].data *= -1

    if "grid" in ds.cf.cf_roles:
        raise ValueError(
            "Dataset already has a 'grid' variable (according to cf_roles). Didn't expect there to be grid metadata on copernicusmarine datasets - please open an issue with more information about your dataset."
        )
    ds["grid"] = xr.DataArray(
        0,
        attrs=sgrid.Grid2DMetadata(  # use dummy *_center dimensions - this is A grid data (all defined on nodes)
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.DimDimPadding("x_center", "lon", sgrid.Padding.LOW),
                sgrid.DimDimPadding("y_center", "lat", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.DimDimPadding("z_center", "depth", sgrid.Padding.LOW),),
        ).to_attrs(),
    )

    return ds
