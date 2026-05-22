from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import cf_xarray  # noqa: F401
import uxarray as ux
import xarray as xr
import xgcm

import parcels._sgrid as sgrid
from parcels._core.basegrid import BaseGrid
from parcels._core.field import Field, VectorField
from parcels._core.uxgrid import UxGrid
from parcels._core.xgrid import _DEFAULT_XGCM_KWARGS, XGrid
from parcels._logger import logger
from parcels._typing import Mesh
from parcels.interpolators import (
    CGrid_Velocity,
    XLinear,
    XLinear_Velocity,
)


class Model(ABC):
    data: Any
    grid: BaseGrid

    @abstractmethod
    def construct_fields(self) -> list[Field | VectorField]: ...


class StructuredModel(Model):
    def __init__(self, data: xr.Dataset, grid: XGrid):
        self.data = data
        self.grid = grid

    def construct_fields(self) -> list[Field | VectorField]:
        # Create fields from data variables, skipping grid metadata variables
        # Skip variables that are SGRID metadata (have cf_role='grid_topology')
        skip_vars = set()
        for var in self.data.data_vars:
            if self.data[var].attrs.get("cf_role") == "grid_topology":
                skip_vars.add(var)

        fields = {}
        if "U" in self.data.data_vars and "V" in self.data.data_vars:
            vector_interp_method = XLinear_Velocity if _is_agrid(self.data) else CGrid_Velocity
            fields["U"] = Field("U", self.data["U"], self.grid, XLinear)
            fields["V"] = Field("V", self.data["V"], self.grid, XLinear)
            fields["UV"] = VectorField("UV", fields["U"], fields["V"], vector_interp_method=vector_interp_method)

            if "W" in self.data.data_vars:
                fields["W"] = Field("W", self.data["W"], self.grid, XLinear)
                fields["UVW"] = VectorField(
                    "UVW", fields["U"], fields["V"], fields["W"], vector_interp_method=vector_interp_method
                )

        for varname in set(self.data.data_vars) - set(fields.keys()) - skip_vars:
            fields[varname] = Field(str(varname), self.data[varname], self.grid, XLinear)

        return list(fields.values())

    @classmethod
    def from_sgrid_conventions(cls, ds: xr.Dataset, mesh: Mesh | None = None) -> Self:
        ds = ds.copy()
        if mesh is None:
            mesh = _get_mesh_type_from_sgrid_dataset(ds)

        # Ensure time dimension has axis attribute if present
        if "time" in ds.dims and "time" in ds.coords:
            if "axis" not in ds["time"].attrs:
                logger.debug(
                    "Dataset contains 'time' dimension but no 'axis' attribute. Setting 'axis' attribute to 'T'."
                )
                ds["time"].attrs["axis"] = "T"

        # Find time dimension based on axis attribute and rename to `time`
        if (time_dims := ds.cf.axes.get("T")) is not None:
            if len(time_dims) > 1:
                raise ValueError("Multiple time coordinates found in dataset. This is not supported by Parcels.")
            (time_dim,) = time_dims
            if time_dim != "time":
                logger.debug(f"Renaming time axis coordinate from {time_dim} to 'time'.")
                ds = ds.rename({time_dim: "time"})

        # Parse SGRID metadata and get xgcm kwargs
        _, xgcm_kwargs = sgrid.xgcm_parse_sgrid(ds)

        # Add time axis to xgcm_kwargs if present
        if "time" in ds.dims:
            if "T" not in xgcm_kwargs["coords"]:
                xgcm_kwargs["coords"]["T"] = {"center": "time"}

        if "lon" not in ds.coords or "lat" not in ds.coords:
            node_dimensions = sgrid.load_mappings(ds.grid.node_dimensions)
            ds["lon"] = ds[node_dimensions[0]]
            ds["lat"] = ds[node_dimensions[1]]

        # Create xgcm Grid object
        xgcm_grid = xgcm.Grid(ds, autoparse_metadata=False, **xgcm_kwargs, **_DEFAULT_XGCM_KWARGS)

        # Wrap in XGrid
        grid = XGrid(xgcm_grid, mesh=mesh)
        return cls(ds, grid)


class UnstructuredModel(Model):
    def __init__(self, data: ux.UxDataset, grid: UxGrid):
        self.data = data
        self.grid = grid

    def construct_fields(self) -> list[Field | VectorField]: ...


# TODO: Refactor later into something like `parcels._metadata.discover(dataset)` helper that can be used to discover important metadata like this. I think this whole metadata handling should be refactored into its own module.
def _get_mesh_type_from_sgrid_dataset(ds_sgrid: xr.Dataset) -> Mesh:
    """Small helper to inspect SGRID metadata and dataset metadata to determine mesh type."""
    sgrid_metadata = ds_sgrid.sgrid.metadata

    fpoint_x, fpoint_y = sgrid_metadata.node_coordinates

    if _is_coordinate_in_degrees(ds_sgrid[fpoint_x]) ^ _is_coordinate_in_degrees(ds_sgrid[fpoint_x]):
        msg = (
            f"Mismatch in units between X and Y coordinates.\n"
            f"  Coordinate {ds_sgrid[fpoint_x]!r} attrs: {ds_sgrid[fpoint_x].attrs}\n"
            f"  Coordinate {ds_sgrid[fpoint_y]!r} attrs: {ds_sgrid[fpoint_y].attrs}\n"
        )
        raise ValueError(msg)

    return "spherical" if _is_coordinate_in_degrees(ds_sgrid[fpoint_x]) else "flat"


def _is_coordinate_in_degrees(da: xr.DataArray) -> bool:
    units = da.attrs.get("units")
    if units is None:
        raise ValueError(
            f"Coordinate {da.name!r} of your dataset has no 'units' attribute - we don't know what the spatial units are."
        )
    if isinstance(units, str) and "degree" in units.lower():
        return True
    return False


def _is_agrid(ds: xr.Dataset) -> bool:
    # check if U and V are defined on the same dimensions
    # if yes, interpret as A grid
    return set(ds["U"].dims) == set(ds["V"].dims)
