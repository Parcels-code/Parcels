from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import uxarray as ux
import xarray as xr
import xgcm

from parcels._core.field import Field, VectorField
from parcels._core.model import StructuredModel, UnstructuredModel
from parcels._core.utils.string import _assert_str_and_python_varname
from parcels._core.utils.time import get_datetime_type_calendar
from parcels._core.utils.time import is_compatible as datetime_is_compatible
from parcels._core.xgrid import _DEFAULT_XGCM_KWARGS, XGrid
from parcels._reprs import fieldset_repr
from parcels._typing import Mesh
from parcels.interpolators import (
    XConstantField,
)

if TYPE_CHECKING:
    from parcels._core.basegrid import BaseGrid
    from parcels._typing import TimeLike
__all__ = ["FieldSet"]


class FieldSet:
    """FieldSet class that holds hydrodynamic data needed to execute particles.

    Parameters
    ----------
    ds : xarray.Dataset | uxarray.UxDataset)
        xarray.Dataset and/or uxarray.UxDataset objects containing the field data.

    Notes
    -----
    The `ds` object is a xarray.Dataset or uxarray.UxDataset object.
    In XArray terminology, the (Ux)Dataset holds multiple (Ux)DataArray objects.
    Each (Ux)DataArray object is a single "field" that is associated with their own
    dimensions and coordinates within the (Ux)Dataset.

    A (Ux)Dataset object is associated with a single mesh, which can have multiple
    types of "points" (multiple "grids") (e.g. for UxDataSets, these are "face_lon",
    "face_lat", "node_lon", "node_lat", "edge_lon", "edge_lat"). Each (Ux)DataArray is
    registered to a specific set of points on the mesh.

    For UxDataset objects, each `UXDataArray.attributes` field dictionary contains
    the necessary metadata to help determine which set of points a field is registered
    to and what parent model the field is associated with. Parcels uses this metadata
    during execution for interpolation.  Each `UXDataArray.attributes` field dictionary
    must have:
    * "location" key set to "face", "node", or "edge" to define which pairing of points a field is associated with.
    * "mesh" key to define which parent model the fields are associated with (e.g. "fesom_mesh", "icon_mesh")

    """

    def __init__(self, fields: list[Field | VectorField]):
        for field in fields:
            if not isinstance(field, (Field, VectorField)):
                raise ValueError(f"Expected `field` to be a Field or VectorField object. Got {field}")
        assert_compatible_calendars(fields)

        self.fields = {f.name: f for f in fields}
        self.constants: dict[str, float] = {}

    def __getattr__(self, name):
        """Get the field by name. If the field is not found, check if it's a constant."""
        if name in self.fields:
            return self.fields[name]
        elif name in self.constants:
            return self.constants[name]
        else:
            raise AttributeError(f"FieldSet has no attribute '{name}'")

    def __repr__(self):
        return fieldset_repr(self)

    @property
    def time_interval(self):
        """Returns the valid executable time interval of the FieldSet,
        which is the intersection of the time intervals of all fields
        in the FieldSet.
        """
        time_intervals = (f.time_interval for f in self.fields.values())

        # Filter out Nones from constant Fields
        time_intervals = [t for t in time_intervals if t is not None]
        if len(time_intervals) == 0:  # All fields are constant fields
            return None
        return functools.reduce(lambda x, y: x.intersection(y), time_intervals)

    def add_field(self, field: Field, name: str | None = None):
        """Add a :class:`parcels.field.Field` object to the FieldSet.

        Parameters
        ----------
        field : parcels.field.Field
            Field object to be added
        name : str
            Name of the :class:`parcels.field.Field` object to be added. Defaults
            to name in Field object.
        """
        if not isinstance(field, (Field, VectorField)):
            raise ValueError(f"Expected `field` to be a Field or VectorField object. Got {type(field)}")
        assert_compatible_calendars((*self.fields.values(), field))

        name = field.name if name is None else name

        if name in self.fields:
            raise ValueError(f"FieldSet already has a Field with name '{name}'")

        self.fields[name] = field

    def add_constant_field(self, name: str, value, mesh: Mesh = "spherical"):
        """Wrapper function to add a Field that is constant in space,
           useful e.g. when using constant horizontal diffusivity

        Parameters
        ----------
        name : str
            Name of the :class:`parcels.field.Field` object to be added
        value :
            Value of the constant field
        mesh : str
            String indicating the type of mesh coordinates,

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        """
        ds = xr.Dataset(
            {name: (["lat", "lon"], np.full((1, 1), value))},
            coords={"lat": (["lat"], [0], {"axis": "Y"}), "lon": (["lon"], [0], {"axis": "X"})},
        )
        xgrid = xgcm.Grid(
            ds, coords={"X": {"left": "lon"}, "Y": {"left": "lat"}}, autoparse_metadata=False, **_DEFAULT_XGCM_KWARGS
        )
        grid = XGrid(xgrid, mesh=mesh)
        self.add_field(Field(name, ds[name], grid, interp_method=XConstantField))

    def add_constant(self, name, value):
        """Add a constant to the FieldSet.

        Parameters
        ----------
        name : str
            Name of the constant
        value :
            Value of the constant

        """
        _assert_str_and_python_varname(name)

        if name in self.constants:
            raise ValueError(f"FieldSet already has a constant with name '{name}'")
        if not isinstance(value, (float, np.floating, int, np.integer)):
            raise ValueError(f"FieldSet constants have to be of type float or int, got a {type(value)}")
        self.constants[name] = value

    @property
    def gridset(self) -> list[BaseGrid]:
        grids = []
        for field in self.fields.values():
            if field.grid not in grids:
                grids.append(field.grid)
        return grids

    @classmethod
    def from_ugrid_conventions(cls, ds: ux.UxDataset, mesh: str = "spherical"):
        """Create a FieldSet from a Parcels compliant uxarray.UxDataset.

        This is the primary ingestion method in Parcels for structured grid datasets.

        The main requirements for a uxDataset are naming conventions for vertical grid dimensions & coordinates

          zf - Name for coordinate and dimension for vertical positions at layer interfaces
          zc - Name for coordinate and dimension for vertical positions at layer centers

        Parameters
        ----------
        ds : uxarray.UxDataset
            uxarray.UxDataset as obtained from the uxarray package but with appropriate named vertical dimensions

        Returns
        -------
        FieldSet
            FieldSet object containing the fields from the dataset that can be used for a Parcels simulation.
        """
        model = UnstructuredModel.from_ugrid_conventions(ds, mesh)
        return cls(list(model.construct_fields()))

    @classmethod
    def from_sgrid_conventions(
        cls, ds: xr.Dataset, mesh: Mesh | None = None
    ):  # TODO: Update mesh to be discovered from the dataset metadata
        """Create a FieldSet from a dataset using SGRID convention metadata.

        This is the primary ingestion method in Parcels for structured grid datasets.

        Assumes that U, V, (and optionally W) variables are named 'U', 'V', and 'W' in the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            xarray.Dataset with SGRID convention metadata.
        mesh : str
            String indicating the type of mesh coordinates used during
            velocity interpolation. Options are "spherical" or "flat".

        Returns
        -------
        FieldSet
            FieldSet object containing the fields from the dataset that can be used for a Parcels simulation.

        Notes
        -----
        This method uses the SGRID convention metadata to parse the grid structure
        and create appropriate Fields for a Parcels simulation. The dataset should
        contain a variable with 'cf_role' attribute set to 'grid_topology'.

        See https://sgrid.github.io/ for more information on the SGRID conventions.
        """
        model = StructuredModel.from_sgrid_conventions(ds, mesh)
        return cls(model.construct_fields())


class CalendarError(Exception):  # TODO: Move to a parcels errors module
    """Exception raised when the calendar of a field is not compatible with the rest of the Fields. The user should ensure that they only add fields to a FieldSet that have compatible CFtime calendars."""


def assert_compatible_calendars(fields: Iterable[Field | VectorField]):
    time_intervals = [f.time_interval for f in fields if f.time_interval is not None]

    if len(time_intervals) == 0:  # All time intervals are none
        return

    reference_datetime_object = time_intervals[0].left

    for field in fields:
        if field.time_interval is None:
            continue

        if not datetime_is_compatible(reference_datetime_object, field.time_interval.left):
            msg = _format_calendar_error_message(field, reference_datetime_object)
            raise CalendarError(msg)


def _datetime_to_msg(example_datetime: TimeLike) -> str:
    datetime_type, calendar = get_datetime_type_calendar(example_datetime)
    msg = str(datetime_type)
    if calendar is not None:
        msg += f" with cftime calendar {calendar}'"
    return msg


def _format_calendar_error_message(field: Field | VectorField, reference_datetime: TimeLike) -> str:
    return f"Expected field {field.name!r} to have calendar compatible with datetime object {_datetime_to_msg(reference_datetime)}. Got field with calendar {_datetime_to_msg(field.time_interval.left)}. Have you considered using xarray to update the time dimension of the dataset to have a compatible calendar?"


_COPERNICUS_MARINE_AXIS_VARNAMES = {
    "X": "lon",
    "Y": "lat",
    "Z": "depth",
    "T": "time",
}


_COPERNICUS_MARINE_CF_STANDARD_NAME_FALLBACKS = {
    "UV": [
        (
            "eastward_sea_water_velocity",
            "northward_sea_water_velocity",
        ),  # GLOBAL_ANALYSISFORECAST_PHY_001_024, MEDSEA_ANALYSISFORECAST_PHY_006_013, BALTICSEA_ANALYSISFORECAST_PHY_003_006, BLKSEA_ANALYSISFORECAST_PHY_007_001, IBI_ANALYSISFORECAST_PHY_005_001, NWSHELF_ANALYSISFORECAST_PHY_004_013, MULTIOBS_GLO_PHY_MYNRT_015_003, MULTIOBS_GLO_PHY_W_3D_REP_015_007
        (
            "surface_geostrophic_eastward_sea_water_velocity",
            "surface_geostrophic_northward_sea_water_velocity",
        ),  # SEALEVEL_GLO_PHY_L4_MY_008_047, SEALEVEL_EUR_PHY_L4_NRT_008_060
        (
            "geostrophic_eastward_sea_water_velocity",
            "geostrophic_northward_sea_water_velocity",
        ),  # MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012
        (
            "sea_surface_wave_stokes_drift_x_velocity",
            "sea_surface_wave_stokes_drift_y_velocity",
        ),  # GLOBAL_ANALYSISFORECAST_WAV_001_027, MEDSEA_MULTIYEAR_WAV_006_012, ARCTIC_ANALYSIS_FORECAST_WAV_002_014, BLKSEA_ANALYSISFORECAST_WAV_007_003, IBI_ANALYSISFORECAST_WAV_005_005, NWSHELF_ANALYSISFORECAST_WAV_004_014
        ("sea_water_x_velocity", "sea_water_y_velocity"),  # ARCTIC_ANALYSISFORECAST_PHY_002_001
        (
            "eastward_sea_water_velocity_vertical_mean_over_pelagic_layer",
            "northward_sea_water_velocity_vertical_mean_over_pelagic_layer",
        ),  # GLOBAL_MULTIYEAR_BGC_001_033
    ],
    "W": ["upward_sea_water_velocity", "vertical_sea_water_velocity"],
}


def _is_agrid(ds: xr.Dataset) -> bool:
    # check if U and V are defined on the same dimensions
    # if yes, interpret as A grid
    return set(ds["U"].dims) == set(ds["V"].dims)
