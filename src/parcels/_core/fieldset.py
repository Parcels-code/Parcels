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
from parcels._core.utils import sgrid
from parcels._core.utils.string import _assert_str_and_python_varname
from parcels._core.utils.time import get_datetime_type_calendar
from parcels._core.utils.time import is_compatible as datetime_is_compatible
from parcels._core.uxgrid import UxGrid
from parcels._core.xgrid import _DEFAULT_XGCM_KWARGS, XGrid
from parcels._logger import logger
from parcels._reprs import fieldset_repr
from parcels._typing import Mesh
from parcels.convert import _ds_rename_using_standard_names
from parcels.interpolators import (
    CGrid_Velocity,
    Ux_Velocity,
    UxConstantFaceConstantZC,
    UxConstantFaceLinearZF,
    UxLinearNodeConstantZC,
    UxLinearNodeLinearZF,
    XConstantField,
    XLinear,
    XLinear_Velocity,
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
        """Add a constant to the FieldSet. Note that all constants are
        stored as 32-bit floats.

        Parameters
        ----------
        name : str
            Name of the constant
        value :
            Value of the constant (stored as 32-bit float)


        Examples
        --------
        Tutorials using fieldset.add_constant:
        `Analytical advection <../examples/tutorial_analyticaladvection.ipynb>`__
        `Diffusion <../examples/tutorial_diffusion.ipynb>`__
        `Periodic boundaries <../examples/tutorial_periodic_boundaries.ipynb>`__
        """
        _assert_str_and_python_varname(name)

        if name in self.constants:
            raise ValueError(f"FieldSet already has a constant with name '{name}'")
        if not isinstance(value, (float, np.floating, int, np.integer)):
            raise ValueError(f"FieldSet constants have to be of type float or int, got a {type(value)}")
        self.constants[name] = np.float32(value)

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
        ds_dims = list(ds.dims)
        if not all(dim in ds_dims for dim in ["time", "zf", "zc"]):
            raise ValueError(
                f"Dataset missing one of the required dimensions 'time', 'zf', or 'zc' for uxDataset. Found dimensions {ds_dims}"
            )

        grid = UxGrid(ds.uxgrid, z=ds.coords["zf"], mesh=mesh)
        ds = _discover_ux_U_and_V(ds)

        fields = {}
        if "U" in ds.data_vars and "V" in ds.data_vars:
            fields["U"] = Field("U", ds["U"], grid, _select_uxinterpolator(ds["U"]))
            fields["V"] = Field("V", ds["V"], grid, _select_uxinterpolator(ds["V"]))

            if "W" in ds.data_vars:
                fields["W"] = Field("W", ds["W"], grid, _select_uxinterpolator(ds["W"]))
                fields["UVW"] = VectorField(
                    "UVW", fields["U"], fields["V"], fields["W"], vector_interp_method=Ux_Velocity
                )
            else:
                fields["UV"] = VectorField("UV", fields["U"], fields["V"], vector_interp_method=Ux_Velocity)

        for varname in set(ds.data_vars) - set(fields.keys()):
            fields[varname] = Field(varname, ds[varname], grid, _select_uxinterpolator(ds[varname]))

        return cls(list(fields.values()))

    @classmethod
    def from_fesom2(cls, ds: ux.UxDataset, mesh: str = "spherical"):
        """Create a FieldSet from a FESOM2 uxarray.UxDataset.

        Parameters
        ----------
        ds : uxarray.UxDataset
            uxarray.UxDataset as obtained from the uxarray package.

        Returns
        -------
        FieldSet
            FieldSet object containing the fields from the dataset that can be used for a Parcels simulation.
        """
        ds = ds.copy()
        ds_dims = list(ds.dims)
        if not all(dim in ds_dims for dim in ["time", "nz", "nz1"]):
            raise ValueError(
                f"Dataset missing one of the required dimensions 'time', 'nz', or 'nz1' for FESOM data. Found dimensions {ds_dims}"
            )
        ds = ds.rename(
            {
                "nz": "zf",  # Vertical Interface
                "nz1": "zc",  # Vertical Center
            }
        ).set_index(zf="zf", zc="zc")

        return FieldSet.from_ugrid_conventions(ds, mesh=mesh)

    @classmethod
    def from_icon(cls, ds: ux.UxDataset, mesh: str = "spherical"):
        """Create a FieldSet from a ICON uxarray.UxDataset.

        Parameters
        ----------
        ds : uxarray.UxDataset
            uxarray.UxDataset as obtained from the uxarray package.

        Returns
        -------
        FieldSet
            FieldSet object containing the fields from the dataset that can be used for a Parcels simulation.
        """
        ds = ds.copy()
        ds_dims = list(ds.dims)
        if not all(dim in ds_dims for dim in ["time", "depth", "depth_2"]):
            raise ValueError(
                f"Dataset missing one of the required dimensions 'time', 'depth', or 'depth_2' for ICON data. Found dimensions {ds_dims}"
            )
        ds = ds.rename(
            {
                "depth_2": "zf",  # Vertical Interface
                "depth": "zc",  # Vertical Center
            }
        ).set_index(zf="zf", zc="zc")
        return FieldSet.from_ugrid_conventions(ds, mesh=mesh)

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
        _, xgcm_kwargs = sgrid.parse_sgrid(ds)

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

        # Create fields from data variables, skipping grid metadata variables
        # Skip variables that are SGRID metadata (have cf_role='grid_topology')
        skip_vars = set()
        for var in ds.data_vars:
            if ds[var].attrs.get("cf_role") == "grid_topology":
                skip_vars.add(var)

        fields = {}
        if "U" in ds.data_vars and "V" in ds.data_vars:
            vector_interp_method = XLinear_Velocity if _is_agrid(ds) else CGrid_Velocity
            fields["U"] = Field("U", ds["U"], grid, XLinear)
            fields["V"] = Field("V", ds["V"], grid, XLinear)
            fields["UV"] = VectorField("UV", fields["U"], fields["V"], vector_interp_method=vector_interp_method)

            if "W" in ds.data_vars:
                fields["W"] = Field("W", ds["W"], grid, XLinear)
                fields["UVW"] = VectorField(
                    "UVW", fields["U"], fields["V"], fields["W"], vector_interp_method=vector_interp_method
                )

        for varname in set(ds.data_vars) - set(fields.keys()) - skip_vars:
            fields[varname] = Field(varname, ds[varname], grid, XLinear)

        return cls(list(fields.values()))


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


def _format_calendar_error_message(field: Field, reference_datetime: TimeLike) -> str:
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


def _discover_ux_U_and_V(ds: ux.UxDataset) -> ux.UxDataset:
    # Common variable names for U and V found in UxDatasets
    common_ux_UV = [("unod", "vnod"), ("u", "v")]
    common_ux_W = ["w"]

    if "W" not in ds:
        for common_W in common_ux_W:
            if common_W in ds:
                ds = _ds_rename_using_standard_names(ds, {common_W: "W"})
                break

    if "U" in ds and "V" in ds:
        return ds  # U and V already present
    elif "U" in ds or "V" in ds:
        raise ValueError(
            "Dataset has only one of the two variables 'U' and 'V'. Please rename the appropriate variable in your dataset to have both 'U' and 'V' for Parcels simulation."
        )

    for common_U, common_V in common_ux_UV:
        if common_U in ds:
            if common_V not in ds:
                raise ValueError(
                    f"Dataset has variable with standard name {common_U!r}, "
                    f"but not the matching variable with standard name {common_V!r}. "
                    "Please rename the appropriate variables in your dataset to have both 'U' and 'V' for Parcels simulation."
                )
            else:
                ds = _ds_rename_using_standard_names(ds, {common_U: "U", common_V: "V"})
                break

        else:
            if common_V in ds:
                raise ValueError(
                    f"Dataset has variable with standard name {common_V!r}, "
                    f"but not the matching variable with standard name {common_U!r}. "
                    "Please rename the appropriate variables in your dataset to have both 'U' and 'V' for Parcels simulation."
                )
            continue

    return ds


def _select_uxinterpolator(da: ux.UxDataArray):
    """Selects the appropriate uxarray interpolator for a given uxarray UxDataArray"""
    supported_uxinterp_mapping = {
        # (zc,n_face): face-center laterally, layer centers vertically — piecewise constant
        "zc,n_face": UxConstantFaceConstantZC,
        # (zc,n_node): node/corner laterally, layer centers vertically — barycentric lateral & piecewise constant vertical
        "zc,n_node": UxLinearNodeConstantZC,
        # (zf,n_node): node/corner laterally, layer interfaces vertically — barycentric lateral & linear vertical
        "zf,n_node": UxLinearNodeLinearZF,
        # (zf,n_face): face-center laterally, layer interfaces vertically — piecewise constant lateral & linear vertical
        "zf,n_face": UxConstantFaceLinearZF,
    }
    # Extract only spatial dimensions, neglecting time
    da_spatial_dims = tuple(d for d in da.dims if d not in ("time",))
    if len(da_spatial_dims) != 2:
        raise ValueError(
            "Fields on unstructured grids must have two spatial dimensions, one vertical (zf or zc) and one lateral (n_face, n_edge, or n_node)"
        )

    # Construct key (string) for mapping to interpolator
    # Find vertical and lateral tokens
    vdim = None
    ldim = None
    for d in da_spatial_dims:
        if d in ("zf", "zc"):
            vdim = d
        if d in ("n_face", "n_node"):
            ldim = d
    # Map to supported interpolators
    if vdim and ldim:
        key = f"{vdim},{ldim}"
        if key in supported_uxinterp_mapping.keys():
            return supported_uxinterp_mapping[key]

    return None


# TODO: Refactor later into something like `parcels._metadata.discover(dataset)` helper that can be used to discover important metadata like this. I think this whole metadata handling should be refactored into its own module.
def _get_mesh_type_from_sgrid_dataset(ds_sgrid: xr.Dataset) -> Mesh:
    """Small helper to inspect SGRID metadata and dataset metadata to determine mesh type."""
    grid_da = sgrid.get_grid_topology(ds_sgrid)
    if grid_da is None:
        raise ValueError("Dataset does not contain SGRID grid topology metadata (cf_role='grid_topology').")

    sgrid_metadata = sgrid.parse_grid_attrs(grid_da.attrs)

    fpoint_x, fpoint_y = sgrid_metadata.node_coordinates

    if _is_coordinate_in_degrees(ds_sgrid[fpoint_x]) ^ _is_coordinate_in_degrees(ds_sgrid[fpoint_x]):
        msg = (
            f"Mismatch in units between X and Y coordinates.\n"
            f"  Coordinate {ds_sgrid[fpoint_x]!r} attrs: {ds_sgrid[fpoint_x].attrs}\n"
            f"  Coordinate {ds_sgrid[fpoint_y]!r} attrs: {ds_sgrid[fpoint_y].attrs}\n"
        )
        raise ValueError(msg)

    return "spherical" if _is_coordinate_in_degrees(ds_sgrid[fpoint_x]) else "flat"


def _is_agrid(ds: xr.Dataset) -> bool:
    # check if U and V are defined on the same dimensions
    # if yes, interpret as A grid
    return set(ds["U"].dims) == set(ds["V"].dims)


def _is_coordinate_in_degrees(da: xr.DataArray) -> bool:
    units = da.attrs.get("units")
    if units is None:
        raise ValueError(
            f"Coordinate {da.name!r} of your dataset has no 'units' attribute - we don't know what the spatial units are."
        )
    if isinstance(units, str) and "degree" in units.lower():
        return True
    return False
