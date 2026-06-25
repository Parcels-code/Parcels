from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import cf_xarray  # noqa: F401
import uxarray as ux
import xarray as xr

import parcels._sgrid as sgrid
from parcels._core.basegrid import BaseGrid
from parcels._core.field import Field, VectorField
from parcels._core.utils.time import TimeInterval
from parcels._core.uxgrid import UxGrid
from parcels._core.xgrid import (
    XGrid,
    _transpose_xfield_data_to_tzyx,
    assert_all_field_dims_have_axis,  # noqa: F401, leave import for now until decision is made # TODO v4: Make decision
)
from parcels._logger import logger
from parcels._typing import Mesh
from parcels.convert import _ds_rename_using_standard_names
from parcels.interpolators import (
    CGrid_Velocity,
    Ux_Velocity,
    UxConstantFaceConstantZC,
    UxConstantFaceLinearZF,
    UxLinearNodeConstantZC,
    UxLinearNodeLinearZF,
    XLinear,
    XLinear_Velocity,
)
from parcels.interpolators._base import ScalarInterpolator, VectorInterpolator


class ModelData(ABC):
    data: Any
    grid: BaseGrid
    field_to_interpolator: dict[str, ScalarInterpolator | VectorInterpolator]

    @abstractmethod
    def construct_fields(self) -> list[Field | VectorField]: ...

    @property
    @abstractmethod
    def scalar_field_names(self) -> list[str]: ...

    @abstractmethod
    def assert_valid_field_data(self, field_data: Any) -> None: ...

    def assert_valid_model_data(self) -> None:
        for field_name in self.scalar_field_names:
            field_data = self.data[field_name]
            try:
                self.assert_valid_field_data(field_data)
            except Exception as e:
                e.add_note(f"Error validating field {field_name!r}.")
                raise e
        return

    @property
    def time_interval(self) -> TimeInterval | None:
        try:
            time_interval = _get_time_interval(self.data)
        except ValueError as e:
            e.add_note(
                f"Error getting time interval for:\n {self.data!r}\n\nAre you sure that the time dimension on the xarray dataset is stored as timedelta, datetime or cftime datetime objects?"
            )
            raise e
        return time_interval


def preprocess_sgrid_model_data(ds: xr.Dataset) -> xr.Dataset:
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata

    for field_name in set(ds.data_vars) - {ds.sgrid._get_grid_topology().name}:
        ds[field_name] = _transpose_xfield_data_to_tzyx(ds[field_name], metadata)
    return ds


class StructuredModelData(ModelData):
    def __init__(self, data: xr.Dataset, mesh: Mesh):
        if not isinstance(data, xr.Dataset):
            raise ValueError(f"Expected `data` to be an xarray.Dataset . Got {type(data)}")

        data = preprocess_sgrid_model_data(data)
        grid = XGrid(data, mesh)

        self.data = data
        self.grid = grid
        self.field_to_interpolator = {}
        self._fields: list[Field | VectorField] | None = None
        self.assert_valid_model_data()

    def assert_valid_field_data(self, field_data: xr.DataArray) -> None:
        # assert_all_field_dims_have_axis(field_data, self.grid.xgcm_grid) #! TODO v4: These checks should be revisited
        _assert_has_time_coordinate(field_data)

    @property
    def scalar_field_names(self) -> list[str]:
        # Create fields from data variables, skipping grid metadata variables
        # Skip variables that are SGRID metadata (have cf_role='grid_topology')
        skip_vars = set()
        for var in self.data.data_vars:
            if self.data[var].attrs.get("cf_role") == "grid_topology":
                skip_vars.add(var)
        return list(set(self.data.data_vars) - skip_vars)

    def construct_fields(self) -> list[Field | VectorField]:
        single_fields: dict[str, Field] = {}
        vector_fields: dict[str, VectorField] = {}
        scalar_field_names = self.scalar_field_names
        if "U" in scalar_field_names and "V" in scalar_field_names:
            interp_method = XLinear_Velocity() if _is_agrid(self.data) else CGrid_Velocity()
            single_fields["U"] = Field("U", self)
            single_fields["V"] = Field("V", self)
            vector_fields["UV"] = VectorField("UV", single_fields["U"], single_fields["V"], interp_method=interp_method)

            if "W" in scalar_field_names:
                single_fields["W"] = Field("W", self)
                vector_fields["UVW"] = VectorField(
                    "UVW",
                    single_fields["U"],
                    single_fields["V"],
                    single_fields["W"],
                    interp_method=interp_method,
                )

        fields: dict[str, Field | VectorField] = {**single_fields, **vector_fields}
        for varname in set(scalar_field_names) - set(fields.keys()):
            fields[varname] = Field(str(varname), self)

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

        # if "lon" not in ds.coords or "lat" not in ds.coords:
        #     node_dimensions = sgrid.load_mappings(ds.grid.node_dimensions)
        #     ds["lon"] = ds[node_dimensions[0]]
        #     ds["lat"] = ds[node_dimensions[1]]

        model = cls(ds, mesh=mesh)
        model._fields = model.construct_fields()
        for f in model._fields:
            if isinstance(f, Field):
                f.interp_method = XLinear()
        return model


CONSTANT_FIELD_MODELS = {
    mesh: StructuredModelData.from_sgrid_conventions(
        xr.Dataset(
            {},
            coords={
                "lat": (["lat"], [0], {"axis": "Y"}),
                "lon": (["lon"], [0], {"axis": "X"}),
                "depth": (["depth"], [0], {"axis": "Z"}),
                "time": (["time"], [0], {"axis": "T"}),
            },
        ).pipe(
            sgrid._attach_sgrid_metadata,
            sgrid.SGrid2DMetadata(
                cf_role="grid_topology",
                topology_dimension=2,
                node_dimensions=("lon", "lat"),
                face_dimensions=(
                    sgrid.FaceNodePadding("XC", "lon", sgrid.Padding.LOW),
                    sgrid.FaceNodePadding("YC", "lat", sgrid.Padding.LOW),
                ),
            ),
        ),
        mesh=mesh,  # type:ignore
    )
    for mesh in ["flat", "spherical"]
}


class UnstructuredModelData(ModelData):
    def __init__(self, data: ux.UxDataset, grid: UxGrid):
        if not isinstance(data, ux.UxDataset):
            raise ValueError(f"Expected `data` to be an uxarray.UxDataset . Got {type(data)}")

        if not isinstance(grid, UxGrid):
            raise ValueError(f"Expected `grid` to be a parcels UxGrid object. Got {type(grid)}.")

        self.data = data
        self.grid = grid
        self.field_to_interpolator = {}
        self._fields: list[Field | VectorField] | None = None

    def construct_fields(self) -> list[Field | VectorField]:
        single_fields: dict[str, Field] = {}
        vector_fields: dict[str, VectorField] = {}
        scalar_field_names = self.scalar_field_names
        if "U" in scalar_field_names and "V" in scalar_field_names:
            single_fields["U"] = Field("U", self)
            single_fields["V"] = Field("V", self)
            vector_fields["UV"] = VectorField("UV", single_fields["U"], single_fields["V"], interp_method=Ux_Velocity())

            if "W" in scalar_field_names:
                single_fields["W"] = Field("W", self)
                vector_fields["UVW"] = VectorField(
                    "UVW", single_fields["U"], single_fields["V"], single_fields["W"], interp_method=Ux_Velocity()
                )

        fields: dict[str, Field | VectorField] = {**single_fields, **vector_fields}
        for varname in set(scalar_field_names) - set(single_fields.keys()):
            fields[varname] = Field(str(varname), self)

        return list(fields.values())

    def assert_valid_field_data(self, field_data: ux.UxDataArray) -> None:
        _assert_valid_uxdataarray(field_data)
        _assert_has_time_coordinate(field_data)

    @property
    def scalar_field_names(self) -> list[str]:
        return list(self.data.data_vars)

    @classmethod
    def from_ugrid_conventions(cls, ds: ux.UxDataset, mesh: str = "spherical"):
        ds_dims = list(ds.dims)
        if not all(dim in ds_dims for dim in ["time", "zf", "zc"]):
            raise ValueError(
                f"Dataset missing one of the required dimensions 'time', 'zf', or 'zc' for uxDataset. Found dimensions {ds_dims}"
            )

        grid = UxGrid(ds.uxgrid, z=ds.coords["zf"], mesh=mesh)
        ds = _discover_ux_U_and_V(ds)
        model = cls(ds, grid)
        model._fields = model.construct_fields()
        for f in model._fields:
            if isinstance(f, Field):
                interp_cls = _select_uxinterpolator(model.data[f.name])
                if interp_cls is not None:
                    f.interp_method = interp_cls()
        return model


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


def _is_agrid(ds: xr.Dataset) -> bool:
    # check if U and V are defined on the same dimensions
    # if yes, interpret as A grid
    return set(ds["U"].dims) == set(ds["V"].dims)


def _get_time_interval(data: xr.DataArray | ux.UxDataArray) -> TimeInterval | None:
    if "time" not in data or data["time"].size == 1:
        return None

    return TimeInterval(data.time.values[0], data.time.values[-1])


def _assert_valid_uxdataarray(data: ux.UxDataArray):
    """Verifies that all the required attributes are present in the xarray.DataArray or
    uxarray.UxDataArray object.
    """
    # Validate dimensions
    if not ("zf" in data.dims or "zc" in data.dims):
        raise ValueError(
            "Field is missing a 'zf' or 'zc' dimension in the field's metadata. "
            "This attribute is required for xarray.DataArray objects."
        )

    if "time" not in data.dims:
        raise ValueError(
            "Field is missing a 'time' dimension in the field's metadata. "
            "This attribute is required for xarray.DataArray objects."
        )


def _assert_has_time_coordinate(da: xr.DataArray) -> None:
    if da.shape[0] > 1:
        if "time" not in da.coords:
            raise ValueError("Field data is missing a 'time' coordinate.")
    return
