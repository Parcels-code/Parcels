import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import xarray as xr

from parcels._python import invert_non_unique_mapping

from .core import FaceNodePadding, Padding, SGrid2DMetadata, SGrid3DMetadata, get_n_faces, get_n_nodes, parse_grid_attrs


@xr.register_dataset_accessor("sgrid")
class SgridAccessor:
    def __init__(self, xarray_obj):
        self._ds: xr.Dataset = xarray_obj

    @property
    def metadata(self) -> SGrid2DMetadata:
        grid_da = self._get_grid_topology()
        grid = parse_grid_attrs(grid_da.attrs)
        if isinstance(grid, SGrid3DMetadata):
            raise NotImplementedError("Support for 3D SGRID metadata not supported.")
        return grid

    def assert_metadata_ds_consistency(self) -> None:
        """Assert that the attached SGRID metadata is consistent with the dataset.

        Raises
        ------
        SGridDatasetInconsistency
            If the dataset is not consistent with its SGRID metadata.
        """
        _assert_metadata_ds_consistency(self._ds, self.metadata)

    def rename(self, name_dict: dict[str, str]) -> xr.Dataset:
        """Similar to Xarray's rename functionality - but also updates attached SGRID metadata."""
        ds = self._ds.copy()
        ds = ds.rename(name_dict)

        grid_da_name = self._get_grid_topology().name
        ds[grid_da_name].attrs = self.metadata.rename(name_dict).to_attrs()
        return ds

    def _get_grid_topology(self) -> xr.DataArray:
        grid_da = None
        for var_name in self._ds.variables:
            if self._ds[var_name].attrs.get("cf_role") == "grid_topology":
                grid_da = self._ds[var_name]

        if grid_da is None:
            raise ValueError(
                "No variable found in dataset with 'cf_role' attribute set to 'grid_topology'. This doesn't look to be an SGrid dataset - please make your dataset conforms to SGrid conventions https://sgrid.github.io/sgrid/"
            )
        return grid_da

    def isel(self, indexers: Mapping[str, Any] | None = None, **indexers_kwargs):
        """Index the dataset along SGRID spatial dimensions, keeping face and node dimensions consistent.

        For a provided index, this function derives the paired index from SGRID metadata, applies the indexes, and asserts the restulting dataset still complies with the SGRID metadata.

        Behaviour:

        - Only spatial (SGRID-registered) dimensions may be indexed.
        - Simultaneously indexing along two dimensions that belong to the same axis is not allowed.
        - For NONE/BOTH padding, only contiguous regions are supported (i.e., ``slice``
        indexers with ``step`` of ``None`` or ``1``). Indexing discontiguous regions are not well defined.


        Parameters
        ----------
        indexers : Mapping[str, Any], optional
            A mapping of dimension name to indexer, e.g. ``{"node_dimension1": slice(0, 10)}``.
            Mutually exclusive with ``**indexers_kwargs``.
        **indexers_kwargs
            Dimension-to-indexer pairs as keyword arguments. Mutually exclusive with
            ``indexers``.

        Returns
        -------
        xr.Dataset
            A new dataset indexed along the requested dimensions with all paired face/node
            dimensions adjusted accordingly.
        """
        if indexers_kwargs != {}:
            if indexers is not None:
                raise ValueError("Cannot provide both positional and keyword argument to .isel .")
            indexers = indexers_kwargs

        if indexers is None:
            raise ValueError("Must provide indexers either as a positional argument or as keyword arguments.")

        metadata = self.metadata

        _assert_not_indexing_along_same_axis(indexers, metadata)
        _assert_all_isel_along_axis(list(indexers.keys()), metadata)

        indexers = _complete_isel_indexing(self._ds, indexers, metadata)

        ds = self._ds.isel(indexers=indexers)
        _assert_metadata_ds_consistency(ds, metadata)
        return ds


def _assert_metadata_ds_consistency(ds: xr.Dataset, metadata: SGrid2DMetadata):
    _assert_face_node_sizes_consistent(ds, metadata)
    _assert_node_dimensions_have_xy_axes(ds, metadata)
    _assert_node_coordinates_well_defined(ds, metadata)


def _assert_face_node_sizes_consistent(ds: xr.Dataset, metadata: SGrid2DMetadata):
    vertical_dimensions: tuple[FaceNodePadding, ...] = metadata.vertical_dimensions or tuple()

    for obj in itertools.chain(metadata.face_dimensions, vertical_dimensions):
        face, node, padding = obj.face, obj.node, obj.padding

        try:
            n_nodes = ds.sizes[node]
        except KeyError:  # node dimension is not in this dataset
            continue
        try:
            n_faces = ds.sizes[face]
        except KeyError:  # face dimension is not in this dataset
            continue

        expected_n_faces = get_n_faces(n_nodes, padding)
        if expected_n_faces != n_faces:
            raise SGridDatasetInconsistency(
                f"Node dimension {node!r} has size {n_nodes}, and face dimension {face!r} has size of {n_faces}. "
                f"Due to dataset padding of {padding!r}, expected face dimension {face} to actually be size {expected_n_faces}."
            )


# Accepted CF units for longitude and latitude (https://cfconventions.org/cf-conventions/cf-conventions.html#longitude-coordinate)
_DEGREE_UNITS = {"degree", "degrees"}
_DEGREE_UNITS_PER_AXIS = {
    "X": {"degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE"} | _DEGREE_UNITS,
    "Y": {"degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN"} | _DEGREE_UNITS,
}


def _assert_node_dimensions_have_xy_axes(ds: xr.Dataset, metadata: SGrid2DMetadata):
    # SGRID conventions don't explicitly mandate an X then Y ordering, but this is evident from all their examples
    for dim, expected_axis in zip(metadata.node_dimensions, ("X", "Y"), strict=True):
        if dim not in ds.variables:  # dimension has no coordinate variable in this dataset
            continue

        axis = ds[dim].attrs.get("axis")
        if axis != expected_axis:
            raise SGridDatasetInconsistency(
                f"SGRID metadata lists {dim!r} in node_dimensions={metadata.node_dimensions!r}, so expected variable {dim!r} "
                f"to have attribute axis={expected_axis!r}. Instead got axis={axis!r}. "
                f'HINT: Make sure node_dimensions in the SGRID metadata is ordered as "X" then "Y", and that the '
                f"coordinate variables in the dataset have the corresponding 'axis' attributes."
            )


def _assert_node_coordinates_well_defined(ds: xr.Dataset, metadata: SGrid2DMetadata):
    if metadata.node_coordinates is None:
        return

    node_dims = metadata.node_dimensions
    both_node_dims_in_ds = all(d in ds.dims for d in node_dims)

    for coord, node_dim, expected_axis in zip(metadata.node_coordinates, node_dims, ("X", "Y"), strict=True):
        if coord not in ds.variables:
            raise SGridDatasetInconsistency(
                f"SGRID metadata lists {coord!r} in node_coordinates={metadata.node_coordinates!r}, "
                f"but there is no variable {coord!r} in the dataset."
            )

        da = ds[coord]
        if not set(da.dims) <= set(node_dims):
            raise SGridDatasetInconsistency(
                f"Node coordinate {coord!r} has dims {da.dims}, but node coordinates can only be defined on the "
                f"node_dimensions={node_dims!r}."
            )
        # A 1D node coordinate (i.e., a rectilinear grid) must lie along its own axis. If one of the node
        # dimensions has been indexed away (e.g., via isel) a 2D coordinate can legitimately be 1D along the other
        if both_node_dims_in_ds and da.ndim == 1 and da.dims != (node_dim,):
            raise SGridDatasetInconsistency(
                f"Node coordinate {coord!r} is 1D with dims {da.dims}, but as entry {expected_axis!r} in "
                f"node_coordinates={metadata.node_coordinates!r} it is expected to be defined on node dimension {node_dim!r}. "
                f'HINT: Make sure node_coordinates in the SGRID metadata is ordered as "X" then "Y".'
            )

        units = da.attrs.get("units")
        if isinstance(units, str) and "degree" in units.lower() and units not in _DEGREE_UNITS_PER_AXIS[expected_axis]:
            raise SGridDatasetInconsistency(
                f"Node coordinate {coord!r} has units={units!r}, but as entry {expected_axis!r} in "
                f"node_coordinates={metadata.node_coordinates!r} expected one of {sorted(_DEGREE_UNITS_PER_AXIS[expected_axis])}. "
                f'HINT: Make sure node_coordinates in the SGRID metadata is ordered as "X" (longitude) then "Y" (latitude).'
            )


class SGridDatasetInconsistency(ValueError):
    """Attached metadata is not compatible with Xarray dataset"""

    pass


def _get_dim_to_axis_mapping(grid: SGrid2DMetadata) -> dict[str, Literal["X", "Y", "Z"]]:
    fnp_x = grid.face_dimensions[0]
    fnp_y = grid.face_dimensions[1]
    fnp_z = grid.vertical_dimensions[0] if grid.vertical_dimensions is not None else None

    d = {
        fnp_x.node: "X",
        fnp_x.face: "X",
        fnp_y.node: "Y",
        fnp_y.face: "Y",
    }
    if fnp_z is not None:
        d.update({fnp_z.node: "Z", fnp_z.face: "Z"})
    return cast(dict[str, Literal["X", "Y", "Z"]], d)


def _get_axis_info(grid: SGrid2DMetadata) -> dict[str, tuple[FaceNodePadding, bool]]:
    """For each spatial dim, return (FaceNodePadding it belongs to, True if node dim)."""
    result: dict[str, tuple[FaceNodePadding, bool]] = {}
    all_fnps = list(grid.face_dimensions) + list(grid.vertical_dimensions or [])
    for fnp in all_fnps:
        result[fnp.node] = (fnp, True)
        result[fnp.face] = (fnp, False)
    return result


def get_dim_position(grid: SGrid2DMetadata, dim: str) -> "Literal['face'] | Padding":
    """Returns 'face' if dim is a face dimension, or the SGRID Padding value if it is a node dimension."""
    axis_info = _get_axis_info(grid)
    if dim not in axis_info:
        raise ValueError(f"Dimension {dim!r} is not a spatial SGRID dimension in this grid.")
    fnp, is_node = axis_info[dim]
    return fnp.padding if is_node else "face"


def _derive_paired_indexer(
    indexer: Any,
    indexer_is_node: bool,
    padding: Padding,
    dim_size: int | None = None,
) -> tuple[Any, Any]:
    """Given a user's indexer for one side of a face-node pair, return ``(normalized_user_indexer, paired_indexer)``.

    For HIGH/LOW padding, face and node dims are the same size so both the normalised user
    indexer and the paired indexer are identical to the original ``user_indexer``.
    For NONE/BOTH padding, the slice is first normalised to non-negative absolute indices (via
    ``slice.indices``) and then the stop of the paired indexer is adjusted by ±1.

    ``n_user_dim`` is required for NONE/BOTH padding so that negative starts and ``None`` stops
    can be resolved to unambiguous absolute positions.

    Scalar (integer) and list indexers raise for NONE/BOTH because there is no unambiguous
    paired position.

    Returns
    -------
    tuple[Any, Any]
        ``(normalized_user_indexer, paired_indexer)`` — the first element is the user's indexer
        after normalisation (unchanged for HIGH/LOW), the second is the derived indexer for the
        other side of the face-node pair.
    """
    if padding in (Padding.HIGH, Padding.LOW):
        return indexer, indexer

    # NONE and BOTH: only slices with step in {None, 1} are supported
    if not isinstance(indexer, slice):
        raise ValueError(
            f"Scalar and list indexers are not supported for NONE/BOTH padding. "
            f"Got indexer {indexer!r}. Use a slice instead."
        )
    if indexer.step not in (None, 1):
        raise ValueError(f"Slices with step != 1 are not supported for NONE/BOTH padding. Got step={indexer.step!r}.")
    if dim_size is None:
        raise ValueError("dim_size must be provided for NONE/BOTH padding to correctly handle slices.")

    # Normalise to non-negative absolute indices so the arithmetic below is unambiguous.
    abs_start, abs_stop, _ = indexer.indices(dim_size)
    normalized_user_indexer = slice(abs_start, abs_stop)

    start, stop = abs_start, abs_stop

    # Adjust stop: positive stops reference from the start of the array, so ±1 is needed.
    if stop is not None and stop > 0:
        stop = get_n_faces(stop, padding=padding) if indexer_is_node else get_n_nodes(stop, padding=padding)

    return normalized_user_indexer, slice(start, stop)


def _assert_not_indexing_along_same_axis(indexers: Mapping[Any, Any], metadata: SGrid2DMetadata) -> None:
    dim_to_axis = _get_dim_to_axis_mapping(metadata)
    indexer_dim_to_axis = {dim: dim_to_axis.get(dim) for dim in indexers}

    indexer_axis_to_dim = invert_non_unique_mapping(indexer_dim_to_axis)
    for axis, dims in indexer_axis_to_dim.items():
        if axis is None:
            continue

        if len(dims) > 1:
            msg = f"Dims {dims} are on the same axis {axis!r} according to SGRID metadata - cannot simultaneously index along multiple dimensions in the same axis."
            raise ValueError(msg)


def _assert_all_isel_along_axis(index_dims: Sequence[str], metadata: SGrid2DMetadata):
    dim_to_axis = _get_dim_to_axis_mapping(metadata)
    for dim in index_dims:
        try:
            dim_to_axis[dim]
        except KeyError as e:
            raise ValueError(
                f"Cannot use SGRID accessor to .isel non-spatial (/SGRID related) dimension {dim!r}."
            ) from e


def _complete_isel_indexing(
    ds: xr.Dataset,
    indexers: Mapping[Any, Any],
    grid: SGrid2DMetadata,
) -> Mapping[Any, Any]:
    """For each user-supplied (dim, indexer), expand to both the face and node dim on that axis,
    deriving the paired indexer according to the padding type.
    """
    axis_info = _get_axis_info(grid)
    ret: dict[Any, Any] = {}

    for user_dim, user_indexer in indexers.items():
        fnp, user_is_node = axis_info[user_dim]

        n_user_dim = ds.sizes.get(user_dim)
        normalized_user, paired_indexer = _derive_paired_indexer(
            user_indexer, user_is_node, fnp.padding, dim_size=n_user_dim
        )

        node_indexer = normalized_user if user_is_node else paired_indexer
        face_indexer = paired_indexer if user_is_node else normalized_user

        if fnp.node in ds.dims:
            ret[fnp.node] = node_indexer
        if fnp.face in ds.dims:
            ret[fnp.face] = face_indexer

    return ret
