import itertools
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import xarray as xr

from parcels._python import invert_non_unique_mapping

from .core import SGrid2DMetadata, SGrid3DMetadata, get_n_faces, parse_grid_attrs


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

    def rename(self, name_dict: dict[str, str]) -> xr.Dataset:
        """Similar to Xarray's rename functionality - but also updates the SGRID metadata attributes."""
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
        """TODO: Docstring"""
        if indexers_kwargs != {}:
            if indexers is not None:
                raise ValueError("Cannot provide both positional and keyword argument to .isel .")
            indexers = indexers_kwargs

        assert indexers is not None

        for k, indexer in indexers.items():
            if not isinstance(indexer, slice):
                raise NotImplementedError(
                    f"sgrid.isel() only works on `slice` objects for the timebeing. Got indexer {indexer!r} for {k!r}"
                )

        metadata = self.metadata

        _assert_not_indexing_along_same_axis(indexers, metadata)
        _assert_all_isel_along_axis(indexers.keys(), metadata)

        indexers = _complete_isel_indexing(indexers, metadata, self._ds.dims.keys())

        ds = self._ds.isel(indexers=indexers)
        assert_metadata_ds_consistency(ds, metadata)
        return ds


def assert_metadata_ds_consistency(ds: xr.Dataset, metadata: SGrid2DMetadata):
    vertical_dimensions = metadata.vertical_dimensions or tuple()

    for obj in itertools.chain(metadata.face_dimensions, vertical_dimensions):
        face, node, padding = obj.face, obj.node, obj.padding

        try:
            n_nodes = ds.dims[node]
        except KeyError:
            # node dimension is not in this dataset
            continue

        try:
            n_faces = ds.dims[face]
        except KeyError:
            # face dimension is not in this dataset
            continue

        expected_n_faces = get_n_faces(n_nodes, padding)

        if expected_n_faces != n_faces:
            raise SGridDatasetInconsistency(
                f"Node dimension {node!r} has size {n_nodes}, and face dimension {face!r} has size of {n_faces}. "
                f"Due to dataset padding of {padding!r}, expected face dimension {face} to actually be size {expected_n_faces}."
            )

    # TODO: Also check on coordinates


class SGridDatasetInconsistency(Exception):
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
    return d


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
    indexers: Mapping[Any, Any], grid: SGrid2DMetadata, dims_in_dataset: Sequence[str]
) -> Mapping[Any, Any]:
    """Copies indexers to the other dataset dimensions defined on the same axis."""
    ret = {}
    dim_to_axis = _get_dim_to_axis_mapping(grid)
    indexers_by_axis = {dim_to_axis[dim]: indexer for dim, indexer in indexers.items()}

    for dim, axis in dim_to_axis.items():
        if dim in dims_in_dataset and axis in indexers_by_axis:
            ret[dim] = indexers_by_axis[axis]
    return ret
