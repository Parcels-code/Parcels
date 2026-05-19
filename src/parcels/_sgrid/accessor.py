import itertools
from collections.abc import Mapping
from typing import Any

import xarray as xr

from .core import SGrid2DMetadata, SGrid3DMetadata, get_n_faces, parse_grid_attrs


@xr.register_dataset_accessor("sgrid")
class SgridAccessor:
    def __init__(self, xarray_obj):
        self._ds = xarray_obj

    @property
    def metadata(self) -> SGrid2DMetadata | SGrid3DMetadata:
        grid_da = self._get_grid_topology()
        return parse_grid_attrs(grid_da.attrs)

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

    def isel(self, indexers: Mapping[Any, Any] | None = None, **indexers_kwargs):
        if indexers is None:
            indexers = {}

        for k, indexer in itertools.chain(indexers.items(), indexers_kwargs.items()):
            if not isinstance(indexer, slice):
                raise NotImplementedError(
                    f"sgrid.isel() only works on `slice` objects for the timebeing. Got indexer {indexer!r} for {k!r}"
                )

        _meta = self.metadata

        ...


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

    ...
