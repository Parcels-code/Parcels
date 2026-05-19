import xarray as xr

from .core import SGrid2DMetadata, SGrid3DMetadata, parse_grid_attrs


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


def assert_metadata_ds_consistency(ds: xr.Dataset, metadata: SGrid2DMetadata | SGrid3DMetadata): ...
