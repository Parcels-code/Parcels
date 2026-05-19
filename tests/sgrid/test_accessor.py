import hypothesis.strategies as st
import xarray as xr
from hypothesis import given

import parcels._strategies as pst
from parcels._datasets.structured.strategies import sgrid_dataset
from parcels._sgrid import SGrid2DMetadata


@st.composite
def grid_and_dataset(draw) -> tuple[SGrid2DMetadata, xr.Dataset]:
    # used only for test_metadata - for all other tests we can simply do `ds.sgrid.metadata` to get the metadata
    metadata_2d = draw(
        pst.sgrid.grid_metadata.filter(
            # parcels can only generate 2D Sgrid datasets, that also have coordinates
            lambda meta: isinstance(meta, SGrid2DMetadata) and meta.node_coordinates is not None
        )
    )
    ds = draw(sgrid_dataset(metadata_2d))
    return metadata_2d, ds


@given(grid_and_dataset())
def test_metadata(metadata_ds):
    metadata = metadata_ds[0]
    ds = metadata_ds[1]
    parsed_metadata = ds.sgrid.metadata
    assert parsed_metadata == metadata


@given(selection_axis=st.sampled_from(["X", "Y", "Z"]), ds=sgrid_dataset())
def test_isel(selection_axis, ds):
    # TODO: Add skip if Z but no Z dimension in ds

    # select nodes in axis direction
    # assert consistent

    # select edges
    # assert consistent
    ...
