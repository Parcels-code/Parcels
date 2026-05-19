import hypothesis.strategies as st
import pytest
import xarray as xr
from hypothesis import assume, given

import parcels._strategies as pst
from parcels._datasets.structured.strategies import sgrid_dataset
from parcels._sgrid import SGrid2DMetadata
from parcels._sgrid.accessor import SGridDatasetInconsistency, assert_metadata_ds_consistency


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


@given(sgrid_dataset())
def test_assert_metadata_ds_consistency(ds):
    metadata: SGrid2DMetadata = ds.sgrid.metadata
    assert_metadata_ds_consistency(ds, metadata)


@given(sgrid_dataset())
def test_assert_metadata_ds_consistency_dropped_dim(ds):
    metadata: SGrid2DMetadata = ds.sgrid.metadata

    # dropping one of the SGRID dimensions is fine
    first_face_dim = metadata.face_dimensions[0].face

    assume(first_face_dim in ds.dims)

    ds = ds.isel({first_face_dim: 0})
    assert_metadata_ds_consistency(ds, metadata)


@given(ds=sgrid_dataset())
def test_assert_metadata_ds_consistency_failures(ds):
    metadata: SGrid2DMetadata = ds.sgrid.metadata
    first_face_dim = metadata.face_dimensions[0].face

    assume(first_face_dim in ds.dims)

    ds = ds.isel({metadata.face_dimensions[0].face: slice(None, -1)})

    with pytest.raises(
        SGridDatasetInconsistency,
        match="Node dimension .* has size .*, and face dimension .* has size of .* .* expected face dimension .* to actually be size .*",
    ):
        assert_metadata_ds_consistency(ds, metadata)


@given(selection_axis=st.sampled_from(["X", "Y", "Z"]), ds=sgrid_dataset(), slice_=st.slices(4))
def test_isel(selection_axis, ds, slice_):
    # TODO: Add skip if Z but no Z dimension in ds

    # select nodes in axis direction
    # assert consistent

    # select edges
    # assert consistent
    ...
