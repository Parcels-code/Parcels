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


@given(ds=sgrid_dataset(), dim=st.sampled_from(['face_dimension1', "face_dimension2", "vertical_dimension"]))
def test_assert_metadata_ds_consistency_dropped_dim(ds, dim):
    # dropping one of the SGRID dimensions still results in a consistent dataset
    metadata: SGrid2DMetadata = ds.sgrid.metadata

    if dim == "face_dimension1":
        fnp = metadata.face_dimensions[0]
    elif dim=="face_dimension2":
        fnp = metadata.face_dimensions[1]
    elif dim=="vertical_dimension":
        assume(metadata.vertical_dimensions is not None)
        assert metadata.vertical_dimensions is not None
        fnp = metadata.vertical_dimensions[0]
    else:
        raise ValueError("Unexpected value for dim")

    assume(fnp.face in ds.dims)

    ds = ds.isel({fnp.face: 0})
    assert_metadata_ds_consistency(ds, metadata)


@given(ds=sgrid_dataset(), dim=st.sampled_from(['face_dimension1', "face_dimension2", "vertical_dimension"]))
def test_assert_metadata_ds_consistency_failures(ds, dim):
    metadata: SGrid2DMetadata = ds.sgrid.metadata

    if dim == "face_dimension1":
        fnp = metadata.face_dimensions[0]
    elif dim=="face_dimension2":
        fnp = metadata.face_dimensions[1]
    elif dim=="vertical_dimension":
        assume(metadata.vertical_dimensions is not None)
        assert metadata.vertical_dimensions is not None
        fnp = metadata.vertical_dimensions[0]
    else:
        raise ValueError("Unexpected value for dim")

    assume(fnp.node in ds.dims)
    assume(fnp.face in ds.dims)

    ds = ds.isel({fnp.face: slice(None, -1)})

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
