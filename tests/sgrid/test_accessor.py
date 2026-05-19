import hypothesis.strategies as st
import pytest
import xarray as xr
from hypothesis import assume, given

import parcels._strategies as pst
from parcels._datasets.structured.generic import datasets_sgrid
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


@given(ds=sgrid_dataset(), dim=st.sampled_from(["face_dimension1", "face_dimension2", "vertical_dimension"]))
def test_assert_metadata_ds_consistency_dropped_dim(ds, dim):
    # dropping one of the SGRID dimensions still results in a consistent dataset
    metadata: SGrid2DMetadata = ds.sgrid.metadata

    if dim == "face_dimension1":
        fnp = metadata.face_dimensions[0]
    elif dim == "face_dimension2":
        fnp = metadata.face_dimensions[1]
    elif dim == "vertical_dimension":
        assume(metadata.vertical_dimensions is not None)
        assert metadata.vertical_dimensions is not None
        fnp = metadata.vertical_dimensions[0]
    else:
        raise ValueError("Unexpected value for dim")

    assume(fnp.face in ds.dims)

    ds = ds.isel({fnp.face: 0})
    assert_metadata_ds_consistency(ds, metadata)


@given(ds=sgrid_dataset(), dim=st.sampled_from(["face_dimension1", "face_dimension2", "vertical_dimension"]))
def test_assert_metadata_ds_consistency_failures(ds, dim):
    metadata: SGrid2DMetadata = ds.sgrid.metadata

    if dim == "face_dimension1":
        fnp = metadata.face_dimensions[0]
    elif dim == "face_dimension2":
        fnp = metadata.face_dimensions[1]
    elif dim == "vertical_dimension":
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


@pytest.mark.parametrize("ds", [pytest.param(ds, id=id_) for id_, ds in datasets_sgrid.items()])
@pytest.mark.parametrize("slice_", [slice(None, None, 3), slice(2, -3)])
@pytest.mark.parametrize(
    "node_dim, face_dim", [("node_dimension1", "face_dimension1"), ("node_dimension2", "face_dimension2")]
)
def test_isel(ds, slice_, node_dim, face_dim):
    # TODO: Extend to padding BOTH and NONE by updating datasets_sgrid

    assert ds.dims[node_dim] == ds.dims[face_dim]

    ds_trimmed = ds.sgrid.isel({node_dim: slice_})

    assert ds_trimmed.dims[node_dim] == ds_trimmed.dims[face_dim]

    # Assert that other dims haven't been affected
    for dim, size_before in ds.dims.items():
        if dim in (node_dim, face_dim):
            continue
        size_after = ds_trimmed.dims[dim]
        assert size_before == size_after


@pytest.mark.parametrize("ds", [datasets_sgrid["ds_2d_padded_high"]])
def test_isel_invalid(ds):
    with pytest.raises(ValueError, match="Cannot use SGRID accessor to .isel non-spatial \(/SGRID related\) dimension.*"):
        ds.sgrid.isel(time=slice(None))

    with pytest.raises(ValueError, match="Dims .* are on the same axis .* according to SGRID metadata.*"):
        ds.sgrid.isel(node_dimension1=slice(None), face_dimension1=slice(None))
