import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given

import parcels._sgrid as sgrid
import parcels._strategies as pst
from parcels._datasets.structured.generic import datasets_sgrid
from parcels._datasets.structured.strategies import sgrid_dataset
from parcels._sgrid.accessor import SGridDatasetInconsistency, assert_metadata_ds_consistency


@st.composite
def grid_and_dataset(draw) -> tuple[sgrid.SGrid2DMetadata, xr.Dataset]:
    # used only for test_metadata - for all other tests we can simply do `ds.sgrid.metadata` to get the metadata
    metadata_2d = draw(
        pst.sgrid.grid_metadata.filter(
            # parcels can only generate 2D Sgrid datasets, that also have coordinates
            lambda meta: isinstance(meta, sgrid.SGrid2DMetadata) and meta.node_coordinates is not None
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


@pytest.mark.parametrize(
    "ds",
    [
        xr.Dataset(
            {
                "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(10, 10, 10, 10)),
                "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(10, 10, 10, 10)),
                "grid": (
                    [],
                    np.array(0),
                    sgrid.SGrid2DMetadata(
                        cf_role="grid_topology",
                        topology_dimension=2,
                        node_dimensions=("XG", "YG"),
                        face_dimensions=(
                            sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.HIGH),
                            sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.HIGH),
                        ),
                        vertical_dimensions=(sgrid.FaceNodePadding("ZC", "ZG", sgrid.Padding.HIGH),),
                        node_coordinates=("lon", "lat"),
                    ).to_attrs(),
                ),
            },
            coords={
                "lon": (["XG"], 2 * np.pi / 10 * np.arange(0, 10)),
                "lat": (["YG"], 2 * np.pi / (10) * np.arange(0, 10)),
                "depth": (["ZG"], np.arange(10)),
                "time": (["time"], xr.date_range("2000", "2001", 10), {"axis": "T"}),
            },
        ),
    ],
)
def test_rename_dataset(ds):
    # Check renaming works for coordinates
    ds_new = ds.sgrid.rename({"lon": "lon_updated"})
    grid_new = ds_new.sgrid.metadata
    assert "lon_updated" in ds_new.coords
    assert "lon_updated" == grid_new.node_coordinates[0]

    # Check renaming works for dim
    ds_new = ds.sgrid.rename({"XC": "XC_updated"})
    grid_new = ds_new.sgrid.metadata
    assert "XC_updated" in ds_new.dims
    assert "XC" not in ds_new.dims
    assert "XC_updated" == grid_new.face_dimensions[0].face


@given(sgrid_dataset())
def test_assert_metadata_ds_consistency(ds):
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    assert_metadata_ds_consistency(ds, metadata)


@given(ds=sgrid_dataset(), dim=st.sampled_from(["face_dimension1", "face_dimension2", "vertical_dimension"]))
def test_assert_metadata_ds_consistency_dropped_dim(ds, dim):
    # dropping one of the SGRID dimensions still results in a consistent dataset
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata

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
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata

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
@pytest.mark.parametrize("indexer", [slice(None, None, 3), slice(2, -3), [0]])
@pytest.mark.parametrize(
    "node_dim, face_dim", [("node_dimension1", "face_dimension1"), ("node_dimension2", "face_dimension2")]
)
def test_isel(ds, indexer, node_dim, face_dim):
    # TODO: Extend to padding BOTH and NONE by updating datasets_sgrid
    # TODO: Expand testing on types of indexers

    assert ds.dims[node_dim] == ds.dims[face_dim]

    ds_trimmed = ds.sgrid.isel({node_dim: indexer})

    assert ds_trimmed.dims[node_dim] == ds_trimmed.dims[face_dim]

    # Assert that other dims haven't been affected
    for dim, size_before in ds.dims.items():
        if dim in (node_dim, face_dim):
            continue
        size_after = ds_trimmed.dims[dim]
        assert size_before == size_after


@pytest.mark.parametrize("ds", [pytest.param(ds, id=id_) for id_, ds in datasets_sgrid.items()])
def test_isel_drop_dim(ds):
    ds = ds.copy()
    assert ds.dims["node_dimension1"] == ds.dims["face_dimension1"]

    ds_trimmed = ds.sgrid.isel({"node_dimension1": 0})

    assert "node_dimension1" not in ds_trimmed.dims
    assert "face_dimension1" not in ds_trimmed.dims

    # Assert that other dims haven't been affected
    for dim, size_after in ds_trimmed.dims.items():
        size_before = ds.dims[dim]
        assert size_before == size_after


@pytest.mark.parametrize("ds", [datasets_sgrid["ds_2d_padded_high"]])
def test_isel_invalid(ds):
    with pytest.raises(
        ValueError, match=r"Cannot use SGRID accessor to .isel non-spatial \(/SGRID related\) dimension.*"
    ):
        ds.sgrid.isel(time=slice(None))

    with pytest.raises(ValueError, match="Dims .* are on the same axis .* according to SGRID metadata.*"):
        ds.sgrid.isel(node_dimension1=slice(None), face_dimension1=slice(None))
