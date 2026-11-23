import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given

import parcels._sgrid as sgrid
import parcels._strategies as pst
from parcels._datasets.structured.generic import datasets_sgrid
from parcels._datasets.structured.strategies import sgrid_dataset
from parcels._sgrid.accessor import SGridDatasetInconsistency


@st.composite
def grid_and_dataset(draw) -> tuple[sgrid.SGrid2DMetadata, xr.Dataset]:
    # used only for test_metadata - for all other tests we can simply do `ds.sgrid.metadata` to get the metadata
    metadata_2d = draw(
        pst.sgrid.grid_metadata.filter(
            # Parcels can only generate 2D Sgrid datasets, that also have coordinates
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
    ds.sgrid.assert_metadata_ds_consistency()


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
    ds.sgrid.assert_metadata_ds_consistency()


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
        ds.sgrid.assert_metadata_ds_consistency()


_SYMMETRIC_DATASETS = {k: v for k, v in datasets_sgrid.items() if k in ("ds_2d_padded_high", "ds_2d_padded_low")}


@given(ds=sgrid_dataset(), axis=st.sampled_from([None, "Y", "Z"]))
def test_assert_metadata_ds_consistency_node_dimension_axis(ds, axis):
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    node_dim1 = metadata.node_dimensions[0]
    if axis is None:
        del ds[node_dim1].attrs["axis"]
    else:
        ds[node_dim1].attrs["axis"] = axis

    with pytest.raises(SGridDatasetInconsistency, match=r"expected variable .* to have attribute axis='X'"):
        ds.sgrid.assert_metadata_ds_consistency()


@given(ds=sgrid_dataset())
def test_assert_metadata_ds_consistency_node_dimensions_swapped(ds):
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    node_dim1, node_dim2 = metadata.node_dimensions
    ds[node_dim1].attrs["axis"], ds[node_dim2].attrs["axis"] = "Y", "X"

    with pytest.raises(SGridDatasetInconsistency, match=r"expected variable .* to have attribute axis='X'"):
        ds.sgrid.assert_metadata_ds_consistency()


@given(ds=sgrid_dataset())
def test_assert_metadata_ds_consistency_node_dimension_without_coordinate(ds):
    # a dimension without a coordinate variable has no axis attribute to check
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    ds = ds.drop_vars(metadata.node_dimensions[0])
    ds.sgrid.assert_metadata_ds_consistency()


@given(ds=sgrid_dataset())
def test_assert_metadata_ds_consistency_missing_node_coordinate(ds):
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    assert metadata.node_coordinates is not None
    ds = ds.drop_vars(metadata.node_coordinates[0])

    with pytest.raises(SGridDatasetInconsistency, match=r"but there is no variable .* in the dataset"):
        ds.sgrid.assert_metadata_ds_consistency()


@given(ds=sgrid_dataset())
def test_assert_metadata_ds_consistency_node_coordinate_on_face(ds):
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    assert metadata.node_coordinates is not None
    coord = metadata.node_coordinates[0]
    face_dim = metadata.face_dimensions[0].face
    assume(face_dim in ds.dims)
    ds[coord] = xr.DataArray(np.arange(ds.sizes[face_dim], dtype=float), dims=[face_dim], attrs={"units": "m"})

    with pytest.raises(SGridDatasetInconsistency, match=r"node coordinates can only be defined on the node_dimensions"):
        ds.sgrid.assert_metadata_ds_consistency()


@pytest.mark.parametrize("ds", [pytest.param(ds, id=id_) for id_, ds in _SYMMETRIC_DATASETS.items()])
def test_assert_metadata_ds_consistency_1d_node_coordinates_swapped(ds):
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    assert metadata.node_coordinates is not None
    coord1, coord2 = metadata.node_coordinates
    ds = ds.assign_coords({coord1: ds[coord2].variable, coord2: ds[coord1].variable})

    with pytest.raises(SGridDatasetInconsistency, match=r"is 1D with dims .* expected to be defined on node dimension"):
        ds.sgrid.assert_metadata_ds_consistency()


@pytest.mark.parametrize(
    "x_units, y_units, valid",
    [
        ("degrees_east", "degrees_north", True),
        ("degree_E", "degreesN", True),
        ("degrees", "degree", True),
        ("m", "m", True),
        ("degrees_north", "degrees_east", False),
        ("degrees_east", "degrees_east", False),
        ("degrees_north", "degrees_north", False),
    ],
)
@pytest.mark.parametrize("ds", [pytest.param(ds, id=id_) for id_, ds in _SYMMETRIC_DATASETS.items()])
def test_assert_metadata_ds_consistency_node_coordinate_units(ds, x_units, y_units, valid):
    ds = ds.copy()
    metadata: sgrid.SGrid2DMetadata = ds.sgrid.metadata
    assert metadata.node_coordinates is not None
    coord1, coord2 = metadata.node_coordinates
    ds[coord1].attrs["units"] = x_units
    ds[coord2].attrs["units"] = y_units

    if valid:
        ds.sgrid.assert_metadata_ds_consistency()
    else:
        with pytest.raises(SGridDatasetInconsistency, match=r"Node coordinate .* has units=.*expected one of"):
            ds.sgrid.assert_metadata_ds_consistency()


@pytest.mark.parametrize("ds", [pytest.param(ds, id=id_) for id_, ds in _SYMMETRIC_DATASETS.items()])
@pytest.mark.parametrize("indexer", [slice(None, None, 3), [0]])
@pytest.mark.parametrize(
    "node_dim, face_dim", [("node_dimension1", "face_dimension1"), ("node_dimension2", "face_dimension2")]
)
def test_isel(ds, indexer, node_dim, face_dim):
    # Covers HIGH/LOW-specific indexer types (lists, step slices) not exercised by property tests.

    ds_trimmed = ds.sgrid.isel({node_dim: indexer})

    ds_trimmed.sgrid.assert_metadata_ds_consistency()

    # Assert that other dims haven't been affected
    for dim, size_before in ds.sizes.items():
        if dim in (node_dim, face_dim):
            continue
        size_after = ds_trimmed.sizes[dim]
        assert size_before == size_after


@pytest.mark.parametrize("ds", [pytest.param(ds, id=id_) for id_, ds in _SYMMETRIC_DATASETS.items()])
def test_isel_drop_dim(ds):
    ds = ds.copy()

    ds_trimmed = ds.sgrid.isel({"node_dimension1": 0})

    assert "node_dimension1" not in ds_trimmed.dims
    assert "face_dimension1" not in ds_trimmed.dims

    for dim, size_after in ds_trimmed.sizes.items():
        size_before = ds.sizes[dim]
        assert size_before == size_after


@pytest.mark.parametrize("ds", [datasets_sgrid["ds_2d_padded_high"]])
def test_isel_invalid(ds):
    with pytest.raises(
        ValueError, match=r"Cannot use SGRID accessor to .isel non-spatial \(/SGRID related\) dimension.*"
    ):
        ds.sgrid.isel(time=slice(None))

    with pytest.raises(ValueError, match="Dims .* are on the same axis .* according to SGRID metadata.*"):
        ds.sgrid.isel(node_dimension1=slice(None), face_dimension1=slice(None))


@st.composite
def valid_isel_slice(draw) -> slice:
    """Slice with no step; covers None, positive, and negative start/stop."""
    start = draw(st.one_of(st.none(), st.integers(-25, 25)))
    stop = draw(st.one_of(st.none(), st.integers(-25, 25)))
    return slice(start, stop)


@given(
    ds=sgrid_dataset(),
    s=valid_isel_slice(),
)
def test_isel_p1_consistency_invariant(ds, s):
    """After any valid isel, the SGRID face-node relationship is preserved."""
    metadata = ds.sgrid.metadata
    fnp = metadata.face_dimensions[0]
    node_dim = fnp.node
    assume(node_dim in ds.dims)
    n_nodes = ds.sizes[node_dim]
    assume(len(range(*s.indices(n_nodes))) > 0)  # exclude empty selections

    result = ds.sgrid.isel({node_dim: s})

    result.sgrid.assert_metadata_ds_consistency()


@given(
    ds=sgrid_dataset(),
    s=valid_isel_slice(),
)
def test_isel_p2_data_correctness(ds, s):
    """Values in the result match those obtained by applying the indexers directly via xarray."""
    metadata = ds.sgrid.metadata
    fnp = metadata.face_dimensions[0]
    node_dim, face_dim = fnp.node, fnp.face
    assume(node_dim in ds.dims)
    n_nodes = ds.sizes[node_dim]
    assume(len(range(*s.indices(n_nodes))) > 0)

    result = ds.sgrid.isel({node_dim: s})

    from parcels._sgrid.accessor import _derive_paired_indexer

    _, face_indexer = _derive_paired_indexer(s, indexer_is_node=True, padding=fnp.padding, dim_size=n_nodes)

    for var in ds.data_vars:
        if var == "grid":
            continue
        var_dims = ds[var].dims
        if node_dim in var_dims and face_dim not in var_dims:
            xr.testing.assert_equal(result[var], ds[var].isel({node_dim: s}))
        elif face_dim in var_dims and node_dim not in var_dims:
            if face_dim in ds.dims:
                xr.testing.assert_equal(result[var], ds[var].isel({face_dim: face_indexer}))


@given(
    ds=sgrid_dataset(),
    s=valid_isel_slice(),
)
def test_isel_p3_specification_symmetry(ds, s):
    """isel(node_dim=s) produces the same dataset as isel(face_dim=derived_face_slice)."""
    metadata = ds.sgrid.metadata
    fnp = metadata.face_dimensions[0]
    node_dim, face_dim = fnp.node, fnp.face
    assume(node_dim in ds.dims and face_dim in ds.dims)
    n_nodes = ds.sizes[node_dim]
    assume(len(range(*s.indices(n_nodes))) > 0)

    from parcels._sgrid.accessor import _derive_paired_indexer

    _, face_indexer = _derive_paired_indexer(s, indexer_is_node=True, padding=fnp.padding, dim_size=n_nodes)

    n_faces = ds.sizes.get(face_dim)
    if n_faces is not None:
        assume(len(range(*face_indexer.indices(n_faces))) > 0)

    result_via_node = ds.sgrid.isel({node_dim: s})
    result_via_face = ds.sgrid.isel({face_dim: face_indexer})

    xr.testing.assert_identical(result_via_node, result_via_face)


@pytest.mark.parametrize("ds_name", ["ds_2d_padded_none", "ds_2d_padded_both"])
@pytest.mark.parametrize(
    "indexer, match",
    [
        (5, "Scalar and list indexers are not supported for NONE/BOTH padding"),
        ([0, 1, 2], "Scalar and list indexers are not supported for NONE/BOTH padding"),
        (slice(0, 8, 2), "Slices with step != 1 are not supported for NONE/BOTH padding"),
    ],
)
def test_isel_asymmetric_padding_invalid_indexers(ds_name, indexer, match):
    ds = datasets_sgrid[ds_name]
    with pytest.raises(ValueError, match=match):
        ds.sgrid.isel({"node_dimension1": indexer})
