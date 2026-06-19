import warnings

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis.errors import NonInteractiveExampleWarning

import parcels._sgrid as sgrid
from parcels._datasets.structured.strategies import sgrid_dataset


def test_sgrid_dataset_raises_when_no_node_coordinates():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        no_coords_grid = sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
            node_coordinates=None,
        )
        with pytest.raises(ValueError, match="node_coordinates"):
            sgrid_dataset(grid=no_coords_grid).example()


@given(sgrid_dataset())
@settings(max_examples=20)
def test_sgrid_dataset_returns_dataset(ds):
    assert isinstance(ds, xr.Dataset)


@given(sgrid_dataset())
@settings(max_examples=20)
def test_sgrid_dataset_has_grid_topology(ds):
    ds.sgrid._get_grid_topology()  # shouldn't error


@given(sgrid_dataset())
@settings(max_examples=20)
def test_sgrid_dataset_node_coordinates_present(ds):
    meta = ds.sgrid.metadata
    assert meta.node_coordinates is not None
    for coord_name in meta.node_coordinates:
        assert coord_name in ds.coords


@given(sgrid_dataset())
@settings(max_examples=20)
def test_sgrid_dataset_coordinate_shapes(ds):
    meta = ds.sgrid.metadata
    coord_name1, coord_name2 = meta.node_coordinates
    node_dim1, node_dim2 = meta.node_dimensions
    coord1 = ds.coords[coord_name1]
    coord2 = ds.coords[coord_name2]
    assert coord1.dims in [(node_dim1,), (node_dim1, node_dim2)]
    assert coord2.dims in [(node_dim2,), (node_dim1, node_dim2)]
    if len(coord1.dims) == 2:
        assert len(coord2.dims) == 2


@given(sgrid_dataset())
@settings(max_examples=20)
def test_sgrid_dataset_has_at_least_one_field(ds):
    non_grid_vars = [v for v in ds.data_vars if v != "grid"]
    assert len(non_grid_vars) >= 1


@given(sgrid_dataset())
@settings(max_examples=20)
def test_sgrid_dataset_field_dims_are_valid(ds):
    meta = ds.sgrid.metadata
    valid_dims = set(meta.node_dimensions)
    valid_dims.add(meta.face_dimensions[0].face)
    valid_dims.add(meta.face_dimensions[1].face)
    if meta.vertical_dimensions is not None:
        valid_dims.add(meta.vertical_dimensions[0].node)
        valid_dims.add(meta.vertical_dimensions[0].face)

    for var_name, var in ds.data_vars.items():
        if var_name == "grid":
            continue
        for dim in var.dims:
            assert dim in valid_dims, f"Field {var_name!r} has unexpected dim {dim!r}"


@given(sgrid_dataset())
@settings(max_examples=20)
def test_sgrid_dataset_no_nan_in_fields(ds):
    for var_name, var in ds.data_vars.items():
        if var_name == "grid":
            continue
        assert not np.any(np.isnan(var.values)), f"NaN found in field {var_name!r}"
