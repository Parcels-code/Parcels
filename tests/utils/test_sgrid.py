import itertools

import numpy as np
import pytest
import xarray as xr
import xgcm
from hypothesis import assume, example, given

from parcels._core.utils import sgrid
from tests.strategies import sgrid as sgrid_strategies


def create_example_grid2dmetadata(with_vertical_dimensions: bool, with_node_coordinates: bool):
    vertical_dimensions = (
        (sgrid.DimDimPadding("vertical_dimensions_dim1", "vertical_dimensions_dim2", sgrid.Padding.LOW),)
        if with_vertical_dimensions
        else None
    )
    node_coordinates = ("node_coordinates_var1", "node_coordinates_var2") if with_node_coordinates else None

    return sgrid.Grid2DMetadata(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions=("node_dimension1", "node_dimension2"),
        face_dimensions=(
            sgrid.DimDimPadding("face_dimension1", "node_dimension1", sgrid.Padding.LOW),
            sgrid.DimDimPadding("face_dimension2", "node_dimension2", sgrid.Padding.LOW),
        ),
        node_coordinates=node_coordinates,
        vertical_dimensions=vertical_dimensions,
    )


def create_example_grid3dmetadata(with_node_coordinates: bool):
    node_coordinates = (
        ("node_coordinates_var1", "node_coordinates_var2", "node_coordinates_dim3") if with_node_coordinates else None
    )
    return sgrid.Grid3DMetadata(
        cf_role="grid_topology",
        topology_dimension=3,
        node_dimensions=("node_dimension1", "node_dimension2", "node_dimension3"),
        volume_dimensions=(
            sgrid.DimDimPadding("face_dimension1", "node_dimension1", sgrid.Padding.LOW),
            sgrid.DimDimPadding("face_dimension2", "node_dimension2", sgrid.Padding.LOW),
            sgrid.DimDimPadding("face_dimension3", "node_dimension3", sgrid.Padding.LOW),
        ),
        node_coordinates=node_coordinates,
    )


grid2dmetadata = create_example_grid2dmetadata(with_vertical_dimensions=True, with_node_coordinates=True)
grid3dmetadata = create_example_grid3dmetadata(with_node_coordinates=True)


@pytest.mark.parametrize(
    ("sgrid_metadata", "id", "value"),
    [
        (grid2dmetadata, "node_dimension1", "node_dimension1"),
        (grid2dmetadata, "node_dimension2", "node_dimension2"),
        (grid2dmetadata, "face_dimension1", "face_dimension1"),
        (grid2dmetadata, "face_dimension2", "face_dimension2"),
        # (grid2dmetadata, "vertical_dimensions_dim1", "vertical_dimensions_dim1"), #! ID's NOT IMPLEMENTED IN SGRID SPEC
        # (grid2dmetadata, "vertical_dimensions_dim2", "vertical_dimensions_dim2"),
        (grid3dmetadata, "node_dimension1", "node_dimension1"),
        (grid3dmetadata, "node_dimension2", "node_dimension2"),
        (grid3dmetadata, "node_dimension3", "node_dimension3"),
        (grid3dmetadata, "face_dimension1", "face_dimension1"),
        (grid3dmetadata, "face_dimension2", "face_dimension2"),
        (grid3dmetadata, "face_dimension3", "face_dimension3"),
        (grid3dmetadata, "type1", sgrid.Padding.LOW),
        (grid3dmetadata, "type2", sgrid.Padding.LOW),
        (grid3dmetadata, "type3", sgrid.Padding.LOW),
    ],
)
def test_get_value_by_id(sgrid_metadata: sgrid.Grid2DMetadata | sgrid.Grid3DMetadata, id, value):
    assert sgrid_metadata.get_value_by_id(id) == value


def dummy_sgrid_ds(grid: sgrid.Grid2DMetadata | sgrid.Grid3DMetadata) -> xr.Dataset:
    if isinstance(grid, sgrid.Grid2DMetadata):
        return dummy_sgrid_2d_ds(grid)
    elif isinstance(grid, sgrid.Grid3DMetadata):
        return dummy_sgrid_3d_ds(grid)
    else:
        raise NotImplementedError(f"Cannot create dummy SGrid dataset for grid type {type(grid)}")


def dummy_sgrid_2d_ds(grid: sgrid.Grid2DMetadata) -> xr.Dataset:
    ds = dummy_comodo_3d_ds()

    # Can't rename dimensions that already exist in the dataset
    assume(sgrid.get_unique_dim_names(grid) & set(ds.dims) == set())

    renamings = {}
    if grid.vertical_dimensions is None:
        ds = ds.isel(ZC=0, ZG=0)
    else:
        renamings.update({"ZC": grid.vertical_dimensions[0].dim2, "ZG": grid.vertical_dimensions[0].dim1})

    for old, new in zip(["XG", "YG"], grid.node_dimensions, strict=True):
        renamings[old] = new

    for old, dim_dim_padding in zip(["XC", "YC"], grid.face_dimensions, strict=True):
        renamings[old] = dim_dim_padding.dim1

    ds = ds.rename_dims(renamings)

    ds["grid"] = xr.DataArray(1, attrs=grid.to_attrs())
    ds.attrs["convention"] = "SGRID"
    return ds


def dummy_sgrid_3d_ds(grid: sgrid.Grid3DMetadata) -> xr.Dataset:
    ds = dummy_comodo_3d_ds()

    # Can't rename dimensions that already exist in the dataset
    assume(sgrid.get_unique_dim_names(grid) & set(ds.dims) == set())

    renamings = {}
    for old, new in zip(["XG", "YG", "ZG"], grid.node_dimensions, strict=True):
        renamings[old] = new

    for old, dim_dim_padding in zip(["XC", "YC", "ZC"], grid.volume_dimensions, strict=True):
        renamings[old] = dim_dim_padding.dim1

    ds = ds.rename_dims(renamings)

    ds["grid"] = xr.DataArray(1, attrs=grid.to_attrs())
    ds.attrs["convention"] = "SGRID"
    return ds


def dummy_comodo_3d_ds() -> xr.Dataset:
    T, Z, Y, X = 7, 6, 5, 4
    TIME = xr.date_range("2000", "2001", T)
    return xr.Dataset(
        {
            "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(T, Z, Y, X)),
            "U_A_grid": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "V_A_grid": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "U_C_grid": (["time", "ZG", "YC", "XG"], np.random.rand(T, Z, Y, X)),
            "V_C_grid": (["time", "ZG", "YG", "XC"], np.random.rand(T, Z, Y, X)),
        },
        coords={
            # "XG": (
            #     ["XG"],
            #     2 * np.pi / X * np.arange(0, X),
            #     {"axis": "X", "c_grid_axis_shift": -0.5},
            # ),
            # "XC": (["XC"], 2 * np.pi / X * (np.arange(0, X) + 0.5), {"axis": "X"}),
            # "YG": (
            #     ["YG"],
            #     2 * np.pi / (Y) * np.arange(0, Y),
            #     {"axis": "Y", "c_grid_axis_shift": -0.5},
            # ),
            # "YC": (
            #     ["YC"],
            #     2 * np.pi / (Y) * (np.arange(0, Y) + 0.5),
            #     {"axis": "Y"},
            # ),
            # "ZG": (
            #     ["ZG"],
            #     np.arange(Z),
            #     {"axis": "Z", "c_grid_axis_shift": -0.5},
            # ),
            # "ZC": (
            #     ["ZC"],
            #     np.arange(Z) + 0.5,
            #     {"axis": "Z"},
            # ),
            # "lon": (["XG"], 2 * np.pi / X * np.arange(0, X)),
            # "lat": (["YG"], 2 * np.pi / (Y) * np.arange(0, Y)),
            # "depth": (["ZG"], np.arange(Z)),
            "time": (["time"], TIME, {"axis": "T"}),
        },
    )


@example(
    edge_node_padding=(
        sgrid.DimDimPadding("edge1", "node1", sgrid.Padding.NONE),
        sgrid.DimDimPadding("edge2", "node2", sgrid.Padding.LOW),
    )
)
@given(sgrid_strategies.mappings)
def test_edge_node_mapping_metadata_roundtrip(edge_node_padding):
    serialized = sgrid.dump_mappings(edge_node_padding)
    parsed = sgrid.load_mappings(serialized)
    assert parsed == edge_node_padding


@pytest.mark.parametrize(
    "input_, expected",
    [
        (
            "edge1: node1(padding: none)",
            (sgrid.DimDimPadding("edge1", "node1", sgrid.Padding.NONE),),
        ),
    ],
)
def test_load_dump_mappings(input_, expected):
    assert sgrid.load_mappings(input_) == expected


@example(grid2dmetadata)
@given(sgrid_strategies.grid2Dmetadata())
def test_Grid2DMetadata_roundtrip(grid: sgrid.Grid2DMetadata):
    attrs = grid.to_attrs()
    parsed = sgrid.Grid2DMetadata.from_attrs(attrs)
    assert parsed == grid


@example(grid3dmetadata)
@given(sgrid_strategies.grid3Dmetadata())
def test_Grid3DMetadata_roundtrip(grid: sgrid.Grid3DMetadata):
    attrs = grid.to_attrs()
    parsed = sgrid.Grid3DMetadata.from_attrs(attrs)
    assert parsed == grid


@given(sgrid_strategies.grid_metadata)
def test_parse_grid_attrs(grid: sgrid.AttrsSerializable):
    attrs = grid.to_attrs()
    parsed = sgrid.parse_grid_attrs(attrs)
    assert parsed == grid


@example(grid2dmetadata)
@given(sgrid_strategies.grid2Dmetadata())
def test_parse_sgrid_2d(grid_metadata: sgrid.Grid2DMetadata):
    """Test the ingestion of datasets in XGCM to ensure that it matches the SGRID metadata provided"""
    ds = dummy_sgrid_2d_ds(grid_metadata)

    _, xgcm_kwargs = sgrid.parse_sgrid(ds)
    grid = xgcm.Grid(ds, autoparse_metadata=False, **xgcm_kwargs)

    for ddp, axis in zip(grid_metadata.face_dimensions, ["X", "Y"], strict=True):
        dim_edge, dim_node, padding = ddp.dim1, ddp.dim2, ddp.padding
        coords = grid.axes[axis].coords
        assert coords["center"] == dim_edge
        assert coords[sgrid.SGRID_PADDING_TO_XGCM_POSITION[padding]] == dim_node

    if grid_metadata.vertical_dimensions is None:
        assert "Z" not in grid.axes
    else:
        ddp = grid_metadata.vertical_dimensions[0]
        dim_edge, dim_node, padding = ddp.dim1, ddp.dim2, ddp.padding
        coords = grid.axes["Z"].coords
        assert coords["center"] == dim_edge
        assert coords[sgrid.SGRID_PADDING_TO_XGCM_POSITION[padding]] == dim_node


@given(sgrid_strategies.grid3Dmetadata())
def test_parse_sgrid_3d(grid_metadata: sgrid.Grid3DMetadata):
    """Test the ingestion of datasets in XGCM to ensure that it matches the SGRID metadata provided"""
    ds = dummy_sgrid_3d_ds(grid_metadata)

    ds, xgcm_kwargs = sgrid.parse_sgrid(ds)
    grid = xgcm.Grid(ds, autoparse_metadata=False, **xgcm_kwargs)

    for ddp, axis in zip(grid_metadata.volume_dimensions, ["X", "Y", "Z"], strict=True):
        dim_edge, dim_node, padding = ddp.dim1, ddp.dim2, ddp.padding
        coords = grid.axes[axis].coords
        assert coords["center"] == dim_edge
        assert coords[sgrid.SGRID_PADDING_TO_XGCM_POSITION[padding]] == dim_node


@pytest.mark.parametrize(
    "grid",
    [
        create_example_grid2dmetadata(with_node_coordinates=i, with_vertical_dimensions=j)
        for i, j in itertools.product([False, True], [False, True])
    ]
    + [create_example_grid3dmetadata(with_node_coordinates=i) for i in [False, True]],
)
def test_rename_dims(grid):
    dims = sgrid.get_unique_dim_names(grid)
    dims_dict = {dim: f"new_{dim}" for dim in dims}
    dims_dict_inv = {v: k for k, v in dims_dict.items()}

    grid_new = grid.rename_dims(dims_dict)
    assert dims & set(sgrid.get_unique_dim_names(grid_new)) == set()

    assert grid == grid_new.rename_dims(dims_dict_inv)


def test_rename_dims_errors():
    # Test various error modes of rename_dims
    grid = grid2dmetadata
    # Non-unique target dimension names
    dims_dict = {
        "node_dimension1": "new_node_dimension",
        "node_dimension2": "new_node_dimension",
    }
    with pytest.raises(AssertionError, match="dims_dict contains duplicate target dimension names"):
        grid.rename_dims(dims_dict)
    # Unexpected attribute in dims_dict
    dims_dict = {
        "unexpected_dimension": "new_unexpected_dimension",
    }
    with pytest.raises(ValueError, match="Dimension 'unexpected_dimension' not found in SGrid metadata dimensions"):
        grid.rename_dims(dims_dict)
