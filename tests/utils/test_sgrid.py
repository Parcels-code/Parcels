import numpy as np
import pytest
import xarray as xr
import xgcm
from hypothesis import assume, example, given

from parcels._core.utils import sgrid
from tests.strategies import sgrid as sgrid_strategies


def get_unique_dim_names(grid: sgrid.Grid2DMetadata | sgrid.Grid3DMetadata) -> set[str]:
    dims = set()
    dims.update(set(grid.node_dimensions))

    for value in [
        grid.node_dimensions,
        grid.face_dimensions if isinstance(grid, sgrid.Grid2DMetadata) else grid.volume_dimensions,
        grid.vertical_dimensions if isinstance(grid, sgrid.Grid2DMetadata) else None,
    ]:
        if value is None:
            continue
        for item in value:
            if isinstance(item, sgrid.DimDimPadding):
                dims.add(item.dim1)
                dims.add(item.dim2)
            else:
                assert isinstance(item, str)
                dims.add(item)
    return dims


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
    assume(get_unique_dim_names(grid) & set(ds.dims) == set())

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
    assume(get_unique_dim_names(grid) & set(ds.dims) == set())

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


@example(
    grid=sgrid.Grid2DMetadata(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions=("node_dimension1", "node_dimension2"),
        face_dimensions=(
            sgrid.DimDimPadding("face_dimension1", "node_dimension1", sgrid.Padding.LOW),
            sgrid.DimDimPadding("face_dimension2", "node_dimension2", sgrid.Padding.LOW),
        ),
        vertical_dimensions=(
            sgrid.DimDimPadding("vertical_dimensions_dim1", "vertical_dimensions_dim2", sgrid.Padding.LOW),
        ),
    )
)
@given(sgrid_strategies.grid2Dmetadata())
def test_Grid2DMetadata_roundtrip(grid: sgrid.Grid2DMetadata):
    attrs = grid.to_attrs()
    parsed = sgrid.Grid2DMetadata.from_attrs(attrs)
    assert parsed == grid


@example(
    grid=sgrid.Grid3DMetadata(
        cf_role="grid_topology",
        topology_dimension=3,
        node_dimensions=("node_dimension1", "node_dimension2", "node_dimension3"),
        volume_dimensions=(
            sgrid.DimDimPadding("face_dimension1", "node_dimension1", sgrid.Padding.LOW),
            sgrid.DimDimPadding("face_dimension2", "node_dimension2", sgrid.Padding.LOW),
            sgrid.DimDimPadding("face_dimension3", "node_dimension3", sgrid.Padding.LOW),
        ),
    )
)
@given(sgrid_strategies.grid3Dmetadata())
def test_Grid3DMetadata_roundtrip(grid: sgrid.Grid3DMetadata):
    attrs = grid.to_attrs()
    parsed = sgrid.Grid3DMetadata.from_attrs(attrs)
    assert parsed == grid


@given(sgrid_strategies.grid_metadata)
def test_parse_grid_attrs(grid: sgrid.SGridMetadataProtocol):
    attrs = grid.to_attrs()
    parsed = sgrid.parse_grid_attrs(attrs)
    assert parsed == grid


@given(sgrid_strategies.grid2Dmetadata())
def test_parse_sgrid_2d(grid_metadata: sgrid.Grid2DMetadata):
    """Test the ingestion of datasets in XGCM to ensure that it matches the SGRID metadata provided"""
    ds = dummy_sgrid_2d_ds(grid_metadata)

    ds, xgcm_kwargs = sgrid.parse_sgrid(ds)
    grid = xgcm.Grid(ds, autoparse_metadata=False, **xgcm_kwargs)

    for ddp, axis in zip(grid_metadata.face_dimensions, ["X", "Y"], strict=True):
        dim_node, dim_edge, padding = ddp.dim1, ddp.dim2, ddp.padding
        coords = grid.axes[axis].coords
        assert coords["center"] == dim_edge
        assert coords[sgrid.SGRID_PADDING_TO_XGCM_POSITION[padding]] == dim_node

    if grid_metadata.vertical_dimensions is None:
        assert "Z" not in grid.axes
    else:
        ddp = grid_metadata.vertical_dimensions[0]
        dim_node, dim_edge, padding = ddp.dim1, ddp.dim2, ddp.padding
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
        dim_node, dim_edge, padding = ddp.dim1, ddp.dim2, ddp.padding
        coords = grid.axes[axis].coords
        assert coords["center"] == dim_edge
        assert coords[sgrid.SGRID_PADDING_TO_XGCM_POSITION[padding]] == dim_node
