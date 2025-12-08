import pytest
from hypothesis import example, given

from parcels._core import sgrid
from tests.strategies import sgrid as sgrid_strategies


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
