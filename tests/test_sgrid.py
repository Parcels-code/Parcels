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
