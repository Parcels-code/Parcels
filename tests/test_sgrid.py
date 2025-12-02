import pytest
from hypothesis import example, given

from parcels._core import sgrid
from tests.strategies import sgrid as sgrid_strategies


@example(
    edge_node_padding=[
        ("edge1", "node1", sgrid.Padding.NONE),
        ("edge2", "node2", sgrid.Padding.LOW),
    ]
)
@given(sgrid_strategies.edge_node_padding_list(min_size=1, max_size=3))
def test_edge_node_mapping_metadata_roundtrip(edge_node_padding):
    serialized = sgrid.serialize_edge_node_mapping(edge_node_padding)
    parsed = sgrid.parse_edge_node_mapping(serialized)
    assert parsed == edge_node_padding


@pytest.mark.parametrize(
    "input_, expected",
    [
        (
            "edge1: node1(padding: none)",
            [("edge1", "node1", sgrid.Padding.NONE)],
        ),
    ],
)
def test_parse_edge_node_mapping(input_, expected):
    assert sgrid.parse_edge_node_mapping(input_) == expected
