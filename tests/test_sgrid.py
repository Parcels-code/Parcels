from hypothesis import given

from parcels._core import sgrid
from tests.strategies import sgrid as sgrid_strategies


@given(sgrid_strategies.edge_node_padding_list(min_size=1, max_size=3))
def test_edge_node_mapping_metadata_roundtrip(edge_node_padding):
    serialized = sgrid.serialize_edge_node_mapping(edge_node_padding)
    parsed = sgrid.parse_edge_node_mapping(serialized)
    assert parsed == edge_node_padding
