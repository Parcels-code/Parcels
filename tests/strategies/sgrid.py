"""Provides Hypothesis strategies to help testing the parsing and serialization of datasets
According to the SGrid conventions.

https://sgrid.github.io/sgrid/

Note that these strategies don't aim to completely cover the SGrid conventions, but aim to
cover SGrid to the extent to which Parcels is concerned.
"""

import string

from hypothesis import strategies as st

from parcels._core.sgrid import Padding as PaddingEnum

ALLOWED_DIM_LETTERS = string.ascii_letters + "-_"

padding = st.sampled_from(PaddingEnum)
dimension_name = st.text(
    min_size=1, alphabet=st.characters(categories=(), whitelist_characters=ALLOWED_DIM_LETTERS)
).filter(lambda s: " " not in s)  # assuming for now spaces are allowed in dimension names in SGrid convention
edge_node_padding_tuple = st.tuples(dimension_name, dimension_name, padding).filter(lambda t: t[0] != t[1])


@st.composite
def edge_node_padding_list(draw, min_size, max_size):
    ret = []
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    for _ in range(n):
        used_edges_and_nodes = set(e for e, _, _ in ret).union(n for _, n, _ in ret)

        def is_used_name(d, used_edges_and_nodes=used_edges_and_nodes):
            return d in used_edges_and_nodes

        new = draw(edge_node_padding_tuple.filter(lambda t: not (is_used_name(t[0]) or is_used_name(t[1]))))
        ret.append(new)
    return ret
