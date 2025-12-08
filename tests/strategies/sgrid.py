"""Provides Hypothesis strategies to help testing the parsing and serialization of datasets
According to the SGrid conventions.

https://sgrid.github.io/sgrid/

Note that these strategies don't aim to completely cover the SGrid conventions, but aim to
cover SGrid to the extent to which Parcels is concerned.
"""

import string

from hypothesis import strategies as st

from parcels._core.sgrid import DimDimPadding
from parcels._core.sgrid import Padding as PaddingEnum

ALLOWED_DIM_LETTERS = (
    string.ascii_letters + string.digits + "_"
)  # We can make this more aligned with SGrid by adjusting our regex - but this is good for now

padding = st.sampled_from(PaddingEnum)
dimension_name = st.text(
    min_size=1, alphabet=st.characters(categories=(), whitelist_characters=ALLOWED_DIM_LETTERS)
).filter(lambda s: " " not in s)  # assuming for now spaces are allowed in dimension names in SGrid convention
dim_dim_padding = (
    st.tuples(dimension_name, dimension_name, padding).filter(lambda t: t[0] != t[1]).map(lambda t: DimDimPadding(*t))
)

mappings = st.lists(dim_dim_padding | dimension_name).map(tuple)
