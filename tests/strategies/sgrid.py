"""Provides Hypothesis strategies to help testing the parsing and serialization of datasets
According to the SGrid conventions.

https://sgrid.github.io/sgrid/

Note that these strategies don't aim to completely cover the SGrid conventions, but aim to
cover SGrid to the extent to which Parcels is concerned.
"""

import xarray.testing.strategies as xr_st
from hypothesis import strategies as st

from parcels._core import sgrid

padding = st.sampled_from(sgrid.Padding)
dimension_name = xr_st.names().filter(
    lambda s: " " not in s
)  # assuming for now spaces are allowed in dimension names in SGrid convention
dim_dim_padding = (
    st.tuples(dimension_name, dimension_name, padding)
    .filter(lambda t: t[0] != t[1])
    .map(lambda t: sgrid.DimDimPadding(*t))
)

mappings = st.lists(dim_dim_padding | dimension_name).map(tuple)


@st.composite
def grid2Dmetadata(draw) -> sgrid.Grid2DMetadata:
    N = 6
    names = draw(st.lists(dimension_name, min_size=N, max_size=N, unique=True))
    node_dimension1 = names[0]
    node_dimension2 = names[1]
    face_dimension1 = names[2]
    face_dimension2 = names[3]
    padding_type1 = draw(padding)
    padding_type2 = draw(padding)

    vertical_dimensions_dim1 = names[4]
    vertical_dimensions_dim2 = names[5]
    vertical_dimensions_padding = draw(padding)
    has_vertical_dimensions = draw(st.booleans())

    if has_vertical_dimensions:
        vertical_dimensions = (
            sgrid.DimDimPadding(vertical_dimensions_dim1, vertical_dimensions_dim2, vertical_dimensions_padding),
        )
    else:
        vertical_dimensions = None

    return sgrid.Grid2DMetadata(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions=(node_dimension1, node_dimension2),
        face_dimensions=(
            sgrid.DimDimPadding(face_dimension1, node_dimension1, padding_type1),
            sgrid.DimDimPadding(face_dimension2, node_dimension2, padding_type2),
        ),
        vertical_dimensions=vertical_dimensions,
    )


@st.composite
def grid3Dmetadata(draw) -> sgrid.Grid3DMetadata:
    N = 6
    names = draw(st.lists(dimension_name, min_size=N, max_size=N, unique=True))
    node_dimension1 = names[0]
    node_dimension2 = names[1]
    node_dimension3 = names[2]
    face_dimension1 = names[3]
    face_dimension2 = names[4]
    face_dimension3 = names[5]
    padding_type1 = draw(padding)
    padding_type2 = draw(padding)
    padding_type3 = draw(padding)

    return sgrid.Grid3DMetadata(
        cf_role="grid_topology",
        topology_dimension=3,
        node_dimensions=(node_dimension1, node_dimension2, node_dimension3),
        volume_dimensions=(
            sgrid.DimDimPadding(face_dimension1, node_dimension1, padding_type1),
            sgrid.DimDimPadding(face_dimension2, node_dimension2, padding_type2),
            sgrid.DimDimPadding(face_dimension3, node_dimension3, padding_type3),
        ),
    )


grid_metadata = grid2Dmetadata() | grid3Dmetadata()
