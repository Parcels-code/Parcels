"""Provides Hypothesis strategies to help testing the parsing and serialization of datasets
According to the SGrid conventions.

This code is best read alongside the SGrid conventions documentation:
https://sgrid.github.io/sgrid/

Note this code doesn't aim to completely cover the SGrid conventions, but aim to
cover SGrid to the extent to which Parcels is concerned.
"""

import xarray.testing.strategies as xr_st
from hypothesis import strategies as st

from parcels._core.utils import sgrid

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
    N = 8
    names = draw(st.lists(dimension_name, min_size=N, max_size=N, unique=True))
    node_dimension1 = names[0]
    node_dimension2 = names[1]
    face_dimension1 = names[2]
    face_dimension2 = names[3]
    padding_type1 = draw(padding)
    padding_type2 = draw(padding)

    node_coordinates_dim1 = names[4]
    node_coordinates_dim2 = names[5]
    has_node_coordinates = draw(st.booleans())

    vertical_dimensions_dim1 = names[6]
    vertical_dimensions_dim2 = names[7]
    vertical_dimensions_padding = draw(padding)
    has_vertical_dimensions = draw(st.booleans())

    if has_node_coordinates:
        node_coordinates = (node_coordinates_dim1, node_coordinates_dim2)
    else:
        node_coordinates = None

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
        node_coordinates=node_coordinates,
        vertical_dimensions=vertical_dimensions,
    )


@st.composite
def grid3Dmetadata(draw) -> sgrid.Grid3DMetadata:
    N = 9
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

    node_coordinates_dim1 = names[6]
    node_coordinates_dim2 = names[7]
    node_coordinates_dim3 = names[8]
    has_node_coordinates = draw(st.booleans())

    if has_node_coordinates:
        node_coordinates = (node_coordinates_dim1, node_coordinates_dim2, node_coordinates_dim3)
    else:
        node_coordinates = None

    return sgrid.Grid3DMetadata(
        cf_role="grid_topology",
        topology_dimension=3,
        node_dimensions=(node_dimension1, node_dimension2, node_dimension3),
        volume_dimensions=(
            sgrid.DimDimPadding(face_dimension1, node_dimension1, padding_type1),
            sgrid.DimDimPadding(face_dimension2, node_dimension2, padding_type2),
            sgrid.DimDimPadding(face_dimension3, node_dimension3, padding_type3),
        ),
        node_coordinates=node_coordinates,
    )


grid_metadata = grid2Dmetadata() | grid3Dmetadata()
