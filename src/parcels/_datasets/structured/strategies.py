import numpy as np
import xarray as xr
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as np_arrays

import parcels._strategies as pst
from parcels._core.utils import sgrid
from parcels._core.utils.sgrid import _attach_sgrid_metadata


def _face_size(node_size: int, padding: sgrid.Padding) -> int:
    if padding == sgrid.Padding.NONE:
        return node_size - 1
    elif padding in (sgrid.Padding.LOW, sgrid.Padding.HIGH):
        return node_size
    else:  # Padding.BOTH
        return node_size + 1


@st.composite
def sgrid_dataset(draw, grid: sgrid.SGrid2DMetadata | None = None) -> xr.Dataset:
    """Strategy to create Xarray Sgrid datasets for testing"""
    if grid is None:
        grid = draw(pst.sgrid.grid2Dmetadata(use_standard_names=True).filter(lambda g: g.node_coordinates is not None))
    elif grid.node_coordinates is None:
        raise ValueError("grid must have node_coordinates set")

    N = draw(st.integers(min_value=5, max_value=100))
    M = draw(st.integers(min_value=5, max_value=100))

    node_dim1, node_dim2 = grid.node_dimensions
    face_dim1 = grid.face_dimensions[0].face
    face_dim2 = grid.face_dimensions[1].face
    N_face = _face_size(N, grid.face_dimensions[0].padding)
    M_face = _face_size(M, grid.face_dimensions[1].padding)

    has_vertical = grid.vertical_dimensions is not None
    if has_vertical:
        P = draw(st.integers(min_value=5, max_value=20))
        vert_node_dim = grid.vertical_dimensions[0].node
        vert_face_dim = grid.vertical_dimensions[0].face
        P_face = _face_size(P, grid.vertical_dimensions[0].padding)

    has_curvilinear_grid = draw(st.booleans())
    coord_name1, coord_name2 = grid.node_coordinates

    if has_curvilinear_grid:
        c1, c2 = np.meshgrid(np.linspace(0, 100, N), np.linspace(0, 100, M), indexing="ij")
        coord1_dims = [node_dim1, node_dim2]
        coord2_dims = [node_dim1, node_dim2]
    else:
        c1 = np.linspace(0, 100, N)
        c2 = np.linspace(0, 100, M)
        coord1_dims = [node_dim1]
        coord2_dims = [node_dim2]

    num_fields = draw(st.integers(min_value=1, max_value=4))
    data_vars = {}

    for i in range(num_fields):
        dim1 = draw(st.sampled_from([node_dim1, face_dim1]))
        size1 = N if dim1 == node_dim1 else N_face

        dim2 = draw(st.sampled_from([node_dim2, face_dim2]))
        size2 = M if dim2 == node_dim2 else M_face

        if has_vertical and draw(st.booleans()):
            vert_dim = draw(st.sampled_from([vert_node_dim, vert_face_dim]))
            vert_size = P if vert_dim == vert_node_dim else P_face
            dims = [vert_dim, dim1, dim2]
            shape = (vert_size, size1, size2)
        else:
            dims = [dim1, dim2]
            shape = (size1, size2)

        data = draw(
            np_arrays(
                dtype=np.float64,
                shape=shape,
                elements=st.floats(min_value=1e-3, max_value=100.0, allow_nan=False, allow_infinity=False),
            )
        )
        data_vars[f"field_{i}"] = (dims, data)

    coords = {
        coord_name1: (coord1_dims, c1),
        coord_name2: (coord2_dims, c2),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return _attach_sgrid_metadata(ds, grid)


if __name__ == "__main__":
    ex = sgrid_dataset().example()
    breakpoint()
