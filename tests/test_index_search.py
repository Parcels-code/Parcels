import numpy as np
import pytest

from parcels._core.fieldset import FieldSet
from parcels._core.index_search import _search_indices_curvilinear_2d, uxgrid_point_in_cell
from parcels._datasets.structured.generic import datasets
from tests.utils import create_uxgrid_triangulated_patch, sample_points_inside_faces


@pytest.fixture
def field_cone():
    ds = datasets["2d_left_unrolled_cone"]
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    return fieldset.data_g


def test_grid_indexing_fpoints(field_cone):
    grid = field_cone.grid

    for yi_expected in range(grid.ydim - 1):
        for xi_expected in range(grid.xdim - 1):
            x = np.array([grid.lon[yi_expected, xi_expected] + 0.00001])
            y = np.array([grid.lat[yi_expected, xi_expected] + 0.00001])

            yi, eta, xi, xsi = _search_indices_curvilinear_2d(grid, y, x)
            if eta > 0.9:
                yi_expected -= 1
            if xsi > 0.9:
                xi_expected -= 1
            assert yi == yi_expected, f"Expected yi {yi_expected} but got {yi}"
            assert xi == xi_expected, f"Expected xi {xi_expected} but got {xi}"

            cell_lon = [
                grid.lon[yi, xi],
                grid.lon[yi, xi + 1],
                grid.lon[yi + 1, xi + 1],
                grid.lon[yi + 1, xi],
            ]
            cell_lat = [
                grid.lat[yi, xi],
                grid.lat[yi, xi + 1],
                grid.lat[yi + 1, xi + 1],
                grid.lat[yi + 1, xi],
            ]
            assert x > np.min(cell_lon) and x < np.max(cell_lon)
            assert y > np.min(cell_lat) and y < np.max(cell_lat)


def _near_edge_barycentric_weights(offset):
    """Weights for points just inside each edge; ``offset`` goes to the opposite vertex.

    The rest is split unevenly so samples avoid edge midpoints.
    """
    rest = 1.0 - offset
    return np.array(
        [
            [offset, rest * 0.61, rest * 0.39],
            [rest * 0.61, offset, rest * 0.39],
            [rest * 0.39, rest * 0.61, offset],
        ]
    )


# One interior point plus three a hair inside the edges, which is where the projection
# defect bites and where a particle crossing between faces sits.
_POINT_IN_CELL_WEIGHTS = np.vstack(
    [
        [[1 / 3, 1 / 3, 1 / 3]],
        _near_edge_barycentric_weights(1e-4),
    ]
)


@pytest.mark.parametrize(
    ("face_deg", "n"),
    [
        pytest.param(1.0, 8, id="1deg"),
        pytest.param(5.0, 8, id="5deg"),
        pytest.param(15.0, 4, id="15deg"),
        pytest.param(25.0, 4, id="25deg-notebook-scale"),
    ],
)
@pytest.mark.xfail(reason="#2878 - orthogonal, not radial, projection onto the face plane")
def test_uxgrid_point_in_cell_locates_interior_points_of_large_faces(face_deg, n):
    """``uxgrid_point_in_cell`` must accept a point inside the face it is given.

    The spherical branch projects onto the face plane along the normal; membership is
    defined by the ray from the origin, so the projection must be radial (gnomonic).
    The error scales with face size times proximity to an edge, not face size alone.

    The face index is passed in directly, bypassing the hash, so the bounding-box
    defect cannot influence the result. The tolerance on the coordinate sum is far
    tighter than the ``rtol=1e-3`` gate in the source: radial projection makes the sum
    exactly 1, so this pins the fix to the projection rather than to a looser gate.
    """
    grid, nodes, faces = create_uxgrid_triangulated_patch(face_deg, centre=(25.0, 10.0), n=n, mesh="spherical")
    lon, lat, expected_face = sample_points_inside_faces(nodes, faces, weights=_POINT_IN_CELL_WEIGHTS)

    is_in_cell, coords = uxgrid_point_in_cell(grid, lat, lon, expected_face, expected_face)

    coord_sum = coords.sum(axis=1)
    worst = int(np.argmax(np.abs(coord_sum - 1.0)))
    assert np.allclose(coord_sum, 1.0, rtol=1e-6, atol=1e-6), (
        f"barycentric coordinates of interior points do not sum to 1; worst is "
        f"{coord_sum[worst]:.6f} at (lon, lat)=({lon[worst]:.3f}, {lat[worst]:.3f}) "
        f"in face {expected_face[worst]}"
    )

    n_rejected = int(np.count_nonzero(is_in_cell == 0))
    assert n_rejected == 0, f"{n_rejected} of {len(lon)} points were rejected from the face containing them"
