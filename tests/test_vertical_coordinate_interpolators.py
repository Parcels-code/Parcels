import numpy as np
import pytest

from parcels import FieldSet
from parcels._datasets.structured.generated import simple_UV_sigma_dataset
from parcels.interpolators import (
    XLinear,
    XVerticalCoordinateConstant,
    XVerticalCoordinateLinear,
    XVerticalCoordinateNearest,
)
from parcels.interpolators._xinterpolators import _get_offsets_dictionary, _position_between_cell_centres

# Time-varying depth used to test the vertical coordinate interpolators. It is linear in level, lateral index and time,
# so the expected z column of each interpolator can be written down exactly, and strictly increasing along the levels.
SIGMA_SHAPE = {"n_times": 3, "nz": 5, "ny": 8, "nx": 8}
DEPTH_PER_LEVEL, DEPTH_PER_Y, DEPTH_PER_X, DEPTH_PER_TIME = 10.0, 0.3, 0.2, 0.1


def _linear_depth(level, y, x, t):
    return DEPTH_PER_LEVEL * level + DEPTH_PER_Y * y + DEPTH_PER_X * x + DEPTH_PER_TIME * t


def _sigma_grid(registration):
    """XGrid with a time-varying depth on the cell centres ("face") or on the cell corners ("node")."""
    ds = simple_UV_sigma_dataset(**SIGMA_SHAPE)
    lateral_dims = ("YC", "XC") if registration == "face" else ("YG", "XG")
    t, level, y, x = np.meshgrid(*(np.arange(n) for n in SIGMA_SHAPE.values()), indexing="ij")
    ds = ds.assign_coords(depth=(("time", "ZF", *lateral_dims), _linear_depth(level, y, x, t), {"axis": "Z"}))
    return FieldSet.from_sgrid_conventions(ds).U.grid


def _random_grid_positions(npart, seed=0):
    """Positions in the interior of the grid, away from the edges where the face registered columns are clamped."""
    rng = np.random.default_rng(seed)
    return {
        "T": {"index": rng.integers(0, SIGMA_SHAPE["n_times"] - 1, npart), "bcoord": rng.random(npart)},
        "Y": {"index": rng.integers(1, SIGMA_SHAPE["ny"] - 3, npart), "bcoord": rng.random(npart)},
        "X": {"index": rng.integers(1, SIGMA_SHAPE["nx"] - 3, npart), "bcoord": rng.random(npart)},
    }


# Expected lateral position of a particle's z column, from its search index (i), barycentric coordinate (b) and the
# padding offset between cell and face indices (o), measured in the indices of the registration's depth
def _cell_centre(i, b, o):
    return i + o


def _nearest_corner(i, b, o):
    return i + (b > 0.5)


def _position_in_face_indices(i, b, o):
    return i + o + b - 0.5


def _position_in_node_indices(i, b, o):
    return i + b


@pytest.mark.filterwarnings("ignore:Time-varying \\(4D\\) z coordinates are experimental")
@pytest.mark.parametrize(
    ("z_interp_method", "registration", "expected_y", "expected_x"),
    [
        # Columns at the particle's cell centre (face indices are the cell indices plus the padding offset)
        pytest.param(XVerticalCoordinateConstant(), "face", _cell_centre, _cell_centre, id="constant-face"),
        pytest.param(XVerticalCoordinateNearest(), "face", _cell_centre, _cell_centre, id="nearest-face"),
        # Column at the nearest corner of the particle's cell
        pytest.param(XVerticalCoordinateNearest(), "node", _nearest_corner, _nearest_corner, id="nearest-node"),
        # Bilinear interpolation reproduces the linear depth at the particle's position, measured in face or node indices
        pytest.param(
            XVerticalCoordinateLinear(), "face", _position_in_face_indices, _position_in_face_indices, id="linear-face"
        ),
        pytest.param(
            XVerticalCoordinateLinear(), "node", _position_in_node_indices, _position_in_node_indices, id="linear-node"
        ),
    ],
)
@pytest.mark.parametrize("time_varying", [True, False])
def test_vertical_coordinate_columns(z_interp_method, registration, expected_y, expected_x, time_varying):
    grid = _sigma_grid(registration)
    grid.z_interp_method = z_interp_method
    offsets = _get_offsets_dictionary(grid)
    gp = _random_grid_positions(npart=200)
    if not time_varying:
        gp["T"] = {"index": 1, "bcoord": 0.0}  # scalar time position, as on a time-constant grid

    columns = grid.z_interp_method.columns(gp, grid)

    expected = _linear_depth(
        np.arange(SIGMA_SHAPE["nz"])[np.newaxis, :],
        expected_y(gp["Y"]["index"], gp["Y"]["bcoord"], offsets["Y"])[:, np.newaxis],
        expected_x(gp["X"]["index"], gp["X"]["bcoord"], offsets["X"])[:, np.newaxis],
        np.add(gp["T"]["index"], gp["T"]["bcoord"])[..., np.newaxis] if time_varying else 1.0,
    )
    assert columns.shape == (200, SIGMA_SHAPE["nz"])
    np.testing.assert_allclose(columns, expected)


@pytest.mark.filterwarnings("ignore:Time-varying \\(4D\\) z coordinates are experimental")
@pytest.mark.parametrize("registration", ["face", "node"])
def test_vertical_coordinate_linear_is_continuous(registration):
    """z is continuous across cell edges, and across the cell centre (where face registered z changes the bracketing
    cell centres it interpolates between)."""
    ds = simple_UV_sigma_dataset(**SIGMA_SHAPE)
    lateral_dims = ("YC", "XC") if registration == "face" else ("YG", "XG")
    rng = np.random.default_rng(1)
    depth = ds["depth"].values + rng.random(ds["depth"].shape) * 0.01  # non-linear depth, unlike _sigma_grid
    ds = ds.assign_coords(depth=(("time", "ZF", *lateral_dims), depth, {"axis": "Z"}))
    grid = FieldSet.from_sgrid_conventions(ds).U.grid
    gp = _random_grid_positions(npart=100)
    yi, npart = gp["Y"]["index"], len(gp["Y"]["index"])

    def columns_at(y_index, y_bcoord):
        return grid.z_interp_method.columns({**gp, "Y": {"index": y_index, "bcoord": np.full(npart, y_bcoord)}}, grid)

    np.testing.assert_allclose(columns_at(yi, 1.0), columns_at(yi + 1, 0.0))
    np.testing.assert_allclose(columns_at(yi, 0.5 - 1e-12), columns_at(yi, 0.5))


@pytest.mark.parametrize(
    ("face_index", "bcoord", "expected_index", "expected_bcoord"),
    [
        pytest.param(3, 0.75, 3, 0.25, id="upper-half-of-cell"),
        pytest.param(3, 0.5, 3, 0.0, id="at-cell-centre"),
        pytest.param(3, 0.25, 2, 0.75, id="lower-half-of-cell"),
        pytest.param(0, 0.25, 0, 0.0, id="clamped-before-first-cell-centre"),
    ],
)
def test_position_between_cell_centres(face_index, bcoord, expected_index, expected_bcoord):
    index, centre_bcoord = _position_between_cell_centres(np.array([face_index]), np.array([bcoord]))
    assert index[0] == expected_index
    np.testing.assert_allclose(centre_bcoord[0], expected_bcoord)


@pytest.mark.filterwarnings("ignore:Time-varying \\(4D\\) z coordinates are experimental")
def test_z_interp_method_defaults_and_validation():
    face_grid, node_grid = _sigma_grid("face"), _sigma_grid("node")
    assert isinstance(face_grid.z_interp_method, XVerticalCoordinateLinear)
    assert isinstance(node_grid.z_interp_method, XVerticalCoordinateLinear)

    with pytest.raises(ValueError, match="needs depth on the cell centres"):
        node_grid.z_interp_method = XVerticalCoordinateConstant()
    with pytest.raises(ValueError, match="must be a `VerticalCoordinateInterpolator`"):
        face_grid.z_interp_method = XLinear()
