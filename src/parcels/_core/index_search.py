from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from parcels._core.statuscodes import _raise_outside_time_interval_error
from parcels._core.utils.time import timedelta_to_float

if TYPE_CHECKING:
    from parcels._core.field import Field
    from parcels._core.xgrid import XGrid


GRID_SEARCH_ERROR = -3
LEFT_OUT_OF_BOUNDS = -2
RIGHT_OUT_OF_BOUNDS = -1


def _search_1d_array(
    arr: np.ndarray,
    x: float,
) -> tuple[int, int]:
    """
    Searches for particle locations in a 1D array and returns barycentric coordinate along dimension.

    Assumptions:
    - array is strictly monotonically increasing.

    Parameters
    ----------
    arr : np.array
        1D array to search in.
    x : float
        Position in the 1D array to search for.

    Returns
    -------
    array of int
        Index of the element just before the position x in the array. Note that this index is -2 if the index is left out of bounds and -1 if the index is right out of bounds.
    array of float
        Barycentric coordinate.
    """
    # TODO v4: We probably rework this to deal with 0D arrays before this point (as we already know field dimensionality)
    if len(arr) < 2:
        return np.zeros(shape=x.shape, dtype=np.int32), np.zeros_like(x)
    index = np.clip(np.searchsorted(arr, x, side="right") - 1, 0, len(arr) - 2)
    # Use broadcasting to avoid repeated array access
    arr_index = arr[index]
    arr_next = arr[np.clip(index + 1, 1, len(arr) - 1)]  # Ensure we don't go out of bounds
    bcoord = (x - arr_index) / (arr_next - arr_index)

    # TODO check how we can avoid searchsorted when grid spacing is uniform
    # dx = arr[1] - arr[0]
    # index = ((x - arr[0]) / dx).astype(int)
    # index = np.clip(index, 0, len(arr) - 2)
    # bcoord = (x - arr[index]) / dx

    index = np.where(x < arr[0], LEFT_OUT_OF_BOUNDS, index)
    index = np.where(x > arr[-1], RIGHT_OUT_OF_BOUNDS, index)

    return np.atleast_1d(index), np.atleast_1d(bcoord)


def _search_time_index(field: Field, time: np.ndarray):
    """Find and return the index and relative coordinate in the time array associated with a given time.

    Parameters
    ----------
    field: Field

    time: datetime
        This is the amount of time, in seconds (time_delta), in unix epoch
    Note that we normalize to either the first or the last index
    if the sampled value is outside the time value range.
    """
    if field.time_interval is None:
        return {
            "T": {
                "index": np.zeros(shape=time.shape, dtype=np.int32),
                "bcoord": np.zeros(shape=time.shape, dtype=np.float32),
            }
        }

    if not field.time_interval.is_all_time_in_interval(time):
        _raise_outside_time_interval_error(time, field=None)

    time_flt = timedelta_to_float(field.data.time.data - field.time_interval.left)
    ti, tau = _search_1d_array(time_flt, time)

    return {"T": {"index": np.atleast_1d(ti), "bcoord": np.atleast_1d(tau)}}


def curvilinear_point_in_cell(grid, y: np.ndarray, x: np.ndarray, yi: np.ndarray, xi: np.ndarray):
    """Locate points within 2D curvilinear grid cells.

    For spherical grids, the bilinear inverse runs in a tangent plane on a sphere
    through each cell's centroid — robust across the antimeridian and the
    polar cap. For flat grids it runs directly in (lon, lat). In both
    cases the returned (xsi, eta) are bilinear weights in the space the inverse
    was solved in; downstream interpolators must agree on that same space.
    """
    clon = np.asarray(
        [grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]],
        dtype=float,
    )
    clat = np.asarray(
        [grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]],
        dtype=float,
    )

    if grid._mesh == "spherical":
        xsi, eta = _bilinear_inverse_tangent_plane(clon, clat, x, y)
        is_in_cell = np.where((xsi >= 0) & (xsi <= 1) & (eta >= 0) & (eta <= 1), 1, 0)
    else:
        xsi, eta = _bilinear_inverse_latlon(clon, clat, x, y, apply_antimeridian_shift=False)
        is_in_cell = np.where((xsi >= 0) & (xsi <= 1) & (eta >= 0) & (eta <= 1), 1, 0)

    return is_in_cell, np.column_stack((xsi, eta))


_invA = np.array(
    [
        [1, 0, 0, 0],
        [-1, 1, 0, 0],
        [-1, 0, 0, 1],
        [1, -1, 1, -1],
    ]
)


def _bilinear_inverse(px, py, xq, yq):
    """Solve for (xsi, eta) s.t. bilinear blend of (px, py) corners equals (xq, yq)."""
    eta_init = -1.0 * np.ones(len(xq), dtype=float)
    a, b = np.dot(_invA, px), np.dot(_invA, py)
    aa = a[3] * b[2] - a[2] * b[3]
    bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + xq * b[3] - yq * a[3]
    cc = a[1] * b[0] - a[0] * b[1] + xq * b[1] - yq * a[1]
    det2 = bb * bb - 4 * aa * cc

    with np.errstate(divide="ignore", invalid="ignore"):
        det = np.where(det2 > 0, np.sqrt(det2), eta_init)
        eta = np.where(abs(aa) < 1e-12, -cc / bb, np.where(det2 > 0, (-bb + det) / (2 * aa), eta_init))
        xsi = np.where(
            abs(a[1] + a[3] * eta) < 1e-12,
            ((yq - py[0]) / (py[1] - py[0]) + (yq - py[3]) / (py[2] - py[3])) * 0.5,
            (xq - a[0] - a[2] * eta) / (a[1] + a[3] * eta),
        )
    return xsi, eta


def _bilinear_inverse_latlon(clon, clat, x, y, apply_antimeridian_shift=True):
    """Bilinear inverse in raw lat/lon, with antimeridian corner shifting."""
    px = clon.copy()
    xq = np.asarray(x, dtype=float).copy()
    if apply_antimeridian_shift:
        px = ((px + 180.0) % 360.0) - 180.0
        xq = ((xq + 180.0) % 360.0) - 180.0
        lon_span = px.max(axis=0) - px.min(axis=0)
        antimeridian_cell = lon_span > 180.0
        if np.any(antimeridian_cell):
            mask = (px < 0.0) & antimeridian_cell[np.newaxis, :] & (xq[np.newaxis, :] > 0.0)
            px[mask] += 360.0
            mask = (px > 0.0) & antimeridian_cell[np.newaxis, :] & (xq[np.newaxis, :] < 0.0)
            px[mask] -= 360.0
    py = clat
    yq = np.asarray(y, dtype=float)
    return _bilinear_inverse(px, py, xq, yq)


def _bilinear_inverse_tangent_plane(clon, clat, x, y):
    """Bilinear inverse after tangent-plane projection through the cell centroid.

    Avoids lat/lon singularities at poles and on the antimeridian.
    """
    px, py, xq, yq = _spherical_project_cell_and_query(clon, clat, x, y)
    return _bilinear_inverse(px, py, xq, yq)


def _spherical_project_cell_and_query(clon, clat, x, y):
    """Project 4 cell corners and the query onto a plane defined by the cell.

    The plane's basis (e_u, e_v) is built from the cell's edge-midpoint
    difference vectors, then orthonormalized using Gramm-Schmidt

    Inputs
    ------
    clon, clat : (4, N) arrays of corner lon/lat in degrees
    x, y       : (N,) query lon/lat in degrees

    Returns
    -------
    px, py : (4, N) projected corner coordinates (u, v)
    xq, yq : (N,) projected query coordinates in the same (u, v)

    Notes
    -----
    Orthogonal projection is 2-to-1 on the full sphere: a query and its
    antipode map to the same (u, v). In practice this is harmless because the
    downstream bilinear gate ``(xsi, eta) in [0, 1]`` only admits queries whose
    projected magnitude is ≲ the cell's angular size.
    """
    cX, cY, cZ = _latlon_rad_to_xyz(np.deg2rad(clat), np.deg2rad(clon))  # (4, N) each
    qX, qY, qZ = _latlon_rad_to_xyz(np.deg2rad(np.asarray(y, dtype=float)), np.deg2rad(np.asarray(x, dtype=float)))

    # Cell-intrinsic tangent basis from edge-midpoint difference vectors.
    # Corner order is CCW: c0=(yi,xi), c1=(yi,xi+1), c2=(yi+1,xi+1), c3=(yi+1,xi).
    # u-axis (along xsi): right-edge midpoint - left-edge midpoint = (c1+c2) - (c0+c3)
    # v-axis (along eta): top-edge   midpoint - bottom-edge midpoint = (c2+c3) - (c0+c1)
    ux = (cX[1] + cX[2]) - (cX[0] + cX[3])
    uy = (cY[1] + cY[2]) - (cY[0] + cY[3])
    uz = (cZ[1] + cZ[2]) - (cZ[0] + cZ[3])
    u_norm = np.sqrt(ux * ux + uy * uy + uz * uz)
    u_norm = np.where(u_norm == 0.0, 1.0, u_norm)
    e_ux, e_uy, e_uz = ux / u_norm, uy / u_norm, uz / u_norm

    vx = (cX[2] + cX[3]) - (cX[0] + cX[1])
    vy = (cY[2] + cY[3]) - (cY[0] + cY[1])
    vz = (cZ[2] + cZ[3]) - (cZ[0] + cZ[1])
    # Gram-Schmidt: make e_v perpendicular to e_u so the bilinear inverse
    # in (u, v) is well-conditioned even on skewed cells.
    v_dot_eu = vx * e_ux + vy * e_uy + vz * e_uz
    vx = vx - v_dot_eu * e_ux
    vy = vy - v_dot_eu * e_uy
    vz = vz - v_dot_eu * e_uz
    v_norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    v_norm = np.where(v_norm == 0.0, 1.0, v_norm)
    e_vx, e_vy, e_vz = vx / v_norm, vy / v_norm, vz / v_norm

    # Orthogonal projection onto span(e_u, e_v): two dot products per point.
    # Same pattern as uxgrid_point_in_cell (drop the component normal to the plane).
    def _project(vx, vy, vz):
        u = vx * e_ux + vy * e_uy + vz * e_uz
        v = vx * e_vx + vy * e_vy + vz * e_vz
        return u, v

    px_u, px_v = _project(cX, cY, cZ)  # (4, N) each
    xq_u, xq_v = _project(qX, qY, qZ)  # (N,) each
    return px_u, px_v, xq_u, xq_v


def _search_indices_curvilinear_2d(
    grid: XGrid, y: np.ndarray, x: np.ndarray, yi: np.ndarray | None = None, xi: np.ndarray | None = None
):
    """Searches a grid for particle locations in 2D curvilinear coordinates.

    Parameters
    ----------
    grid : XGrid
        The curvilinear grid to search within.
    y : np.ndarray
        Array of latitude-coordinates of the points to locate.
    x : np.ndarray
        Array of longitude-coordinates of the points to locate.
    yi : np.ndarray | None, optional
        Array of initial guesses for the j indices of the points to locate.
    xi : np.ndarray | None, optional
        Array of initial guesses for the i indices of the points to locate.

    Returns
    -------
    tuple
        A tuple containing four elements:
        - yi (np.ndarray): Array of found j-indices corresponding to the input coordinates.
        - eta (np.ndarray): Array of barycentric coordinates in the j-direction within the found grid cells.
        - xi (np.ndarray): Array of found i-indices corresponding to the input cooordinates.
        - xsi (np.ndarray): Array of barycentric coordinates in the i-direction within the found grid cells.
    """
    if np.any(xi):
        # If an initial guess is provided, we first perform a point in cell check for all guessed indices
        is_in_cell, coords = curvilinear_point_in_cell(grid, y, x, yi, xi)
        y_check = y[is_in_cell == 0]
        x_check = x[is_in_cell == 0]
        zero_indices = np.where(is_in_cell == 0)[0]
    else:
        # Otherwise, we need to check all points
        yi = np.full(len(y), GRID_SEARCH_ERROR, dtype=np.int32)
        xi = np.full(len(x), GRID_SEARCH_ERROR, dtype=np.int32)
        y_check = y
        x_check = x
        coords = -1.0 * np.ones((len(y), 2), dtype=np.float32)
        zero_indices = np.arange(len(y))

    # If there are any points that were not found in the first step, we query the spatial hash for those points
    if len(zero_indices) > 0:
        yi_q, xi_q, coords_q = grid.get_spatial_hash().query(y_check, x_check)
        # Only those points that were not found in the first step are updated
        coords[zero_indices, :] = coords_q
        yi[zero_indices] = yi_q
        xi[zero_indices] = xi_q

    xsi = coords[:, 0]
    eta = coords[:, 1]

    return (yi, eta, xi, xsi)


def uxgrid_point_in_cell(grid, y: np.ndarray, x: np.ndarray, yi: np.ndarray, xi: np.ndarray):
    """Check if points are inside the grid cells defined by the given face indices.

    Parameters
    ----------
    grid : ux.grid.Grid
        The uxarray grid object containing the unstructured grid data.
    y : np.ndarray
        Array of latitudes of the points to check.
    x : np.ndarray
        Array of longitudes of the points to check.
    yi : np.ndarray
        Not used, but included for compatibility with other search functions.
    xi : np.ndarray
        Array of face indices corresponding to the points.

    Returns
    -------
    is_in_cell : np.ndarray
        An array indicating whether each point is inside (1) or outside (0) the corresponding cell.
    coords : np.ndarray
        Barycentric coordinates of the points within their respective cells.
    """
    if grid._mesh == "spherical":
        lon_rad = np.deg2rad(x)
        lat_rad = np.deg2rad(y)
        x_cart, y_cart, z_cart = _latlon_rad_to_xyz(lat_rad, lon_rad)
        points = np.column_stack((x_cart.flatten(), y_cart.flatten(), z_cart.flatten()))  # (M,3)

        # Get the vertex indices for each face
        nids = grid.uxgrid.face_node_connectivity[xi].values
        face_vertices = np.stack(
            (
                grid.uxgrid.node_x[nids.ravel()].values.reshape(nids.shape),
                grid.uxgrid.node_y[nids.ravel()].values.reshape(nids.shape),
                grid.uxgrid.node_z[nids.ravel()].values.reshape(nids.shape),
            ),
            axis=-1,
        )

        # Get projection points onto element plane. Keep the leading
        # dimension even for single-face queries so shapes remain (M,3).
        r1 = face_vertices[:, 1, :] - face_vertices[:, 0, :]
        r2 = face_vertices[:, 2, :] - face_vertices[:, 0, :]
        nhat = np.cross(r1, r2)
        norm = np.linalg.norm(nhat, axis=-1)
        # Avoid division by zero for degenerate faces
        norm = np.where(norm == 0.0, 1.0, norm)
        nhat = nhat / norm[:, None]
        # Calculate the component of the points in the direction of nhat
        ptilde = points - face_vertices[:, 0, :]
        pdotnhat = np.sum(ptilde * nhat, axis=-1)
        # Reconstruct points with normal component removed.
        points = ptilde - pdotnhat[:, None] * nhat + face_vertices[:, 0, :]

    else:
        nids = grid.uxgrid.face_node_connectivity[xi, :].values
        face_vertices = np.stack(
            (
                grid.uxgrid.node_lon[nids.ravel()].values.reshape(nids.shape),
                grid.uxgrid.node_lat[nids.ravel()].values.reshape(nids.shape),
            ),
            axis=-1,
        )
        points = np.stack((x, y), axis=-1)

    M = len(points)

    coords = _barycentric_coordinates(face_vertices, points)

    is_in_cell = np.zeros(M, dtype=np.int32)
    is_in_cell = np.where(np.all((coords >= -1e-6), axis=1), 1, 0)
    is_in_cell &= np.isclose(np.sum(coords, axis=1), 1.0, rtol=1e-3, atol=1e-6)

    return is_in_cell, coords


def _triangle_area(A, B, C):  # noqa: N803
    """Compute the area of a triangle given by three points."""
    d1 = B - A
    d2 = C - A
    if A.shape[-1] == 2:
        # 2D case: cross product reduces to scalar z-component
        cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        area = 0.5 * cross
    elif A.shape[-1] == 3:
        # 3D case: full vector cross product
        cross = np.cross(d1, d2)
        area = 0.5 * np.linalg.norm(cross, axis=-1)
    else:
        raise ValueError(f"Expected last dim=2 or 3, got {A.shape[-1]}")

    return area


def _barycentric_coordinates(nodes, points, min_area=1e-8):
    """
    Compute the barycentric coordinates of a point P inside a convex polygon using area-based weights.
    This method currently only works for triangular cells (K=3)

    TODO: So that this method generalizes to n-sided polygons, we can use the Waschpress points as the generalized
    barycentric coordinates, which is valid for convex polygons.

    Parameters
    ----------
        nodes : numpy.ndarray
            Polygon verties per query of shape (M, 3, 2/3) where M is the number of query points. The second dimension corresponds to the number
            of vertices
            The last dimension can be either 2 or 3, where 3 corresponds to the (x, y, z) coordinates of each vertex and 2 corresponds to the
            (lon, lat) coordinates of each vertex.

        points : numpy.ndarray
            Spherical coordinates of the point (M,2/3) where M is the number of query points.

    Returns
    -------
    numpy.ndarray
        Barycentric coordinates corresponding to each vertex.

    """
    M, K = nodes.shape[:2]

    vi0 = nodes[:, 0, :]  # (M,K,2)
    vi1 = nodes[:, 1, :]  # (M,K,2)
    vi2 = nodes[:, 2, :]  # (M,K,2)

    a = _triangle_area(vi0, vi1, vi2)  # (M,K)

    P = points
    a0 = _triangle_area(P, vi1, vi2)
    a1 = _triangle_area(P, vi2, vi0)
    a2 = _triangle_area(P, vi0, vi1)

    bcoords = np.zeros((M, K))
    bcoords[:, 0] = a0 / a
    bcoords[:, 1] = a1 / a
    bcoords[:, 2] = a2 / a

    return bcoords


def _latlon_rad_to_xyz(
    lat,
    lon,
):
    """Converts Spherical latitude and longitude coordinates into Cartesian x,
    y, z coordinates.
    """
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z
