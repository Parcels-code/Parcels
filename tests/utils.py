"""General helper functions and utilies for test suite."""

from __future__ import annotations

import struct
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
import xarray as xr

import parcels
from parcels import FieldSet, Particle, Variable
from parcels._core.xgrid import _FIELD_DATA_ORDERING, XGrid, get_axis_from_dim_name
from parcels._datasets.structured.generated import simple_UV_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "tests"
TEST_DATA = TEST_ROOT / "test_data"

# Define default particle classes for different built-in kernels
DEFAULT_PARTICLES = defaultdict(lambda: Particle)
DEFAULT_PARTICLES[parcels.kernels.AdvectionRK45] = Particle.add_variable(Variable("next_dt"))


def create_fieldset_unit_mesh(xdim=20, ydim=20, mesh="flat") -> FieldSet:
    """Standard unit mesh fieldset with U and V equivalent to longitude and latitude."""
    lon = np.linspace(0.0, 1.0, xdim, dtype=np.float32)
    lat = np.linspace(0.0, 1.0, ydim, dtype=np.float32)
    V, U = np.meshgrid(lon, lat)
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
    dimensions = {"lat": lat, "lon": lon}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


def create_fieldset_zeros_3d(zdim=5, ydim=10, xdim=10):
    """3d fieldset with U, V, and W equivalent to longitude, latitude, and depth."""
    tdim = 20
    ds = xr.Dataset(
        {
            "U": (("time", "depth", "lat", "lon"), np.zeros((tdim, zdim, ydim, xdim))),
            "V": (("time", "depth", "lat", "lon"), np.zeros((tdim, zdim, ydim, xdim))),
            "W": (("time", "depth", "lat", "lon"), np.zeros((tdim, zdim, ydim, xdim))),
        },
        coords={
            "time": np.linspace(0, tdim - 1, tdim),
            "depth": np.linspace(0, 1, zdim),
            "lat": np.linspace(0, 1, ydim),
            "lon": np.linspace(0, 1, xdim),
        },
    )
    variables = {"U": "U", "V": "V", "W": "W"}
    dimensions = {"time": "time", "lon": "lon", "lat": "lat", "depth": "depth"}
    return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh="flat")


def create_fieldset_zeros_unit_mesh(xdim=100, ydim=100):
    """Standard unit mesh fieldset with flat mesh, and zero velocity."""
    data = {"U": np.zeros((ydim, xdim), dtype=np.float32), "V": np.zeros((ydim, xdim), dtype=np.float32)}
    dimensions = {"lon": np.linspace(0, 1, xdim, dtype=np.float32), "lat": np.linspace(0, 1, ydim, dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh="flat")


def create_fieldset_global(xdim=200, ydim=100):
    """Standard fieldset spanning the earth's coordinates with U and V equivalent to longitude and latitude in deg."""
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    V, U = np.meshgrid(lon, lat)
    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat}
    return FieldSet.from_data(data, dimensions, mesh="flat")


def create_fieldset_zeros_conversion(mesh="spherical", xdim=200, ydim=100) -> FieldSet:
    """Zero velocity field with lat and lon determined by a conversion factor."""
    mesh_conversion = 1 / 1852.0 / 60 if mesh == "spherical" else 1
    ds = simple_UV_dataset(dims=(2, 1, ydim, xdim), mesh=mesh)
    ds["lon"].data = np.linspace(-1e6 * mesh_conversion, 1e6 * mesh_conversion, xdim)
    ds["lat"].data = np.linspace(-1e6 * mesh_conversion, 1e6 * mesh_conversion, ydim)
    return FieldSet.from_sgrid_conventions(ds, mesh=mesh)


def create_simple_pset(n=1):
    zeros = np.zeros(n)
    return parcels.ParticleSet(
        fieldset=create_fieldset_unit_mesh(),
        pclass=parcels.Particle,
        x=zeros,
        y=zeros,
        depth=zeros,
        time=zeros,
    )


def create_spherical_positions(n_particles, max_depth=100000):
    yrange = 2 * np.random.rand(n_particles)
    lat = 180 * (np.arccos(1 - yrange) - 0.5 * np.pi) / np.pi
    lon = 360 * np.random.rand(n_particles)
    depth = max_depth * np.random.rand(n_particles)
    return np.array((depth, lat, lon))


def create_flat_positions(n_particle):
    return np.random.rand(n_particle * 3).reshape(3, n_particle)


def create_fieldset_zeros_simple(xdim=40, ydim=100, withtime=False):
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(-60, 60, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    dimensions = {"lat": lat, "lon": lon, "depth": depth}
    if withtime:
        tdim = 10
        time = np.linspace(0, 86400, tdim, dtype=np.float64)
        dimensions["time"] = time
        datadims = (tdim, ydim, xdim)
        allow_time_extrapolation = False
    else:
        datadims = (ydim, xdim)
        allow_time_extrapolation = True
    U = np.zeros(datadims, dtype=np.float32)
    V = np.zeros(datadims, dtype=np.float32)
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, allow_time_extrapolation=allow_time_extrapolation)


def assert_empty_folder(path: Path):
    assert [p.name for p in path.iterdir()] == []


def assert_valid_field_data(data: xr.DataArray, grid: XGrid):
    assert len(data.shape) == 4, f"Field data should have 4 dimensions (time, depth, lat, lon), got dims {data.dims}"

    for ax_expected, dim in zip(_FIELD_DATA_ORDERING, data.dims, strict=True):
        ax_actual = get_axis_from_dim_name(grid.sgrid_metadata, dim)
        if ax_actual is None:
            continue  # None is ok
        assert ax_actual == ax_expected, f"Expected axis {ax_expected} for dimension '{dim}', got {ax_actual}"


def round_and_hash_float_array(arr, decimals=6):
    arr = np.round(arr, decimals=decimals)

    # Adapted from https://cs.stackexchange.com/a/37965
    h = 1
    for f in arr.flat:
        # Mimic Float.floatToIntBits: converts float to 4-byte binary, then interprets as int
        float_as_int = struct.unpack("!i", struct.pack("!f", f))[0]
        h = 31 * h + float_as_int

    # Mimic Java's HashMap hash transformation
    h ^= (h >> 20) ^ (h >> 12)
    return h ^ (h >> 7) ^ (h >> 4)


def assert_cftime_like_particlefile(parquet_path: Path) -> None:
    assert parquet_path.suffix == ".parquet", "Path must be a parquet file"

    df = parcels.read_particlefile(parquet_path, decode_times=True)

    # check first value (and hence rest of array) is what we expect
    # TODO explore use of cftime in polars
    assert isinstance(df["t"][0], (cftime.datetime, datetime)), (
        "CF-time values in Parquet did not get properly decoded. Are the attributes correct?"
    )
    return


def create_uxgrid_from_triangulation(node_lon, node_lat, faces, mesh="spherical", z=(0.0, 1.0)):
    """Wrap a bare triangulation (node lon/lat in degrees, triangles) in a parcels UxGrid."""
    import uxarray as ux

    from parcels._core.uxgrid import UxGrid

    uxgrid = ux.Grid.from_topology(
        node_lon=np.asarray(node_lon, dtype=np.float64),
        node_lat=np.asarray(node_lat, dtype=np.float64),
        face_node_connectivity=np.asarray(faces, dtype=np.int64),
    )
    zc = ux.UxDataArray(np.asarray(z, dtype=np.float64), dims="zf", uxgrid=uxgrid)
    return UxGrid(uxgrid, zc, mesh=mesh)


def create_uxgrid_triangulated_patch(face_deg, centre=(0.0, 0.0), n=8, mesh="spherical"):
    """Regular ``n`` x ``n`` lon/lat patch of ``face_deg`` quads, split into triangles.

    ``centre`` matters on spherical meshes: the Cartesian coordinate functions have
    stationary points at lon in {0, +-90, 180} on the equator and at the poles, and a
    face straddling one varies quadratically rather than linearly there.

    Returns ``(grid, nodes, faces)`` with nodes as (lon, lat) degrees.
    """
    half = 0.5 * face_deg * n
    if abs(centre[1]) + half > 90.0:
        raise ValueError(
            f"patch of {n} x {face_deg} deg faces centred at lat {centre[1]} runs past the pole; reduce n or face_deg"
        )
    lons = np.linspace(centre[0] - half, centre[0] + half, n + 1)
    lats = np.linspace(centre[1] - half, centre[1] + half, n + 1)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    nodes = np.column_stack((grid_lon.ravel(), grid_lat.ravel()))

    faces = []
    for j in range(n):
        for i in range(n):
            sw = j * (n + 1) + i
            se = sw + 1
            nw = sw + n + 1
            ne = nw + 1
            faces.append([sw, se, ne])
            faces.append([sw, ne, nw])
    faces = np.asarray(faces, dtype=np.int64)

    grid = create_uxgrid_from_triangulation(nodes[:, 0], nodes[:, 1], faces, mesh=mesh)
    return grid, nodes, faces


# Asymmetric so samples avoid the centroid, edge midpoints, and the shared quad diagonal.
_INTERIOR_BARYCENTRIC_WEIGHTS = np.array(
    [
        [1 / 3, 1 / 3, 1 / 3],
        [0.70, 0.19, 0.11],
        [0.11, 0.70, 0.19],
        [0.19, 0.11, 0.70],
        [0.46, 0.31, 0.23],
        [0.23, 0.46, 0.31],
    ]
)


def sample_points_inside_faces(nodes, faces, weights=None):
    """Sample points strictly inside each triangle, with the containing face known exactly.

    Triangles are straight-sided in lon/lat, so strictly positive barycentric weights
    put a point inside its face by construction - no point-in-polygon library needed.

    Returns ``(lon, lat, expected_face)``.
    """
    if weights is None:
        weights = _INTERIOR_BARYCENTRIC_WEIGHTS
    weights = np.asarray(weights, dtype=np.float64)
    assert np.all(weights > 0.0), "weights must be strictly positive to lie inside the face"
    assert np.allclose(weights.sum(axis=1), 1.0), "weights must sum to 1"

    verts = np.asarray(nodes, dtype=np.float64)[np.asarray(faces)]  # (n_face, 3, 2)
    # (n_face, n_weights, 2)
    pts = np.einsum("wk,fkc->fwc", weights, verts)
    expected_face = np.repeat(np.arange(len(faces)), len(weights))
    return pts[..., 0].ravel(), pts[..., 1].ravel(), expected_face


def cartesian_face_bounds_from_vertices(grid):
    """Per-face Cartesian bounding box from vertices only, as ``SpatialHash`` builds it.

    Lets tests assert on box geometry directly instead of inferring it from a failed query.
    Returns ``(low, high)``, each shape (n_face, 3).
    """
    from parcels._core.index_search import _latlon_rad_to_xyz

    nids = grid.uxgrid.face_node_connectivity.values
    lon = np.deg2rad(grid.uxgrid.node_lon.values[nids])
    lat = np.deg2rad(grid.uxgrid.node_lat.values[nids])
    verts = np.stack(_latlon_rad_to_xyz(lat, lon), axis=-1)  # (n_face, 3, 3)
    return verts.min(axis=1), verts.max(axis=1)
