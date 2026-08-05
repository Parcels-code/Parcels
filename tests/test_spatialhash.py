from io import StringIO

import numpy as np

from parcels._core.fieldset import FieldSet
from parcels._core.spatialhash import _HASH_ENTRIES_PER_FACE, _HASH_ENTRY_BUDGET_MIN
from parcels._datasets.structured.generic import datasets


def _cell_centers(grid):
    lon, lat = grid.lon, grid.lat
    clon = 0.25 * (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:])
    clat = 0.25 * (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:])
    jj, ii = np.meshgrid(np.arange(clat.shape[0]), np.arange(clat.shape[1]), indexing="ij")
    return clat, clon, jj, ii


def test_spatialhash_init():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g.grid
    spatialhash = grid.get_spatial_hash()
    assert spatialhash is not None


def test_spatialhash_describe():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g.grid
    spatialhash = grid.get_spatial_hash()

    io = StringIO()
    expected = """\
Spatial Hash Grid Statistics
Grid type                                       : XGrid
Mesh                                            : FlatMesh()
Total mesh faces                                : 1,711
Bitwidth (current / max)                        : 1023 / 1023  (higher = finer resolution hash grid)
Total hash cells                                : 1,073,741,824
Occupied hash cells                             : 796,054, 0.0741%
Total (hash cell --> grid face) entries         : 1,080,194
Entries per occupied hash cell (avg)            : 1.36
Entries per face (avg)                          : 631.32
Faces per occupied hash cell (min / mean / max) : 1 / 1.36 / 4
"""
    spatialhash.describe(io)
    actual = io.getvalue()
    assert actual == expected


def test_invalid_positions():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g.grid

    j, i, _ = grid.get_spatial_hash().query([np.nan, np.inf], [np.nan, np.inf])
    assert np.all(j == -3)
    assert np.all(i == -3)


def test_spherical_regional_bounds():
    """Hash-grid bounds for spherical meshes are the Cartesian bounding box of the
    (regional) grid, not the whole unit cube, so quantization resolution is not
    wasted on parts of the sphere the grid does not cover.
    """
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="spherical").data_g.grid
    spatialhash = grid.get_spatial_hash()

    extents = np.array(
        [
            spatialhash._xmax - spatialhash._xmin,
            spatialhash._ymax - spatialhash._ymin,
            spatialhash._zmax - spatialhash._zmin,
        ]
    )
    assert np.all(extents > 0.0)
    assert np.all(extents < 2.0)  # strictly tighter than the unit cube

    # Queries at cell centers must still resolve to the correct cell
    clat, clon, jj, ii = _cell_centers(grid)
    j, i, _ = spatialhash.query(clat.ravel(), clon.ravel())
    assert np.array_equal(j, jj.ravel())
    assert np.array_equal(i, ii.ravel())

    # Points far outside the regional domain must not match any cell
    j, i, _ = spatialhash.query([-60.0, 80.0], [120.0, -150.0])
    assert np.all(j == -3)
    assert np.all(i == -3)


def test_hash_entry_budget():
    """When the requested bitwidth would blow the hash-entry budget (e.g. tilted
    regional spherical meshes, where face bounding boxes overlap 3-D blocks of
    hash cells), the resolution is reduced to fit; queries still resolve exactly.
    """
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="spherical").data_g.grid
    spatialhash = grid.get_spatial_hash()

    budget = max(_HASH_ENTRIES_PER_FACE * np.size(spatialhash._xlow), _HASH_ENTRY_BUDGET_MIN)
    assert spatialhash._total_hash_entries(1023) > budget  # this grid requires the cap
    assert spatialhash._bitwidth < 1023
    assert spatialhash._total_hash_entries(spatialhash._bitwidth) <= budget
    assert spatialhash._hash_table["faces"].size <= budget

    clat, clon, jj, ii = _cell_centers(grid)
    j, i, _ = spatialhash.query(clat.ravel(), clon.ravel())
    assert np.array_equal(j, jj.ravel())
    assert np.array_equal(i, ii.ravel())


def test_mixed_positions():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g.grid
    lat = grid.lat.mean()
    lon = grid.lon.mean()
    y = [lat, np.nan]
    x = [lon, np.nan]
    j, i, _ = grid.get_spatial_hash().query(y, x)
    assert j[0] == 29  # Actual value for 2d_left_rotated center
    assert i[0] == 14  # Actual value for 2d_left_rotated center
    assert j[1] == -3
    assert i[1] == -3


def test_nan_node_invalidates_touching_faces():
    """Any mesh face that touches a NaN node should not be added to the HashTable."""
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g.grid
    clat, clon, jj, ii = _cell_centers(grid)

    # `grid._ds` shares its lon/lat arrays with the module-level `datasets` fixture
    # (from_sgrid_conventions does not copy them), so deep-copy before mutating,
    # otherwise the NaN injected below leaks into every other test in the session.
    grid._ds = grid._ds.copy(deep=True)

    # Set one interior node to NaN, and calculate the indexes of faces that touch it.
    nj, ni = 10, 10
    touching = [
        (nj - 1, ni - 1),
        (nj - 1, ni),
        (nj, ni - 1),
        (nj, ni),
    ]
    grid._ds["lon"].values[nj, ni] = np.nan
    grid._ds["lat"].values[nj, ni] = np.nan
    spatialhash = grid.get_spatial_hash(reconstruct=True)

    # From the indexes of the faces that touch the NaN node, calculate their
    # face_ids.
    n_faces_x = clon.shape[1]
    invalid_ids = set()
    for j, i in touching:
        invalid_ids.add(j * n_faces_x + i)

    # Get a set of all of the face_ids that are in the table, and assert that the
    # ones touching the NaN node are not among them.
    faces_in_table = set(np.unique(spatialhash._hash_table["faces"]).tolist())
    assert invalid_ids.isdisjoint(faces_in_table)

    # The total number of mesh faces should be greater than the number in the table,
    # since the NaN faces are filtered from the table.
    n_total_faces = jj.size
    assert n_total_faces > len(faces_in_table)

    # Queries landing on those 4 faces should return GridSearchErrors (-3).
    touching_lat = np.array([clat[j, i] for j, i in touching])
    touching_lon = np.array([clon[j, i] for j, i in touching])
    j_touch, i_touch, _ = spatialhash.query(touching_lat, touching_lon)
    assert np.all(j_touch == -3)
    assert np.all(i_touch == -3)

    # All mesh cells not contacting the NaN node should resolve queries.
    mask = np.ones(clat.shape, dtype=bool)
    for j, i in touching:
        mask[j, i] = False
    j_rest, i_rest, _ = spatialhash.query(clat[mask], clon[mask])
    assert np.array_equal(j_rest, jj[mask])
    assert np.array_equal(i_rest, ii[mask])
