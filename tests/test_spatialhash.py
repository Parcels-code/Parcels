import numpy as np

from parcels._core.fieldset import FieldSet
from parcels._datasets.structured.generic import datasets


def test_spatialhash_init():
    ds = datasets["2d_left_rotated"]
    grid = FieldSet.from_sgrid_conventions(ds, mesh="flat").data_g.grid
    spatialhash = grid.get_spatial_hash()
    assert spatialhash is not None


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
    lon, lat = grid.lon, grid.lat
    clon = 0.25 * (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:])
    clat = 0.25 * (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:])
    jj, ii = np.meshgrid(np.arange(clat.shape[0]), np.arange(clat.shape[1]), indexing="ij")
    j, i, _ = spatialhash.query(clat.ravel(), clon.ravel())
    assert np.array_equal(j, jj.ravel())
    assert np.array_equal(i, ii.ravel())

    # Points far outside the regional domain must not match any cell
    j, i, _ = spatialhash.query([-60.0, 80.0], [120.0, -150.0])
    assert np.all(j == -3)
    assert np.all(i == -3)


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
