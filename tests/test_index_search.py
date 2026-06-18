import numpy as np
import pytest
import xgcm

import parcels.tutorial
from parcels import XGrid
from parcels._core.fieldset import FieldSet
from parcels._core.index_search import _latlon_rad_to_xyz, _search_indices_curvilinear_2d
from parcels._datasets.structured.generic import datasets


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


# XGrid no longer accepts xgcm.Grid objects; now requires xr.Dataset with SGRID metadata; NEMO curvilinear workflow via xgcm is not the construction path
@pytest.mark.skip("Uses now removed API. TODO: What is the goal of this test?")
def test_indexing_nemo_curvilinear():
    ds = parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask")
    ds = ds.isel({"z_a": 0}, drop=True).rename({"glamf": "lon", "gphif": "lat", "z": "depth"})
    xgcm_grid = xgcm.Grid(ds, coords={"X": {"left": "x"}, "Y": {"left": "y"}}, periodic=False, autoparse_metadata=False)
    grid = XGrid(xgcm_grid, mesh="spherical")

    # Test points on the NEMO 1/4 degree curvilinear grid
    lats = np.array([-30, 0, 88])
    lons = np.array([30, 60, -150])

    yi, eta, xi, xsi = _search_indices_curvilinear_2d(grid, lats, lons)

    # Construct cornerpoints px
    px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])

    # Maximum 5 degree difference between px values
    for i in range(lons.shape[0]):
        np.testing.assert_allclose(px[1, i], px[:, i], atol=5)

    # Each query should have been located inside some cell: xsi, eta in [0, 1]
    assert np.all((xsi >= 0) & (xsi <= 1)), f"xsi out of [0,1]: {xsi}"
    assert np.all((eta >= 0) & (eta <= 1)), f"eta out of [0,1]: {eta}"

    # Reconstruct query lat/lon by bilinear-blending the 4 corner 3D unit-sphere
    # vectors with (xsi, eta) and renormalizing onto the unit sphere. Tolerance
    # reflects the residual spherical curvature for a 1/4° NEMO cell.
    clat = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])
    clon = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
    cX, cY, cZ = _latlon_rad_to_xyz(np.deg2rad(clat), np.deg2rad(clon))
    w = np.array([(1 - xsi) * (1 - eta), xsi * (1 - eta), xsi * eta, (1 - xsi) * eta])
    x = np.sum(w * cX, axis=0)
    y = np.sum(w * cY, axis=0)
    z = np.sum(w * cZ, axis=0)
    n = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    lat_recon = np.rad2deg(np.arcsin(z))
    lon_recon = np.rad2deg(np.arctan2(y, x))
    lons_wrapped = ((lons + 180) % 360) - 180
    dlon = ((lon_recon - lons_wrapped + 180) % 360) - 180
    np.testing.assert_allclose(lat_recon, lats, atol=1e-5)
    np.testing.assert_allclose(dlon, 0.0, atol=1e-5)
