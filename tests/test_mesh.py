import numpy as np
import pytest

from parcels import FieldSet, ParticleSet, SphericalMesh
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.kernels import AdvectionRK4

EARTH_DEG2M = 1852 * 60.0


def test_spherical_mesh_deg2m():
    assert SphericalMesh().radius is None
    assert SphericalMesh().deg2m == EARTH_DEG2M
    r = 3389500.0  # Mars radius
    assert SphericalMesh(radius=r).deg2m == pytest.approx(r * np.pi / 180)


@pytest.mark.parametrize(
    "mesh, exp_radius, exp_deg2m",
    [
        ("spherical", None, EARTH_DEG2M),
        (SphericalMesh(), None, EARTH_DEG2M),
        (SphericalMesh(radius=3389500.0), 3389500.0, 3389500.0 * np.pi / 180),
    ],
)
def test_xgrid_radius_and_deg2m(mesh, exp_radius, exp_deg2m):
    grid = FieldSet.from_sgrid_conventions(simple_UV_dataset(), mesh=mesh).U.grid
    assert grid._mesh == "spherical"
    assert grid._radius == exp_radius
    assert grid.deg2m == pytest.approx(exp_deg2m)


@pytest.mark.parametrize("radius", [None, 3389500.0, 6051800.0, 6371000.0])  # Mars, Venus, Earth
def test_advection_uses_custom_radius(radius, npart=10):
    ds = simple_UV_dataset()
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=SphericalMesh(radius=radius))

    runtime = 7200
    startlat = np.linspace(0, 80, npart)
    startlon = 20.0 + np.zeros(npart)
    pset = ParticleSet(fieldset, x=startlon, y=startlat)
    pset.execute(AdvectionRK4, runtime=runtime, dt=np.timedelta64(15, "m"))

    deg2m = EARTH_DEG2M if radius is None else radius * np.pi / 180
    expected_dlon = runtime / (deg2m * np.cos(np.deg2rad(pset.y)))
    np.testing.assert_allclose(pset.x - startlon, expected_dlon, atol=1e-5)
    np.testing.assert_allclose(pset.y, startlat, atol=1e-5)


@pytest.mark.parametrize("bad_radius", ["6371000", [6371000], (1, 2), {}])
def test_spherical_mesh_rejects_non_numeric_radius(bad_radius):
    with pytest.raises(TypeError):
        SphericalMesh(radius=bad_radius)


@pytest.mark.parametrize("bad_radius", [0, -1.0, -6371000])
def test_spherical_mesh_rejects_nonpos_radius(bad_radius):
    with pytest.raises(ValueError):
        SphericalMesh(radius=bad_radius)
