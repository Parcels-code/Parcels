import numpy as np
import pytest
import xarray as xr

import parcels
from parcels import Field, FieldSet, Particle, ParticleFile, ParticleSet, StatusCode, Variable, VectorField, XGrid
from parcels._core.utils.time import timedelta_to_float
from parcels._datasets.structured.generated import (
    decaying_moving_eddy_dataset,
    moving_eddy_dataset,
    peninsula_dataset,
    radial_rotation_dataset,
    simple_UV_dataset,
    stommel_gyre_dataset,
)
from parcels.interpolators import CGrid_Velocity, XLinear
from parcels.kernels import (
    AdvectionDiffusionEM,
    AdvectionDiffusionM1,
    AdvectionEE,
    AdvectionRK2,
    AdvectionRK2_3D,
    AdvectionRK4,
    AdvectionRK4_3D,
    AdvectionRK45,
)
from tests.utils import DEFAULT_PARTICLES


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_advection_zonal(mesh, npart=10):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    pset = ParticleSet(fieldset, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    if mesh == "spherical":
        assert (np.diff(pset.lon) > 1.0e-4).all()
    else:
        assert (np.diff(pset.lon) < 1.0e-4).all()


def test_advection_zonal_with_particlefile(tmp_store):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    npart = 10
    ds = simple_UV_dataset(mesh="flat")
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    pset = ParticleSet(fieldset, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pfile = ParticleFile(tmp_store, outputdt=np.timedelta64(30, "m"))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"), output_file=pfile)

    assert (np.diff(pset.lon) < 1.0e-4).all()
    ds = xr.open_zarr(tmp_store)
    np.testing.assert_allclose(ds.isel(obs=-1).lon.values, pset.lon)


def periodicBC(particles, fieldset):
    particles.total_dlon += particles.dlon
    particles.lon = np.fmod(particles.lon, 2)


def test_advection_zonal_periodic():
    ds = simple_UV_dataset(dims=(2, 2, 2, 2), mesh="flat")
    ds["U"].data[:] = 0.1
    ds["lon"].data = np.array([0, 2])
    ds["lat"].data = np.array([0, 2])

    # add a halo
    halo = ds.isel(XG=0)
    halo.lon.values = ds.lon.values[1] + 1
    halo.XG.values = ds.XG.values[1] + 2
    ds = xr.concat([ds, halo], dim="XG")

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    PeriodicParticle = Particle.add_variable(Variable("total_dlon", initial=0))
    startlon = np.array([0.5, 0.4])
    pset = ParticleSet(fieldset, pclass=PeriodicParticle, lon=startlon, lat=[0.5, 0.5])
    pset.execute([AdvectionEE, periodicBC], runtime=np.timedelta64(40, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.total_dlon, 4.1, atol=1e-5)
    np.testing.assert_allclose(pset.lon, startlon, atol=1e-5)
    np.testing.assert_allclose(pset.lat, 0.5, atol=1e-5)


def test_horizontal_advection_in_3D_flow(npart=10):
    """Flat 2D zonal flow that increases linearly with z from 0 m/s to 1 m/s."""
    ds = simple_UV_dataset(mesh="flat")
    ds["U"].data[:] = 1.0
    ds["U"].data[:, 0, :, :] = 0.0  # Set U to 0 at the surface
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), z=np.linspace(0.1, 0.9, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    expected_lon = pset.z * pset.time
    np.testing.assert_allclose(pset.lon, expected_lon, atol=1.0e-1)


@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("wErrorThroughSurface", [True, False])
def test_advection_3D_outofbounds(direction, wErrorThroughSurface):
    ds = simple_UV_dataset(mesh="flat")
    ds["W"] = ds["V"].copy()  # Just to have W field present
    ds["U"].data[:] = 0.01  # Set U to small value (to avoid horizontal out of bounds)
    ds["W"].data[:] = -1.0 if direction == "up" else 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    def DeleteParticle(particles, fieldset):  # pragma: no cover
        particles.state = np.where(particles.state == StatusCode.ErrorOutOfBounds, StatusCode.Delete, particles.state)
        particles.state = np.where(
            particles.state == StatusCode.ErrorThroughSurface, StatusCode.Delete, particles.state
        )

    def SubmergeParticle(particles, fieldset):  # pragma: no cover
        if len(particles.state) == 0:
            return
        inds = np.argwhere(particles.state == StatusCode.ErrorThroughSurface).flatten()
        if len(inds) == 0:
            return
        (u, v) = fieldset.UV[particles[inds]]
        particles[inds].dlon = u * particles.dt
        particles[inds].dlat = v * particles.dt
        particles[inds].dz = 0.0
        particles[inds].z = 0
        particles[inds].state = StatusCode.Evaluate

    kernels = [AdvectionRK4_3D]
    if wErrorThroughSurface:
        kernels.append(SubmergeParticle)
    kernels.append(DeleteParticle)

    pset = ParticleSet(fieldset=fieldset, lon=0.5, lat=0.5, z=0.9)
    pset.execute(kernels, runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"))

    if direction == "up" and wErrorThroughSurface:
        np.testing.assert_allclose(pset.lon[0], 0.6, atol=1e-5)
        np.testing.assert_allclose(pset.z[0], 0, atol=1e-5)
    else:
        assert len(pset) == 0


@pytest.mark.parametrize("u", [-0.3, np.array(0.2)])
@pytest.mark.parametrize("v", [0.2, np.array(1)])
@pytest.mark.parametrize("w", [None, -0.2, np.array(0.7)])
def test_length1dimensions(u, v, w):  # TODO: Refactor this test to be more readable (and isolate test setup)
    (lon, xdim) = (np.linspace(-10, 10, 21), 21) if isinstance(u, np.ndarray) else (np.array([0]), 1)
    (lat, ydim) = (np.linspace(-15, 15, 31), 31) if isinstance(v, np.ndarray) else (np.array([-4]), 1)
    (depth, zdim) = (
        (np.linspace(-5, 5, 11), 11) if (isinstance(w, np.ndarray) and w is not None) else (np.array([3]), 1)
    )

    tdim = 2  # TODO make this also work for length-1 time dimensions
    dims = (tdim, zdim, ydim, xdim)
    U = u * np.ones(dims, dtype=np.float32)
    V = v * np.ones(dims, dtype=np.float32)
    if w is not None:
        W = w * np.ones(dims, dtype=np.float32)

    ds = xr.Dataset(
        {
            "U": (["time", "depth", "YG", "XG"], U),
            "V": (["time", "depth", "YG", "XG"], V),
        },
        coords={
            "time": (["time"], [np.timedelta64(0, "s"), np.timedelta64(10, "s")], {"axis": "T"}),
            "depth": (["depth"], depth, {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X", "c_grid_axis_shift": -0.5}),
            "lat": (["YG"], lat, {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "lon": (["XG"], lon, {"axis": "X", "c_grid_axis_shift": -0.5}),
        },
    )
    if w:
        ds["W"] = (["time", "depth", "YG", "XG"], W)

    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    fields = [U, V, VectorField("UV", U, V)]
    if w:
        W = Field("W", ds["W"], grid, interp_method=XLinear)
        fields.append(VectorField("UVW", U, V, W))
    fieldset = FieldSet(fields)

    x0, y0, z0 = 2, 8, -4
    pset = ParticleSet(fieldset, lon=x0, lat=y0, z=z0)
    kernel = AdvectionRK4 if w is None else AdvectionRK4_3D
    pset.execute(kernel, runtime=np.timedelta64(4, "s"), dt=np.timedelta64(1, "s"))

    assert len(pset.lon) == len([p.lon for p in pset])
    np.testing.assert_allclose(np.array([p.lon - x0 for p in pset]), 4 * u, atol=1e-6)
    np.testing.assert_allclose(np.array([p.lat - y0 for p in pset]), 4 * v, atol=1e-6)
    if w:
        np.testing.assert_allclose(np.array([p.z - z0 for p in pset]), 4 * w, atol=1e-6)


def test_radialrotation(npart=10):
    ds = radial_rotation_dataset()
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = parcels.Field("U", ds["U"], grid, interp_method=XLinear)
    V = parcels.Field("V", ds["V"], grid, interp_method=XLinear)
    UV = parcels.VectorField("UV", U, V)
    fieldset = parcels.FieldSet([U, V, UV])

    dt = np.timedelta64(30, "s")
    lon = np.linspace(32, 50, npart)
    lat = np.ones(npart) * 30
    starttime = np.arange(np.timedelta64(0, "s"), npart * dt, dt)
    endtime = np.timedelta64(10, "m")

    pset = parcels.ParticleSet(fieldset, lon=lon, lat=lat, time=starttime)
    pset.execute(parcels.kernels.AdvectionRK4, endtime=endtime, dt=dt)

    theta = 2 * np.pi * (pset.time - timedelta_to_float(starttime)) / (24 * 3600)
    true_lon = (lon - 30.0) * np.cos(theta) + 30.0
    true_lat = -(lon - 30.0) * np.sin(theta) + 30.0

    np.testing.assert_allclose(pset.lon, true_lon, atol=5e-2)
    np.testing.assert_allclose(pset.lat, true_lat, atol=5e-2)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionEE, 1e-2),
        (AdvectionDiffusionEM, 1e-2),
        (AdvectionDiffusionM1, 1e-2),
        (AdvectionRK2, 1e-4),
        (AdvectionRK2_3D, 1e-4),
        (AdvectionRK4, 1e-5),
        (AdvectionRK4_3D, 1e-5),
        (AdvectionRK45, 1e-4),
    ],
)
def test_moving_eddy(kernel, rtol):
    ds = moving_eddy_dataset()
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    if kernel in [AdvectionRK2_3D, AdvectionRK4_3D]:
        # Using W to test 3D advection (assuming same velocity as V)
        W = Field("W", ds["V"], grid, interp_method=XLinear)
        UVW = VectorField("UVW", U, V, W)
        fieldset = FieldSet([U, V, W, UVW])
    else:
        UV = VectorField("UV", U, V)
        fieldset = FieldSet([U, V, UV])
    if kernel in [AdvectionDiffusionEM, AdvectionDiffusionM1]:
        # Add zero diffusivity field for diffusion kernels
        ds["Kh"] = (["time", "depth", "YG", "XG"], np.full(ds["U"].shape, 0))
        fieldset.add_field(Field("Kh", ds["Kh"], grid, interp_method=XLinear), "Kh_zonal")
        fieldset.add_field(Field("Kh", ds["Kh"], grid, interp_method=XLinear), "Kh_meridional")
        fieldset.add_constant("dres", 0.1)

    start_lon, start_lat, start_z = 12000, 12500, 12500
    dt = np.timedelta64(30, "m")
    endtime = np.timedelta64(1, "h")

    if kernel == AdvectionRK45:
        fieldset.add_constant("RK45_tol", rtol)

    pset = ParticleSet(
        fieldset, pclass=DEFAULT_PARTICLES[kernel], lon=start_lon, lat=start_lat, z=start_z, time=np.timedelta64(0, "s")
    )
    pset.execute(kernel, dt=dt, endtime=endtime)

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lat = y_0 - (ds.u_0 - ds.u_g) / ds.f * (1 - np.cos(ds.f * t))
        lon = x_0 + ds.u_g * t + (ds.u_0 - ds.u_g) / ds.f * np.sin(ds.f * t)
        return lon, lat

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, endtime)
    np.testing.assert_allclose(pset.lon, exp_lon, rtol=rtol)
    np.testing.assert_allclose(pset.lat, exp_lat, rtol=rtol)
    if kernel == AdvectionRK4_3D:
        np.testing.assert_allclose(pset.z, exp_lat, rtol=rtol)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionEE, 1e-1),
        (AdvectionRK2, 3e-3),
        (AdvectionRK4, 1e-5),
        (AdvectionRK45, 1e-4),
    ],
)
def test_decaying_moving_eddy(kernel, rtol):
    ds = decaying_moving_eddy_dataset()
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    start_lon, start_lat = 10000, 10000
    dt = np.timedelta64(60, "m")
    endtime = np.timedelta64(23, "h")

    if kernel == AdvectionRK45:
        fieldset.add_constant("RK45_tol", rtol)
        fieldset.add_constant("RK45_min_dt", 10 * 60)

    pset = ParticleSet(
        fieldset, pclass=DEFAULT_PARTICLES[kernel], lon=start_lon, lat=start_lat, time=np.timedelta64(0, "s")
    )
    pset.execute(kernel, dt=dt, endtime=endtime)

    def truth_moving(x_0, y_0, t):
        t /= np.timedelta64(1, "s")
        lon = (
            x_0
            + (ds.u_g / ds.gamma_g) * (1 - np.exp(-ds.gamma_g * t))
            + ds.f
            * ((ds.u_0 - ds.u_g) / (ds.f**2 + ds.gamma**2))
            * ((ds.gamma / ds.f) + np.exp(-ds.gamma * t) * (np.sin(ds.f * t) - (ds.gamma / ds.f) * np.cos(ds.f * t)))
        )
        lat = y_0 - ((ds.u_0 - ds.u_g) / (ds.f**2 + ds.gamma**2)) * ds.f * (
            1 - np.exp(-ds.gamma * t) * (np.cos(ds.f * t) + (ds.gamma / ds.f) * np.sin(ds.f * t))
        )
        return lon, lat

    exp_lon, exp_lat = truth_moving(start_lon, start_lat, endtime)
    np.testing.assert_allclose(pset.lon, exp_lon, rtol=rtol)
    np.testing.assert_allclose(pset.lat, exp_lat, rtol=rtol)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionRK2, 0.1),
        (AdvectionRK4, 0.1),
        (AdvectionRK45, 0.1),
    ],
)
@pytest.mark.parametrize("grid_type", ["A", "C"])
def test_stommelgyre_fieldset(kernel, rtol, grid_type):
    npart = 2
    ds = stommel_gyre_dataset(grid_type=grid_type)
    grid = XGrid.from_dataset(ds, mesh="flat")
    vector_interp_method = None if grid_type == "A" else CGrid_Velocity
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    P = Field("P", ds["P"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V, vector_interp_method=vector_interp_method)
    fieldset = FieldSet([U, V, P, UV])

    dt = np.timedelta64(30, "m")
    runtime = np.timedelta64(1, "D")
    start_lon = np.linspace(10e3, 100e3, npart)
    start_lat = np.ones_like(start_lon) * 5000e3

    SampleParticle = DEFAULT_PARTICLES[kernel].add_variable(
        [Variable("p", initial=0.0, dtype=np.float32), Variable("p_start", initial=0.0, dtype=np.float32)]
    )

    if kernel == AdvectionRK45:
        fieldset.add_constant("RK45_tol", rtol)

    def UpdateP(particles, fieldset):  # pragma: no cover
        particles.p = fieldset.P[particles.time, particles.z, particles.lat, particles.lon]
        particles.p_start = np.where(particles.time == 0, particles.p, particles.p_start)

    pset = ParticleSet(fieldset, pclass=SampleParticle, lon=start_lon, lat=start_lat, time=np.timedelta64(0, "s"))
    pset.execute([kernel, UpdateP], dt=dt, runtime=runtime)
    np.testing.assert_allclose(pset.p, pset.p_start, rtol=rtol)


@pytest.mark.parametrize(
    "kernel, rtol",
    [
        (AdvectionRK2, 2e-2),
        (AdvectionRK4, 5e-3),
        (AdvectionRK45, 1e-3),
    ],
)
@pytest.mark.parametrize("grid_type", ["A"])  # TODO also implement C-grid once available
def test_peninsula_fieldset(kernel, rtol, grid_type):
    npart = 2
    ds = peninsula_dataset(grid_type=grid_type)
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U"], grid, interp_method=XLinear)
    V = Field("V", ds["V"], grid, interp_method=XLinear)
    P = Field("P", ds["P"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, P, UV])

    dt = np.timedelta64(30, "m")
    runtime = np.timedelta64(23, "h")
    start_lat = np.linspace(3e3, 47e3, npart)
    start_lon = 3e3 * np.ones_like(start_lat)

    SampleParticle = DEFAULT_PARTICLES[kernel].add_variable(
        [Variable("p", initial=0.0, dtype=np.float32), Variable("p_start", initial=0.0, dtype=np.float32)]
    )

    if kernel == AdvectionRK45:
        fieldset.add_constant("RK45_tol", rtol)

    def UpdateP(particles, fieldset):  # pragma: no cover
        particles.p = fieldset.P[particles.time, particles.z, particles.lat, particles.lon]
        particles.p_start = np.where(particles.time == 0, particles.p, particles.p_start)

    pset = ParticleSet(fieldset, pclass=SampleParticle, lon=start_lon, lat=start_lat, time=np.timedelta64(0, "s"))
    pset.execute([kernel, UpdateP], dt=dt, runtime=runtime)
    np.testing.assert_allclose(pset.p, pset.p_start, rtol=rtol)


def test_nemo_curvilinear_fieldset():
    data_folder = parcels.download_example_dataset("NemoCurvilinear_data")
    ds_fields = xr.open_mfdataset(
        data_folder.glob("*.nc4"),
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    ds_fields = ds_fields.isel(time=0, z_a=0, z=0, drop=True)
    fieldset = parcels.FieldSet.from_nemo(ds_fields)

    npart = 20
    lonp = 30 * np.ones(npart)
    latp = np.linspace(-70, 88, npart)
    runtime = np.timedelta64(160, "D")

    pset = parcels.ParticleSet(fieldset, lon=lonp, lat=latp)
    pset.execute(AdvectionEE, runtime=runtime, dt=np.timedelta64(10, "D"))
    np.testing.assert_allclose(pset.lat, latp, atol=1e-1)


@pytest.mark.parametrize(
    "kernel",
    [
        AdvectionRK4,
        pytest.mark.xfail(
            AdvectionRK4_3D,
            reason="from_nemo had 'fieldset.V.units = GeographicPolar()', I'm not sure _why_ this code is needed to get this to pass. To be further investigated.",
        ),
    ],
)
def test_nemo_3D_curvilinear_fieldset(kernel):
    data_folder = parcels.download_example_dataset("NemoNorthSeaORCA025-N006_data")
    ds_fields = xr.open_mfdataset(
        data_folder.glob("ORCA*.nc"),
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )
    ds_coords = xr.open_dataset(data_folder / "coordinates.nc", decode_times=False)
    ds_coords = ds_coords.isel(time=0, drop=True)
    ds = xr.merge([ds_fields, ds_coords[["glamf", "gphif"]]])
    fieldset = parcels.FieldSet.from_nemo(ds)

    npart = 10
    lons_initial = np.linspace(1.9, 3.4, npart)
    lats_initial = np.linspace(52.5, 51.6, npart)
    z_initial = np.ones_like(lons_initial)
    pset = parcels.ParticleSet(fieldset, lon=lons_initial, lat=lats_initial, z=z_initial)

    pset.execute(kernel, runtime=np.timedelta64(3, "D") + np.timedelta64(18, "h"), dt=np.timedelta64(6, "h"))

    if kernel == AdvectionRK4:
        np.testing.assert_allclose([p.z for p in pset], z_initial)
    elif kernel == AdvectionRK4_3D:
        # TODO check why decimals needs to be so low in RK4_3D (compare to v3)
        np.testing.assert_allclose(
            [p.z for p in pset],
            [0.666162, 0.8667131, 0.92150104, 0.9605109, 0.9577529, 1.0041442, 1.0284728, 1.0033542, 1.2949713, 1.3928112],
        )  # fmt:skip
