import numpy as np
import pytest
import xarray as xr

import parcels._sgrid as sgrid
from parcels import (
    FieldSet,
    Particle,
    ParticleFile,
    ParticleSet,
    StatusCode,
    Variable,
    VectorField,
    read_particlefile,
)
from parcels._core.index_search import _search_time_index
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.interpolators import (
    XFreeslip,
    XLinear,
    XLinearInvdistLandTracer,
    XNearest,
    XPartialslip,
)
from parcels.kernels import AdvectionRK4_3D
from tests.utils import TEST_DATA


@pytest.fixture
def field():
    """Reference data used for testing interpolation."""
    z0 = np.array(  # each x is +1 from the previous, each y is +2 from the previous
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0, 9.0],
        ]
    )
    spatial_data = np.array([z0, z0 + 3, z0 + 6, z0 + 9])  # each z is +3 from the previous
    temporal_data = np.array([spatial_data, spatial_data + 10, spatial_data + 20])  # each t is +10 from the previous

    ds = xr.Dataset(
        {"U": (["time", "depth", "lat", "lon"], temporal_data)},
        coords={
            "time": (["time"], [np.timedelta64(t, "s") for t in [0, 2, 4]], {"axis": "T"}),
            "depth": (["depth"], [0, 1, 2, 3], {"axis": "Z"}),
            "lat": (["lat"], [0, 1, 2, 3], {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "lon": (["lon"], [0, 1, 2, 3], {"axis": "X", "c_grid_axis_shift": -0.5}),
            "x": (["x"], [0.5, 1.5, 2.5, 3.5], {"axis": "X"}),
            "y": (["y"], [0.5, 1.5, 2.5, 3.5], {"axis": "Y"}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("x", "lon", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("y", "lat", sgrid.Padding.LOW),
            ),
            node_coordinates=("lon", "lat"),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.HIGH),),
        ),
    )
    field = FieldSet.from_sgrid_conventions(ds, mesh="flat").U
    assert isinstance(field.interp_method, XLinear)

    return field


@pytest.mark.parametrize(
    "interpolator, t, z, y, x, expected",
    [
        pytest.param(
            XLinear(),
            [0, 1],
            [0, 0],
            [0.49, 0.49],
            [0.51, 0.51],
            [1.49, 6.49],
            id="Linear-1",
        ),
        pytest.param(XLinear(), 1, 2.5, 0.49, 0.51, 13.99, id="Linear-2"),
        pytest.param(
            XLinear(),
            [0, 1, 1],
            [0, 0, 2.5],
            [0.49, 0.49, 0.49],
            [0.51, 0.51, 0.51],
            [1.49, 6.49, 13.99],
            id="Linear-3",
        ),
        pytest.param(XLinearInvdistLandTracer(), 1, 2.5, 0.49, 0.51, 13.99, id="LinearInvDistLand"),
        pytest.param(
            XNearest(),
            [0, 3],
            [0.2, 0.2],
            [0.2, 0.2],
            [0.51, 0.51],
            [1.0, 16.0],
            id="Nearest",
        ),
    ],
)
def test_raw_2d_interpolation(field, interpolator, t, z, y, x, expected):
    """Test the interpolation functions on the Field."""
    particle_positions = {"time": t, "z": z, "lat": y, "lon": x}
    grid_positions = field.grid.search(z, y, x)
    grid_positions.update(_search_time_index(field, t))

    value = interpolator.interp(particle_positions, grid_positions, field)
    np.testing.assert_equal(value, expected)


@pytest.mark.parametrize(
    "func, t, z, y, x, expected",
    [
        (XPartialslip(), 1, 0, 0, 0.0, [[1], [1]]),
        (XFreeslip(), 1, 0, 0.5, 1.5, [[1], [0.5]]),
        (XPartialslip(), 1, 0, 2.5, 1.5, [[0.75], [0.5]]),
        (XFreeslip(), 1, 0, 2.5, 1.5, [[1], [0.5]]),
        (XPartialslip(), 1, 0, 1.5, 0.5, [[0.5], [0.75]]),
        (XFreeslip(), 1, 0, 1.5, 0.5, [[0.5], [1]]),
        (
            XFreeslip(),
            [1, 0],
            [0, 2],
            [1.5, 1.5],
            [2.5, 0.5],
            [[0.5, 0.5], [1, 1]],
        ),
    ],
)
def test_spatial_slip_interpolation(field, func, t, z, y, x, expected):
    field.data[:] = 1.0
    field.data[:, :, 1:3, 1:3] = 0.0  # Set zero land value to test spatial slip
    U = field
    V = field
    UV = VectorField("UV", U, V, interp_method=func)

    velocities = UV[t, z, y, x]
    np.testing.assert_array_almost_equal(velocities, expected)


@pytest.mark.parametrize(
    "interpolator, t, z, y, x, expected",
    [
        (XLinearInvdistLandTracer(), 1, 0, 0.5, 0.5, 1.0),
        (XLinearInvdistLandTracer(), 1, 0, 1.5, 1.5, 0.0),
        (
            XLinearInvdistLandTracer(),
            [0, 1],
            [0, 2],
            [0.5, 0.5],
            [0.5, 0.5],
            1.0,
        ),
        (
            XLinearInvdistLandTracer(),
            [0, 1],
            [0, 2],
            [0.5, 1.5],
            [0.5, 1.5],
            [1.0, 0.0],
        ),
    ],
)
def test_invdistland_interpolation(field, interpolator, t, z, y, x, expected):
    field.data[:] = 1.0
    field.data[:, :, 1:3, 1:3] = 0  # Set NaN land value to test inv_dist
    field.interp_method = interpolator

    value = field[t, z, y, x]
    np.testing.assert_array_almost_equal(value, expected)


@pytest.mark.parametrize("mesh", ["spherical", "flat"])
def test_interpolation_mesh_type(mesh, npart=10):
    ds = simple_UV_dataset(mesh=mesh)
    ds["U"].data[:] = 1.0
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh=mesh)

    lat = 30.0
    time = 0.0
    u_expected = 1.0 if mesh == "flat" else 1.0 / (1852 * 60 * np.cos(np.radians(lat)))

    assert fieldset.U.eval(time, 0, lat, 0) == 1.0
    assert fieldset.V[time, 0, lat, 0] == 0.0

    u, v = fieldset.UV[time, 0, lat, 0]
    assert np.isclose(u, u_expected, atol=1e-7)
    assert v == 0.0


@pytest.mark.parametrize(("interp_name", "interp_method"), [("linear", XLinear)])
def test_interp_regression_v3(interp_name, interp_method):
    """Test that the v4 versions of the interpolation are the same as the v3 versions."""
    ds_input = xr.open_dataset(str(TEST_DATA / f"test_interpolation_data_random_{interp_name}.nc"))
    ydim = ds_input["U"].shape[2]
    xdim = ds_input["U"].shape[3]
    time = [np.timedelta64(int(t), "s") for t in ds_input["time"].values]

    ds = xr.Dataset(
        {
            "U": (["time", "depth", "YG", "XG"], ds_input["U"].values),
            "V": (["time", "depth", "YG", "XG"], ds_input["V"].values),
            "W": (["time", "depth", "YG", "XG"], ds_input["W"].values),
        },
        coords={
            "time": (["time"], time, {"axis": "T"}),
            "depth": (["depth"], ds_input["depth"].values, {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X", "c_grid_axis_shift": -0.5}),
            "lat": (["YG"], ds_input["lat"].values, {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "lon": (["XG"], ds_input["lon"].values, {"axis": "X", "c_grid_axis_shift": -0.5}),
        },
    ).pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.HIGH),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.HIGH),
            ),
            node_coordinates=("lon", "lat"),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "depth", sgrid.Padding.HIGH),),
        ),
    )

    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")
    assert isinstance(fieldset.U.interp_method, interp_method)
    assert isinstance(fieldset.V.interp_method, interp_method)
    assert isinstance(fieldset.W.interp_method, interp_method)

    x, y, z = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 13), np.linspace(0, 1, 5))

    TestP = Particle.add_variable(Variable("pid", dtype=np.int32, initial=0))
    pset = ParticleSet(fieldset, pclass=TestP, x=x, y=y, z=z, pid=np.arange(x.size))

    def DeleteParticle(particles, fieldset):
        any_error = particles.state >= 50  # This captures all Errors
        particles[any_error].state = StatusCode.Delete

    outfile = ParticleFile(f"test_interpolation_v4_{interp_name}.parquet", outputdt=np.timedelta64(1, "s"), mode="w")
    pset.execute(
        [AdvectionRK4_3D, DeleteParticle],
        runtime=np.timedelta64(4, "s"),
        dt=np.timedelta64(1, "s"),
        output_file=outfile,
    )

    print(str(TEST_DATA / f"test_interpolation_jit_{interp_name}.zarr"))
    ds_v3 = xr.open_zarr(str(TEST_DATA / f"test_interpolation_jit_{interp_name}.zarr"))
    ds_v4 = read_particlefile(f"test_interpolation_v4_{interp_name}.parquet")

    v3_starts = np.column_stack([ds_v3.lon[:, 0].values, ds_v3.lat[:, 0].values, ds_v3.z[:, 0].values])
    unique_starts_v3, inverse_indices = np.unique(v3_starts, axis=0, return_inverse=True)

    v4_pid_to_data = {pid: ds_v4.filter(ds_v4["particle_id"] == pid) for pid in ds_v4["particle_id"].unique()}

    for start_lon, start_lat, start_z in unique_starts_v3:
        # Find particles in v3 with this starting position
        ind_v3 = np.where(
            inverse_indices
            == np.where(
                (unique_starts_v3[:, 0] == start_lon)
                & (unique_starts_v3[:, 1] == start_lat)
                & (unique_starts_v3[:, 2] == start_z)
            )[0][0]
        )[0][0]

        # Find particles in v4 with this starting position using vectorized filter
        v4_mask = (ds_v4["x"] == start_lon) & (ds_v4["y"] == start_lat) & (ds_v4["z"] == start_z)
        ind_v4 = ds_v4.filter(v4_mask)["particle_id"].unique().to_numpy()

        v3_lon = ds_v3.lon[ind_v3, :].values
        v3_lat = ds_v3.lat[ind_v3, :].values
        v3_z = ds_v3.z[ind_v3, :].values

        # Use cached v4 data
        v4_data = v4_pid_to_data[ind_v4[0]]
        v4_lon = v4_data["x"].to_numpy()[:-1]
        v4_lat = v4_data["y"].to_numpy()[:-1]
        v4_z = v4_data["z"].to_numpy()[:-1]

        # Skip if all NaN
        if np.all(np.isnan(v3_lon)) or np.all(np.isnan(v4_lon)):
            continue

        tol = 1e-6
        np.testing.assert_allclose(v3_lon, v4_lon, atol=tol)
        np.testing.assert_allclose(v3_lat, v4_lat, atol=tol)
        np.testing.assert_allclose(v3_z, v4_z, atol=tol)
