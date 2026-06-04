import tempfile
from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr

import parcels.tutorial
from parcels import (
    Field,
    FieldSet,
    ParticleFile,
    ParticleSet,
    ParticleSetWarning,
    StatusCode,
    Variable,
)
from parcels._core.particle import Particle, get_default_particle
from parcels._core.particlefile import _get_schema
from parcels._core.utils.time import TimeInterval, timedelta_to_float
from parcels._datasets.structured.generated import peninsula_dataset
from parcels.convert import copernicusmarine_to_sgrid
from parcels.interpolators import XLinear
from parcels.kernels import AdvectionRK4
from tests.common_kernels import DoNothing


def test_metadata(fieldset, tmp_parquet):
    pset = ParticleSet(fieldset, pclass=Particle, lon=0, lat=0)

    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    tab = pq.read_table(tmp_parquet)
    assert tab.schema.metadata[b"parcels_kernels"].decode().lower() == "DoNothing".lower()


@pytest.mark.parametrize("compression", ["zstd", "gzip", "snappy", "brotli", None])
def test_compression(fieldset, tmp_parquet, compression):
    pset = ParticleSet(fieldset, pclass=Particle, lon=0, lat=0)

    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), compression=compression)
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    tab = pq.ParquetFile(tmp_parquet)
    for i in range(tab.num_row_groups):
        row_group = tab.metadata.row_group(i)
        for j in range(row_group.num_columns):
            col = row_group.column(j)
            assert col.compression.lower() == compression or (
                compression is None and col.compression.lower() == "uncompressed"
            )


def test_write_fieldset_without_time(tmp_parquet):
    ds = peninsula_dataset()  # DataSet without time
    assert "time" not in ds.dims
    fieldset = FieldSet.from_sgrid_conventions(ds, mesh="flat")

    pset = ParticleSet(fieldset, pclass=Particle, lon=0, lat=0)

    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    table = pq.read_table(tmp_parquet)
    assert table.schema.field("time").metadata[b"units"] == b"seconds"
    assert b"calendar" not in table.schema.field("time").metadata
    assert table["time"].to_numpy()[1] == 1.0


def test_pfile_array_remove_particles(fieldset, tmp_parquet):
    """If a particle from the middle of a particleset is removed, that writing doesn't crash"""
    npart = 10
    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        lon=np.linspace(0, 1, npart),
        lat=0.5 * np.ones(npart),
        time=fieldset.time_interval.left,
    )
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset._data["time"][:] = 0
    pfile.write(pset, time=fieldset.time_interval.left)
    pset.remove_indices(3)
    new_time = 86400  # s in a day
    pset._data["time"][:] = new_time
    pfile.write(pset, new_time)
    pfile.close()


def test_pfile_array_remove_all_particles(fieldset, tmp_parquet):
    npart = 10
    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        lon=np.linspace(0, 1, npart),
        lat=0.5 * np.ones(npart),
        time=fieldset.time_interval.left,
    )
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pfile.write(pset, time=0)
    for _ in range(npart):
        pset.remove_indices(-1)
    pfile.write(pset, fieldset.time_interval.left + np.timedelta64(1, "D"))
    pfile.write(pset, fieldset.time_interval.left + np.timedelta64(2, "D"))
    pfile.close()

    df = pd.read_parquet(tmp_parquet)
    assert df["particle_id"].nunique() == npart


def test_write_dtypes_pfile(fieldset, tmp_parquet):
    dtypes = [
        np.float32,
        np.float64,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
    ]

    extra_vars = [Variable(f"v_{d.__name__}", dtype=d, initial=0.0) for d in dtypes]
    MyParticle = Particle.add_variable(extra_vars)

    pset = ParticleSet(fieldset, pclass=MyParticle, lon=0, lat=0, time=fieldset.time_interval.left)
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pfile.write(pset, time=fieldset.time_interval.left)
    pfile.close()

    tab = pq.read_table(tmp_parquet)
    for d in dtypes:
        assert tab[f"v_{d.__name__}"].type == pa.from_numpy_dtype(d)


@pytest.mark.skip(reason="Pending ParticleFile refactor; see issue #2386")
@pytest.mark.parametrize("dt", [-np.timedelta64(1, "s"), np.timedelta64(1, "s")])
@pytest.mark.parametrize("maxvar", [2, 4, 10])
def test_pset_repeated_release_delayed_adding_deleting(fieldset, tmp_parquet, dt, maxvar):
    """Tests that if particles are released and deleted based on age that resulting output file is correct."""
    npart = 10
    fieldset.add_constant("maxvar", maxvar)

    MyParticle = Particle.add_variable(
        [Variable("sample_var", initial=0.0), Variable("v_once", dtype=np.float64, initial=0.0)]
    )

    pset = ParticleSet(
        fieldset,
        lon=np.zeros(npart),
        lat=np.zeros(npart),
        pclass=MyParticle,
        time=fieldset.time_interval.left + [np.timedelta64(i + 1, "s") for i in range(npart)],
    )
    pfile = ParticleFile(tmp_parquet, outputdt=abs(dt))

    def IncrLon(particles, fieldset):  # pragma: no cover
        particles.sample_var += 1.0
        particles.state = np.where(
            particles.sample_var > fieldset.maxvar,
            StatusCode.Delete,
            particles.state,
        )

    for _ in range(npart):
        pset.execute(IncrLon, dt=dt, runtime=np.timedelta64(1, "s"), output_file=pfile)

    ds = xr.open_zarr(tmp_parquet)
    samplevar = ds["sample_var"][:]
    assert samplevar.shape == (npart, min(maxvar, npart + 1))
    # test whether samplevar[:, k] = k
    for k in range(samplevar.shape[1]):
        assert np.allclose([p for p in samplevar[:, k] if np.isfinite(p)], k + 1)


def test_file_warnings(fieldset, tmp_parquet):
    pset = ParticleSet(fieldset, lon=[0, 0], lat=[0, 0], time=[np.timedelta64(0, "s"), np.timedelta64(1, "s")])
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(2, "s"))
    with pytest.warns(ParticleSetWarning, match="Some of the particles have a start time difference.*"):
        pset.execute(AdvectionRK4, runtime=3, dt=1, output_file=pfile)


@pytest.mark.parametrize(
    "outputdt, expectation",
    [
        (np.timedelta64(5, "s"), does_not_raise()),
        (timedelta(seconds=2), does_not_raise()),
        (5.0, does_not_raise()),
        (np.datetime64("2001-01-02T00:00:00"), pytest.raises(ValueError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(ValueError)),
        (-np.timedelta64(5, "s"), pytest.raises(ValueError)),
    ],
)
def test_outputdt_types(outputdt, expectation, tmp_parquet):
    with expectation:
        pfile = ParticleFile(tmp_parquet, outputdt=outputdt)
        assert pfile.outputdt == timedelta_to_float(outputdt)


def test_write_timebackward(fieldset, tmp_parquet):
    release_time = fieldset.time_interval.left + [np.timedelta64(i + 1, "s") for i in range(3)]
    pset = ParticleSet(fieldset, lat=[0, 1, 2], lon=[0, 0, 0], time=release_time)
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(3, "s"), dt=-np.timedelta64(1, "s"), output_file=pfile)

    df = pd.read_parquet(tmp_parquet)

    assert df["particle_id"].dtype == "int64"
    assert bool(
        df.groupby("particle_id")
        .apply(
            lambda x: (np.diff(x["time"]) < 0).all()  # for each particle - set True if it has decreasing time
        )
        .all()  # ensure for all particles
    )


@pytest.mark.xfail
@pytest.mark.v4alpha
def test_write_xiyi(fieldset, tmp_parquet):
    fieldset.U.data[:] = 1  # set a non-zero zonal velocity
    fieldset.add_field(
        Field(name="P", data=np.zeros((3, 20)), lon=np.linspace(0, 1, 20), lat=[-2, 0, 2], interp_method=XLinear)
    )
    dt = np.timedelta64(3600, "s")

    particle = get_default_particle(np.float64)
    XiYiParticle = particle.add_variable(
        [
            Variable("pxi0", dtype=np.int32, initial=0.0),
            Variable("pxi1", dtype=np.int32, initial=0.0),
            Variable("pyi", dtype=np.int32, initial=0.0),
        ]
    )

    def Get_XiYi(particles, fieldset):  # pragma: no cover
        """Kernel to sample the grid indices of the particle.
        Note that this sampling should be done _before_ the advection kernel
        and that the first outputted value is zero.
        Be careful when using multiple grids, as the index may be different for the grids.
        """
        particles.pxi0 = fieldset.U.unravel_index(particles.ei)[2]
        particles.pxi1 = fieldset.P.unravel_index(particles.ei)[2]
        particles.pyi = fieldset.U.unravel_index(particles.ei)[1]

    def SampleP(particles, fieldset):  # pragma: no cover
        if np.any(particles.time > 5 * 3600):
            _ = fieldset.P[particles]  # To trigger sampling of the P field

    pset = ParticleSet(fieldset, pclass=XiYiParticle, lon=[0, 0.2], lat=[0.2, 1])
    pfile = ParticleFile(tmp_parquet, outputdt=dt)
    pset.execute([SampleP, Get_XiYi, AdvectionRK4], endtime=10 * dt, dt=dt, output_file=pfile)

    ds = xr.open_zarr(tmp_parquet)
    pxi0 = ds["pxi0"][:].values.astype(np.int32)
    pxi1 = ds["pxi1"][:].values.astype(np.int32)
    lons = ds["lon"][:].values
    pyi = ds["pyi"][:].values.astype(np.int32)
    lats = ds["lat"][:].values

    for p in range(pyi.shape[0]):
        assert (pxi0[p, 0] == 0) and (pxi0[p, -1] == pset[p].pxi0)  # check that particle has moved
        assert np.all(pxi1[p, :6] == 0)  # check that particle has not been sampled on grid 1 until time 6
        assert np.all(pxi1[p, 6:] > 0)  # check that particle has not been sampled on grid 1 after time 6
        for xi, lon in zip(pxi0[p, 1:], lons[p, 1:], strict=True):
            assert fieldset.U.grid.lon[xi] <= lon < fieldset.U.grid.lon[xi + 1]
        for xi, lon in zip(pxi1[p, 6:], lons[p, 6:], strict=True):
            assert fieldset.P.grid.lon[xi] <= lon < fieldset.P.grid.lon[xi + 1]
        for yi, lat in zip(pyi[p, 1:], lats[p, 1:], strict=True):
            assert fieldset.U.grid.lat[yi] <= lat < fieldset.U.grid.lat[yi + 1]


@pytest.mark.parametrize("outputdt", [np.timedelta64(1, "s"), np.timedelta64(2, "s"), np.timedelta64(3, "s")])
def test_time_is_age(fieldset, tmp_parquet, outputdt):
    # Test that particle age is same as time - initial_time
    npart = 10

    AgeParticle = get_default_particle(np.float64).add_variable(Variable("age", initial=0.0))

    def IncreaseAge(particles, fieldset):  # pragma: no cover
        particles.age += particles.dt

    time = fieldset.time_interval.left + np.arange(npart) * np.timedelta64(1, "s")
    pset = ParticleSet(fieldset, pclass=AgeParticle, lon=npart * [0], lat=npart * [0], time=time)
    ofile = ParticleFile(tmp_parquet, outputdt=outputdt)

    pset.execute(IncreaseAge, runtime=np.timedelta64(npart * 2, "s"), dt=np.timedelta64(1, "s"), output_file=ofile)

    df = parcels.read_particlefile(tmp_parquet)

    # Map sorted particle IDs to release times (0, 1, ..., npart-1 seconds)
    for i, df_traj in enumerate(df.partition_by("particle_id", maintain_order=True)):
        release_time = pd.Timestamp(time[i]).to_pydatetime()
        traj_time = (df_traj["time"] - release_time).dt.total_seconds()
        assert (df_traj["age"] == traj_time).all()


def test_reset_dt(fieldset, tmp_parquet):
    # Assert that p.dt gets reset when a write_time is not a multiple of dt
    # for p.dt=0.02 to reach outputdt=0.05 and endtime=0.1, the steps should be [0.2, 0.2, 0.1, 0.2, 0.2, 0.1], resulting in 6 kernel executions
    dt = np.timedelta64(20, "s")

    def Update_lon(particles, fieldset):  # pragma: no cover
        particles.dlon += 0.1

    particle = get_default_particle(np.float64)
    pset = ParticleSet(fieldset, pclass=particle, lon=[0], lat=[0])
    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(50, "s"))
    pset.execute(Update_lon, runtime=5 * dt, dt=dt, output_file=ofile)

    assert np.allclose(pset.lon, 0.6)


def test_correct_misaligned_outputdt_dt(fieldset, tmp_parquet):
    """Testing that outputdt does not need to be a multiple of dt."""

    def Update_lon(particles, fieldset):  # pragma: no cover
        particles.lon += particles.dt

    particle = get_default_particle(np.float64)
    pset = ParticleSet(fieldset, pclass=particle, lon=[0], lat=[0])
    ofile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(3, "s"))
    pset.execute(Update_lon, runtime=np.timedelta64(11, "s"), dt=np.timedelta64(2, "s"), output_file=ofile)

    df = pd.read_parquet(tmp_parquet)
    assert np.allclose(df["lon"].values, [0, 3, 6, 9])
    assert np.allclose(df.time - df.time.min(), [0, 3, 6, 9])


def setup_pset_execute(*, fieldset: FieldSet, outputdt: timedelta, execute_kwargs, particle_class=Particle):
    npart = 10

    pset = ParticleSet(
        fieldset,
        pclass=particle_class,
        lon=np.full(npart, fieldset.U.data.lon.mean()),
        lat=np.full(npart, fieldset.U.data.lat.mean()),
    )

    with tempfile.TemporaryDirectory() as dir:
        name = f"{dir}/tmp.parquet"
        output_file = ParticleFile(name, outputdt=outputdt)

        pset.execute(DoNothing, output_file=output_file, **execute_kwargs)
        df = parcels.read_particlefile(name)

    return df


def test_pset_execute_outputdt_forwards(fieldset):
    """Testing output data dt matches outputdt in forward time."""
    outputdt = timedelta(hours=1)
    runtime = timedelta(hours=5)
    dt = timedelta(minutes=5)

    df = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))
    particle_0_times = df.filter(pl.col("particle_id") == 0)["time"]
    np.testing.assert_equal(np.diff(particle_0_times) / 1e9, outputdt.seconds)


def test_pset_execute_output_time_forwards(fieldset):
    """Testing output times start at initial time and end at initial time + runtime."""
    outputdt = np.timedelta64(1, "h")
    runtime = np.timedelta64(5, "h")
    dt = np.timedelta64(5, "m")

    df = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))
    assert df["time"].min() == pd.Timestamp(fieldset.time_interval.left)
    assert df["time"].max() - df["time"].min() == runtime


def test_pset_execute_outputdt_backwards(fieldset):
    """Testing output data dt matches outputdt in backwards time."""
    outputdt = timedelta(hours=1)
    runtime = timedelta(days=2)
    dt = -timedelta(minutes=5)

    df = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))
    particle_0_times = df.filter(pl.col("particle_id") == 0)["time"]
    np.testing.assert_equal(np.diff(particle_0_times) / 1e9, -outputdt.seconds)


def test_pset_execute_outputdt_backwards_fieldset_timevarying():
    """test_pset_execute_outputdt_backwards() still passed despite #1722 as it doesn't account for time-varying fields,
    which for some reason #1722
    """
    outputdt = timedelta(hours=1)
    runtime = timedelta(days=2)
    dt = -timedelta(minutes=5)

    # TODO: Not ideal using the `open_dataset` here, but I'm struggling to recreate this error using the test suite fieldsets we have
    ds_in = parcels.tutorial.open_dataset("CopernicusMarine_data_for_Argo_tutorial/data")
    fields = {"U": ds_in["uo"], "V": ds_in["vo"]}
    ds_fset = copernicusmarine_to_sgrid(fields=fields)
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)

    df = setup_pset_execute(outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt), fieldset=fieldset)
    particle_0_times = df.filter(pl.col("particle_id") == 0)["time"]
    np.testing.assert_equal(np.diff(particle_0_times) / 1e9, -outputdt.seconds)


def test_particlefile_init(tmp_parquet):
    ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))


def test_particlefile_init_existing_path_modes(fieldset, tmp_parquet):
    pset = ParticleSet(fieldset, pclass=Particle, lon=0, lat=0)

    first_file = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=first_file)

    with pytest.raises(ValueError, match="already exists"):
        ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))

    overwrite_file = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), if_exists="overwrite")
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=overwrite_file)

    df_overwrite = pd.read_parquet(tmp_parquet)

    append_file = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), if_exists="append")
    pset.execute(DoNothing, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"), output_file=append_file)

    df_append = pd.read_parquet(tmp_parquet)
    assert len(df_append) == 2 * len(df_overwrite)


def test_particlefile_init_invalid_if_exists(tmp_parquet):
    with pytest.raises(ValueError, match="Invalid if_exists value"):
        ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"), if_exists="something-else")


@pytest.mark.parametrize("name", ["path", "outputdt"])
def test_particlefile_readonly_attrs(tmp_parquet, name):
    pfile = ParticleFile(tmp_parquet, outputdt=np.timedelta64(1, "s"))
    with pytest.raises(AttributeError, match="property .* of 'ParticleFile' object has no setter"):
        setattr(pfile, name, "something")


def test_particlefile_init_invalid(tmp_path):
    path = tmp_path / "file.not-parquet"
    with pytest.raises(ValueError, match="file extension must be '.parquet'"):
        ParticleFile(path, outputdt=np.timedelta64(1, "s"))


def test_pfile_write_custom_particle():
    # Test the writing of a custom particle with variables that are to_write, some to_write once, and some not to_write
    # ? This is more of an integration test... Should it be housed here?
    ...


@pytest.mark.xfail(
    reason="set_variable_write_status should be removed - with Particle writing defined on the particle level. GH2186"
)
def test_pfile_set_towrite_False(fieldset, tmp_parquet):
    npart = 10
    pset = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=0.5 * np.ones(npart))
    pset.set_variable_write_status("z", False)
    pset.set_variable_write_status("lat", False)
    pfile = pset.ParticleFile(tmp_parquet, outputdt=1)

    def Update_lon(particles, fieldset):  # pragma: no cover
        particles.dlon += 0.1

    pset.execute(Update_lon, runtime=10, output_file=pfile)

    ds = xr.open_zarr(tmp_parquet)
    assert "time" in ds
    assert "z" not in ds
    assert "lat" not in ds
    ds.close()

    # For pytest purposes, we need to reset to original status
    pset.set_variable_write_status("z", True)
    pset.set_variable_write_status("lat", True)


@pytest.mark.parametrize(
    "particle",
    [
        Particle,
        parcels.ParticleClass(
            variables=[
                Variable(
                    "lon",
                    dtype=np.float32,
                    attrs={"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
                ),
                Variable(
                    "lat",
                    dtype=np.float32,
                    attrs={"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
                ),
                Variable(
                    "z",
                    dtype=np.float32,
                    attrs={"standard_name": "vertical coordinate", "units": "m", "positive": "down"},
                ),
            ]
        ),
    ],
)
def test_particle_schema(particle):
    s = _get_schema(particle, {}, TimeInterval(datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 2, 12, 0)))

    written_variables = [v for v in particle.variables if v.to_write]

    assert len(s.names) == len(written_variables), (
        "Number of particles in the output schema should be the same as the writable variables in the ParticleClass object."
    )

    for variable, pyarrow_field in zip(
        written_variables,
        s,
        strict=False,
    ):
        assert variable.name == pyarrow_field.name
        if variable.name != "time":
            assert variable.attrs == {k.decode(): v.decode() for k, v in pyarrow_field.metadata.items()}
        else:
            assert b"units" in pyarrow_field.metadata
            assert b"calendar" in pyarrow_field.metadata
        assert pa.from_numpy_dtype(variable.dtype) == pyarrow_field.type
