from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta
from operator import attrgetter

import numpy as np
import pytest
import xarray as xr

from parcels import (
    FieldSet,
    Particle,
    ParticleSet,
    ParticleSetWarning,
    Variable,
)
from parcels._datasets.structured.generated import simple_UV_dataset
from tests.common_kernels import DoNothing
from tests.utils import round_and_hash_float_array


def test_pset_create_lon_lat(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, x=lon, y=lat, pclass=Particle)
    assert np.allclose([p.x for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.y for p in pset], lat, rtol=1e-12)


def test_create_empty_pset(fieldset):
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.size == 0

    pset.execute(DoNothing, endtime=1.0, dt=1.0)
    assert pset.size == 0


def test_particleset_repr(fieldset):
    npart = 100
    pset = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.ones(npart))
    expected = """\
<ParticleSet>
    Number of particles: 100
    Particles:
        P[0]: t=nan, z=0.000000, y=1.000000, x=0.000000, particle_id=0
        P[1]: t=nan, z=0.000000, y=1.000000, x=0.010101, particle_id=1
        P[2]: t=nan, z=0.000000, y=1.000000, x=0.020202, particle_id=2
        P[3]: t=nan, z=0.000000, y=1.000000, x=0.030303, particle_id=3
        P[4]: t=nan, z=0.000000, y=1.000000, x=0.040404, particle_id=4
        P[5]: t=nan, z=0.000000, y=1.000000, x=0.050505, particle_id=5
        P[6]: t=nan, z=0.000000, y=1.000000, x=0.060606, particle_id=6
        ...
        P[99]: t=nan, z=0.000000, y=1.000000, x=1.000000, particle_id=99
    Pclass:
        Variable(name='t', dtype='float64', initial=0, to_write=True, attrs={'standard_name': 'time', 'units': 'seconds', 'axis': 'T'})
        Variable(name='z', dtype='float32', initial=0, to_write=True, attrs={'standard_name': 'vertical coordinate', 'units': 'm', 'positive': 'down'})
        Variable(name='y', dtype='float32', initial=0, to_write=True, attrs={'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'})
        Variable(name='x', dtype='float32', initial=0, to_write=True, attrs={'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'})
        Variable(name='dz', dtype='float32', initial=0, to_write=False)
        Variable(name='dy', dtype='float32', initial=0, to_write=False)
        Variable(name='dx', dtype='float32', initial=0, to_write=False)
        Variable(name='particle_id', dtype='int64', initial=0, to_write=True, attrs={'long_name': 'Unique identifier for each particle', 'cf_role': 'trajectory_id'})
        Variable(name='dt', dtype='float64', initial=1.0, to_write=False)
        Variable(name='state', dtype='int32', initial=0, to_write=False)"""
    assert str(pset) == expected

    npart = 5
    pset = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.ones(npart), t=npart * [np.timedelta64(0, "s")])
    expected = """\
<ParticleSet>
    Number of particles: 5
    Particles:
        P[0]: t=2000-01-01T00:00:00, z=0.000000, y=1.000000, x=0.000000, particle_id=0
        P[1]: t=2000-01-01T00:00:00, z=0.000000, y=1.000000, x=0.250000, particle_id=1
        P[2]: t=2000-01-01T00:00:00, z=0.000000, y=1.000000, x=0.500000, particle_id=2
        P[3]: t=2000-01-01T00:00:00, z=0.000000, y=1.000000, x=0.750000, particle_id=3
        P[4]: t=2000-01-01T00:00:00, z=0.000000, y=1.000000, x=1.000000, particle_id=4
    Pclass:
        Variable(name='t', dtype='float64', initial=0, to_write=True, attrs={'standard_name': 'time', 'units': 'seconds', 'axis': 'T'})
        Variable(name='z', dtype='float32', initial=0, to_write=True, attrs={'standard_name': 'vertical coordinate', 'units': 'm', 'positive': 'down'})
        Variable(name='y', dtype='float32', initial=0, to_write=True, attrs={'standard_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'})
        Variable(name='x', dtype='float32', initial=0, to_write=True, attrs={'standard_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'})
        Variable(name='dz', dtype='float32', initial=0, to_write=False)
        Variable(name='dy', dtype='float32', initial=0, to_write=False)
        Variable(name='dx', dtype='float32', initial=0, to_write=False)
        Variable(name='particle_id', dtype='int64', initial=0, to_write=True, attrs={'long_name': 'Unique identifier for each particle', 'cf_role': 'trajectory_id'})
        Variable(name='dt', dtype='float64', initial=1.0, to_write=False)
        Variable(name='state', dtype='int32', initial=0, to_write=False)"""
    assert str(pset) == expected


@pytest.mark.parametrize("offset", [0, 1, 200])
def test_pset_with_pids(fieldset, offset, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    trajectory_ids = np.arange(offset, npart + offset)
    pset = ParticleSet(fieldset, x=lon, y=lat, particle_ids=trajectory_ids)
    assert np.allclose([p.particle_id for p in pset], trajectory_ids, atol=1e-12)


@pytest.mark.parametrize("aslist", [True, False])
def test_pset_customvars_on_pset(fieldset, aslist):
    if aslist:
        MyParticle = Particle.add_variable([Variable("sample_var"), Variable("sample_var2")])
        pset = ParticleSet(fieldset, x=0, y=0, pclass=MyParticle, sample_var=5.0, sample_var2=10.0)
    else:
        MyParticle = Particle.add_variable(Variable("sample_var"))
        pset = ParticleSet(fieldset, x=0, y=0, pclass=MyParticle, sample_var=5.0)

    pset.execute(DoNothing, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert np.allclose([p.sample_var for p in pset], 5.0)
    if aslist:
        assert np.allclose([p.sample_var2 for p in pset], 10.0)


def test_pset_custominit_on_pset_attrgetter(fieldset):
    MyParticle = Particle.add_variable(Variable("sample_var", initial=attrgetter("x")))

    pset = ParticleSet(fieldset, x=3, y=0, pclass=MyParticle)

    pset.execute(DoNothing, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert np.allclose([p.sample_var for p in pset], 3.0)


@pytest.mark.parametrize("pset_override", [True, False])
def test_pset_custominit_on_pclass(fieldset, pset_override):
    MyParticle = Particle.add_variable(Variable("sample_var", initial=4))

    if pset_override:
        pset = ParticleSet(fieldset, x=0, y=0, pclass=MyParticle, sample_var=5)
    else:
        pset = ParticleSet(fieldset, x=0, y=0, pclass=MyParticle)

    pset.execute(DoNothing, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))

    check_val = 5.0 if pset_override else 4.0
    assert np.allclose([p.sample_var for p in pset], check_val)


@pytest.mark.parametrize(
    "time, expectation",
    [
        (np.timedelta64(0, "ns"), does_not_raise()),
        (np.datetime64("2000-01-02T00:00:00"), does_not_raise()),
        (datetime(2000, 1, 1, 0, 0, 0), does_not_raise()),
        (datetime(2000, 1, 1, 0, 0, 0).date(), does_not_raise()),
        (timedelta(seconds=0), does_not_raise()),
        (0.0, pytest.raises(TypeError)),
        ("invalid_time_string", pytest.raises(TypeError)),
    ],
)
def test_particleset_init_time_type(fieldset, time, expectation):
    with expectation:
        ParticleSet(fieldset, x=[0.2], y=[5.0], t=[time], pclass=Particle)


def test_pset_create_outside_time(fieldset):
    time = xr.date_range("1999", "2001", 20)
    with pytest.warns(ParticleSetWarning, match="Some particles are set to be released*"):
        ParticleSet(fieldset, pclass=Particle, x=[0] * len(time), y=[0] * len(time), t=time)


def test_pset_starttime_not_multiple_dt(fieldset):
    times = [0, 1, 2]
    datetimes = [fieldset.time_interval.left + np.timedelta64(t, "s") for t in times]
    pset = ParticleSet(fieldset, x=[0] * len(times), y=[0] * len(times), pclass=Particle, t=datetimes)

    def Addlon(particles, fieldset):  # pragma: no cover
        particles.dx += particles.dt

    pset.execute(Addlon, dt=np.timedelta64(2, "s"), runtime=np.timedelta64(8, "s"), verbose_progress=False)
    assert np.allclose([p.x + p.dx for p in pset], [8 - t for t in times])


def test_populate_indices(fieldset):
    npart = 11
    pset = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.linspace(1, 0, npart))
    pset.populate_indices()
    np.testing.assert_equal(round_and_hash_float_array(pset.ei, decimals=0), 935996932384571063274191)


def test_pset_add_explicit(fieldset):
    npart = 11
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, x=lon[0], y=lat[0], pclass=Particle)
    for i in range(1, npart):
        particle = ParticleSet(pclass=Particle, x=lon[i], y=lat[i], fieldset=fieldset)
        pset.add(particle)
    assert len(pset) == npart
    assert np.allclose([p.x for p in pset], lon, atol=1e-12)
    assert np.allclose([p.y for p in pset], lat, atol=1e-12)
    assert np.allclose(np.diff(pset._data["particle_id"]), np.ones(pset._data["particle_id"].size - 1), atol=1e-12)


def test_pset_add_implicit(fieldset):
    pset = ParticleSet(fieldset, x=np.zeros(3), y=np.ones(3), pclass=Particle)
    pset += ParticleSet(fieldset, x=np.ones(4), y=np.zeros(4), pclass=Particle)
    assert len(pset) == 7
    assert np.allclose(np.diff(pset._data["particle_id"]), np.ones(6), atol=1e-12)


def test_pset_add_implicit_in_loop(fieldset, npart=10):
    pset = ParticleSet(fieldset, x=[], y=[])
    for _ in range(npart):
        pset += ParticleSet(pclass=Particle, x=0.1, y=0.1, fieldset=fieldset)
    assert pset.size == npart


def test_pset_merge_inplace(fieldset, npart=100):
    pset1 = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.linspace(1, 0, npart))
    pset2 = ParticleSet(fieldset, x=np.linspace(0, 1, npart), y=np.linspace(0, 1, npart))
    assert pset1.size == npart
    assert pset2.size == npart
    pset1.add(pset2)
    assert pset1.size == 2 * npart


def test_pset_remove_index(fieldset, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, x=lon, y=lat)
    indices_to_remove = [0, 10, 20]
    pset.remove_indices(indices_to_remove)
    assert pset.size == 97
    assert not np.any(np.isin(pset.particle_id, indices_to_remove))


def test_pset_iterator(fieldset):
    npart = 10
    pset = ParticleSet(fieldset, x=np.zeros(npart), y=np.ones(npart))
    for i, particle in enumerate(pset):
        assert particle.particle_id == i
    assert i == npart - 1


@pytest.mark.parametrize(
    "depths",
    [
        pytest.param(np.linspace(1, 10, 10), id="all_depths_positive"),
        pytest.param(np.linspace(-10, -1, 10), id="all_depths_negative"),
    ],
)
def test_pset_default_z_is_in_domain(depths):
    ds = simple_UV_dataset(n_times=1, nz=len(depths), ny=10, nx=10, mesh_str="flat")
    ds = ds.assign_coords(depth=depths)
    fieldset = FieldSet.from_sgrid_conventions(ds)

    pset = ParticleSet(fieldset, x=[0], y=[0])
    expected_z = depths[np.argmin(np.abs(depths))]
    assert np.isclose(pset.z[0], expected_z)


@pytest.mark.parametrize(
    "depths",
    [
        pytest.param(np.concatenate([np.linspace(-15, -1, 5), np.linspace(0, 2, 5)]), id="depths_include_zero"),
        pytest.param(np.concatenate([np.linspace(-9, -3, 3), np.linspace(2, 8, 3)]), id="closest_depth_is_positive"),
        pytest.param(np.concatenate([np.linspace(-8, -2, 3), np.linspace(3, 9, 3)]), id="closest_depth_is_negative"),
    ],
)
def test_pset_default_z_closest_to_zero(depths):
    ds = simple_UV_dataset(n_times=1, nz=len(depths), ny=10, nx=10, mesh_str="flat")
    ds = ds.assign_coords(depth=depths)
    fieldset = FieldSet.from_sgrid_conventions(ds)

    pset = ParticleSet(fieldset, x=[0], y=[0])
    expected_z = depths[np.argmin(np.abs(depths))]
    assert np.isclose(pset.z[0], expected_z)


@pytest.mark.parametrize("npart", [1, 10])
@pytest.mark.parametrize("witht", [True, False])
def test_sampling_pset(fieldset, npart, witht):
    # Test that inital value of a field gets sampled
    fieldset.U.data[:] = 2.0

    x = np.zeros(npart)
    y = np.zeros(npart)
    if witht:
        t = npart * [np.timedelta64(0, "s")]
        pset = ParticleSet(fieldset, x=x, y=y, t=t)
        pset.sample, _ = fieldset.UV[pset]
        np.testing.assert_allclose(pset.sample, 2.0, rtol=1e-12)
    else:
        with pytest.raises(ValueError, match="Time values for particles with indices .* cannot be NaN."):
            pset = ParticleSet(fieldset, x=x, y=y)
            pset.sample, _ = fieldset.UV[pset]
