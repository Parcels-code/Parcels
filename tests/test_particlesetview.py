import numpy as np
import pytest

from parcels import Field, FieldSet, Particle, ParticleSet, Variable, VectorField, XGrid
from parcels._core.statuscodes import StatusCode
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels.interpolators import XLinear


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U_A_grid"], grid, interp_method=XLinear)
    V = Field("V", ds["V_A_grid"], grid, interp_method=XLinear)
    UV = VectorField("UV", U, V)
    return FieldSet([U, V, UV])


def test_execution_changing_particle_mask(fieldset):
    """Test that particle masks can change during kernel execution."""
    npart = 10
    initial_lons = np.linspace(0, 1, npart)
    pset = ParticleSet(fieldset, lon=initial_lons.copy(), lat=np.zeros(npart))

    def IncrementLowLon(particles, fieldset):  # pragma: no cover
        # Increment lon for particles with lon < 0.5
        # The mask changes as particles cross the threshold
        particles[particles.lon < 0.5].dlon += 0.1

    pset.execute(IncrementLowLon, runtime=np.timedelta64(5, "s"), dt=np.timedelta64(1, "s"))

    # Particles that started below 0.5 should have moved more
    # Particles that started above 0.5 should not have moved
    particles_started_low = initial_lons < 0.5
    particles_started_high = initial_lons >= 0.5

    # Low particles should have increased lon
    assert np.all(pset.lon[particles_started_low] > initial_lons[particles_started_low])
    # High particles should not have moved
    assert np.allclose(pset.lon[particles_started_high], initial_lons[particles_started_high], atol=1e-6)


def test_particle_mask_conditional_state_changes(fieldset):
    """Test setting particle state based on a condition using particle masks."""
    npart = 10
    initial_lons = np.linspace(0, 1, npart)
    pset = ParticleSet(fieldset, lon=initial_lons.copy(), lat=np.zeros(npart))

    def StopFastParticles(particles, fieldset):  # pragma: no cover
        # Stop particles that have moved beyond lon=0.5
        particles[particles.lon > 0.5].state = StatusCode.StopExecution

    def AdvanceLon(particles, fieldset):  # pragma: no cover
        particles.dlon += 0.2

    pset.execute([AdvanceLon, StopFastParticles], runtime=np.timedelta64(5, "s"), dt=np.timedelta64(1, "s"))

    # All particles should have stopped when they crossed lon > 0.5
    # Verify all final positions are > 0.5 (since they stop after crossing)
    assert np.all(pset.lon > 0.5)
    # Particles that started closer to 0.5 should have stopped sooner (lower final lon)
    # while particles that started farther should have moved more before stopping
    assert pset.lon[0] < pset.lon[-1]  # First particle stopped earliest, last stopped latest


def test_particle_mask_conditional_updates(fieldset):
    """Test applying different updates to different particle subsets using masks."""
    npart = 20
    MyParticle = Particle.add_variable(Variable("temp", initial=10.0))
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart), pclass=MyParticle)

    def ConditionalHeating(particles, fieldset):  # pragma: no cover
        # Warm particles on the left, cool particles on the right
        particles[particles.lon < 0.5].temp += 1.0
        particles[particles.lon >= 0.5].temp -= 0.5

    pset.execute(ConditionalHeating, runtime=np.timedelta64(4, "s"), dt=np.timedelta64(1, "s"))

    # After 5 timesteps (0, 1, 2, 3, 4): left particles should be at 15.0, right at 7.5
    left_particles = pset.lon < 0.5
    right_particles = pset.lon >= 0.5
    assert np.allclose(pset.temp[left_particles], 15.0, atol=1e-6)
    assert np.allclose(pset.temp[right_particles], 7.5, atol=1e-6)


def test_particle_mask_progressive_changes(fieldset):
    """Test masks that change dynamically as particle properties change during execution."""
    npart = 10
    # Start all particles at lon=0, they will progressively move right
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.linspace(0, 1, npart))

    def MoveAndStopAtBoundary(particles, fieldset):  # pragma: no cover
        # Move all particles right
        particles.dlon += 0.15
        # Stop particles that cross lon=0.5
        particles[particles.lon + particles.dlon > 0.5].state = StatusCode.StopExecution

    pset.execute(MoveAndStopAtBoundary, runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"))

    # All particles should have stopped at or before lon=0.5
    # After first step: all reach 0.15
    # After second step: all reach 0.30
    # After third step: all reach 0.45
    # After fourth step: all would reach 0.60, so they stop
    assert np.all(pset.lon <= 0.6)
    assert np.all(pset.lon >= 0.45)  # At least 3 steps completed


def test_particle_mask_multiple_sequential_operations(fieldset):
    """Test applying multiple different mask operations in sequence within one kernel."""
    npart = 30
    MyParticle = Particle.add_variable([Variable("group", initial=0), Variable("counter", initial=0)])

    # Divide particles into three groups by initial position
    lons = np.linspace(0, 1, npart)
    pset = ParticleSet(fieldset, lon=lons, lat=np.zeros(npart), pclass=MyParticle)

    def MultiMaskOperations(particles, fieldset):  # pragma: no cover
        # Classify particles into groups based on lon
        particles[particles.lon < 0.33].group = 1
        particles[(particles.lon >= 0.33) & (particles.lon < 0.67)].group = 2
        particles[particles.lon >= 0.67].group = 3

        # Apply different operations to each group
        particles[particles.group == 1].counter += 1
        particles[particles.group == 2].counter += 2
        particles[particles.group == 3].counter += 3

    pset.execute(MultiMaskOperations, runtime=np.timedelta64(5, "s"), dt=np.timedelta64(1, "s"))

    # Verify groups were assigned correctly and counters incremented appropriately
    group1 = pset.lon < 0.33
    group2 = (pset.lon >= 0.33) & (pset.lon < 0.67)
    group3 = pset.lon >= 0.67

    assert np.allclose(pset.counter[group1], 6, atol=1e-6)  # 6 timesteps * 1
    assert np.allclose(pset.counter[group2], 12, atol=1e-6)  # 6 timesteps * 2
    assert np.allclose(pset.counter[group3], 18, atol=1e-6)  # 6 timesteps * 3


def test_particle_mask_empty_mask_handling(fieldset):
    """Test that kernels handle empty masks (no particles matching condition) correctly."""
    npart = 10
    MyParticle = Particle.add_variable(Variable("modified", initial=0))
    # All particles start at lon > 0
    pset = ParticleSet(fieldset, lon=np.linspace(0.1, 1.0, npart), lat=np.zeros(npart), pclass=MyParticle)

    def ModifyNegativeLon(particles, fieldset):  # pragma: no cover
        # This mask should be empty (no particles have lon < 0)
        particles[particles.lon < 0].modified = 1
        # This should affect all particles
        particles.dlon += 0.01

    # Should execute without errors even though the first mask is always empty
    pset.execute(ModifyNegativeLon, runtime=np.timedelta64(3, "s"), dt=np.timedelta64(1, "s"))

    # No particles should have been modified
    assert np.all(pset.modified == 0)
    # But all should have moved
    assert np.all(pset.lon > 0.1)


def test_particle_mask_with_delete_state(fieldset):
    """Test using particle masks to delete particles based on conditions."""
    npart = 20
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))
    initial_size = pset.size

    def DeleteEdgeParticles(particles, fieldset):  # pragma: no cover
        # Delete particles at the edges
        particles[(particles.lon < 0.2) | (particles.lon > 0.8)].state = StatusCode.Delete

    def MoveLon(particles, fieldset):  # pragma: no cover
        particles.dlon += 0.01

    pset.execute([DeleteEdgeParticles, MoveLon], runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))

    # Should have deleted edge particles
    assert pset.size < initial_size
    # Remaining particles should be in the middle range
    assert np.all((pset.lon >= 0.2) & (pset.lon <= 0.8))
