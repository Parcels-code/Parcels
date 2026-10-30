import numpy as np
import pytest

from parcels import (
    FieldSet,
    KernelWarning,
    Particle,
    ParticleFile,
    ParticleSet,
    StatusCode,
    Variable,
)
from parcels._core.kernel import Kernel
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels.kernels import AdvectionRK4, AdvectionRK45
from tests.common_kernels import DoNothing, MoveEast, MoveNorth
from tests.mark import ignore_kernel_warnings


def test_unknown_var_in_kernel(fieldset):
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])

    def ErrorKernel(particles, fieldset):  # pragma: no cover
        particles.unknown_varname += 0.2

    with pytest.raises(KeyError, match="'unknown_varname'"):
        pset.execute(ErrorKernel, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))


@ignore_kernel_warnings
def test_context_in_kernel(fieldset):
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])

    fieldset.add_context("fix_lon", -0.5)

    def ContextKernel(particles, fieldset):
        particles.x = fieldset.fix_lon

    pset.execute(ContextKernel, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))
    assert pset.x == -0.5


@ignore_kernel_warnings
def test_func_context_in_kernel(fieldset):
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])

    def ContextFunc(x):
        return 2 * x

    fieldset.add_context("func", ContextFunc)

    def FuncContextKernel(particles, fieldset):
        particles.x = fieldset.func(particles.x)

    pset.execute(FuncContextKernel, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))
    assert pset.x == 2.0


def test_kernel_init(fieldset):
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])
    Kernel(kernels=[AdvectionRK4], pset=pset)


def test_kernel_merging(fieldset):
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])
    merged_kernel = Kernel(kernels=[AdvectionRK4, MoveEast, MoveNorth], pset=pset)
    assert merged_kernel.funcname == "AdvectionRK4MoveEastMoveNorth"
    assert len(merged_kernel._kernels) == 3
    assert merged_kernel._kernels == [AdvectionRK4, MoveEast, MoveNorth]

    merged_kernel = Kernel(kernels=[MoveEast, MoveNorth, AdvectionRK4], pset=pset)
    assert merged_kernel.funcname == "MoveEastMoveNorthAdvectionRK4"
    assert len(merged_kernel._kernels) == 3
    assert merged_kernel._kernels == [MoveEast, MoveNorth, AdvectionRK4]


def test_kernel_from_list(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that a Kernel can be created from a list functions, or a list of
    mixed functions and kernel objects.
    """
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])
    kernels_single = Kernel(kernels=[AdvectionRK4], pset=pset)
    kernels_functions = Kernel(kernels=[AdvectionRK4, MoveEast, MoveNorth], pset=pset)

    # Check if the kernels were combined correctly
    assert kernels_single.funcname == "AdvectionRK4"
    assert kernels_functions.funcname == "AdvectionRK4MoveEastMoveNorth"


def test_kernel_from_list_error_checking(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that various error cases raise appropriate messages.
    """
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])

    with pytest.raises(ValueError, match="List of `kernels` should have at least one function."):
        Kernel(kernels=[], pset=pset)

    with pytest.raises(TypeError, match=r"Argument `kernels` should be a function or list of functions.*"):
        Kernel(kernels=[AdvectionRK4, "something else"], pset=pset)

    with pytest.raises(TypeError, match=r".* should be a function or list of functions.*"):
        kernels_mixed = Kernel(kernels=[Kernel(kernels=[AdvectionRK4], pset=pset), MoveEast, MoveNorth], pset=pset)
        assert kernels_mixed.funcname == "AdvectionRK4MoveEastMoveNorth"


def test_RK45Kernel_error_no_next_dt(fieldset):
    """Tests that kernel throws error if Particle class does not have next_dt for RK45"""
    pset = ParticleSet(fieldset, x=[0.5], y=[0.5])

    with pytest.raises(ValueError, match='ParticleClass requires a "next_dt" for AdvectionRK45 Kernel.'):
        Kernel(kernels=[AdvectionRK45], pset=pset)


def test_rk45_kernel_warnings(fieldset):
    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle.add_variable(Variable("next_dt", dtype=np.float32, initial=1)),
        x=[0],
        y=[0],
        next_dt=1,
    )
    with pytest.warns(KernelWarning):
        pset.execute(AdvectionRK45, runtime=1, dt=1)


@ignore_kernel_warnings
@pytest.mark.parametrize("kernel_type", ["update_lon", "update_dlon"])
def test_execution_order(kernel_type):
    ds = simple_UV_dataset(n_times=1, nz=1, ny=2, nx=2, mesh_str="flat")
    ds["U"].data[:, :] = [[0, 1], [2, 3]]
    ds["lon"].data = [0, 2]
    fieldset = FieldSet.from_sgrid_conventions(ds)

    def MoveLon_Update_X(particles, fieldset):  # pragma: no cover
        particles.x += 0.2

    def MoveLon_Update_DX(particles, fieldset):  # pragma: no cover
        particles.dx += 0.2

    def SampleP(particles, fieldset):  # pragma: no cover
        particles.p, _ = fieldset.UV[particles]

    SampleParticle = Particle.add_variable(Variable("p", dtype=np.float32, initial=0.0))

    MoveLon = MoveLon_Update_DX if kernel_type == "update_dlon" else MoveLon_Update_X

    kernels = [MoveLon, SampleP]
    lons = []
    ps = []
    for dir in [1, -1]:
        pset = ParticleSet(fieldset, pclass=SampleParticle, x=0, y=0)
        pset.execute(kernels[::dir], runtime=1, dt=1)
        lons.append(pset.x)
        ps.append(pset.p)

    if kernel_type == "update_dlon":
        assert np.isclose(lons[0], lons[1])
        assert np.isclose(ps[0], ps[1])
        assert np.allclose(lons[0], 0.2)
    else:
        assert np.isclose(ps[0] - ps[1], 0.1)
        assert np.allclose(lons[0], 0.2)


@pytest.mark.parametrize(("with_output", "expected_num_kernel_loops"), [(True, 7), (False, 5)])
def test_dt_modify_in_kernel(fieldset, tmp_parquet, with_output, expected_num_kernel_loops):
    npart = 10
    TestParticle = Particle.add_variable(Variable("num_kernel_loops", dtype=np.float32, initial=0))
    pset = ParticleSet(fieldset, pclass=TestParticle, x=np.linspace(0, 1, npart), y=[0] * npart)

    def ModifyDt(particles, fieldset):  # pragma: no cover
        particles.num_kernel_loops += 1
        particles.dt = 4
        particles[particles.x > 0.8].state = StatusCode.Delete

    runtime = 17
    output = ParticleFile(tmp_parquet, outputdt=5.0) if with_output else None
    pset.execute(ModifyDt, runtime=runtime, dt=1.0, output_file=output, verbose_progress=False)

    np.testing.assert_allclose(pset.t, runtime)
    np.testing.assert_allclose(pset.num_kernel_loops, expected_num_kernel_loops)


def test_dt_modify_in_kernel_vector(fieldset):
    TestParticle = Particle.add_variable(Variable("num_kernel_loops", dtype=np.float32, initial=0))
    pset = ParticleSet(fieldset, pclass=TestParticle, x=[0.5], y=[0])

    dt_in_kernel = np.array([1, 4, 3, 4, 5, 5, 10, 2, 2])
    fieldset.add_context("dt_in_kernel", dt_in_kernel)

    def ModifyDt_FromArray(particles, fieldset):  # pragma: no cover
        i = np.asarray(particles.num_kernel_loops).astype(int)
        particles.dt = fieldset.dt_in_kernel[i]
        assert particles.t == np.cumsum(np.r_[0, fieldset.dt_in_kernel])[i]
        particles.num_kernel_loops += 1

    runtime = sum(dt_in_kernel)
    expected_num_kernel_loops = len(dt_in_kernel)
    pset.execute(ModifyDt_FromArray, runtime=runtime, dt=1.0)

    np.testing.assert_allclose(pset.t[0], runtime)
    np.testing.assert_allclose(pset.num_kernel_loops[0], expected_num_kernel_loops)


@pytest.mark.parametrize("dt", [1e-2, 1e-5, 1e-6, 1e-9])
def test_small_dt(fieldset, dt):
    pset = ParticleSet(fieldset, x=[0], y=[0])

    pset.execute(DoNothing, dt=dt, runtime=dt * 100)
    assert np.allclose([p.t for p in pset], dt * 100)
