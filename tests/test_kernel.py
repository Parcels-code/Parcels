import numpy as np
import pytest

from parcels import (
    Field,
    FieldSet,
    Kernel,
    Particle,
    ParticleSet,
    XGrid,
)
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels.interpolators import XLinear
from parcels.kernels import AdvectionRK4, AdvectionRK45
from tests.common_kernels import MoveEast, MoveNorth


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U_A_grid"], grid, interp_method=XLinear)
    V = Field("V", ds["V_A_grid"], grid, interp_method=XLinear)
    return FieldSet([U, V])


def test_unknown_var_in_kernel(fieldset):
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    def ErrorKernel(particles, fieldset):  # pragma: no cover
        particles.unknown_varname += 0.2

    with pytest.raises(KeyError, match="'unknown_varname'"):
        pset.execute(ErrorKernel, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))


def test_kernel_init(fieldset):
    Kernel(kernels=[AdvectionRK4], fieldset=fieldset, ptype=Particle)


def test_kernel_merging(fieldset):
    k1 = Kernel(kernels=[AdvectionRK4], fieldset=fieldset, ptype=Particle)
    k2 = Kernel(kernels=[MoveEast, MoveNorth], fieldset=fieldset, ptype=Particle)

    merged_kernel = k1 + k2
    assert merged_kernel.funcname == "AdvectionRK4MoveEastMoveNorth"
    assert len(merged_kernel._kernels) == 3
    assert merged_kernel._kernels == [AdvectionRK4, MoveEast, MoveNorth]

    merged_kernel = k2 + k1
    assert merged_kernel.funcname == "MoveEastMoveNorthAdvectionRK4"
    assert len(merged_kernel._kernels) == 3
    assert merged_kernel._kernels == [MoveEast, MoveNorth, AdvectionRK4]


def test_kernel_from_list(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that a Kernel can be created from a list functions, or a list of
    mixed functions and kernel objects.
    """
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])
    kernels_single = Kernel(kernels=[AdvectionRK4], fieldset=fieldset, ptype=pset._ptype)
    kernels_functions = Kernel(kernels=[AdvectionRK4, MoveEast, MoveNorth], fieldset=fieldset, ptype=pset._ptype)

    # Check if the kernels were combined correctly
    assert kernels_single.funcname == "AdvectionRK4"
    assert kernels_functions.funcname == "AdvectionRK4MoveEastMoveNorth"


def test_kernel_from_list_error_checking(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that various error cases raise appropriate messages.
    """
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    with pytest.raises(ValueError, match="List of `kernels` should have at least one function."):
        Kernel(kernels=[], fieldset=fieldset, ptype=pset._ptype)

    with pytest.raises(TypeError, match=r"Argument `kernels` should be a function or list of functions.*"):
        Kernel(kernels=[AdvectionRK4, "something else"], fieldset=fieldset, ptype=pset._ptype)

    with pytest.raises(TypeError, match=r".*is not a callable object"):
        kernels_mixed = Kernel(
            kernels=[Kernel(kernels=[AdvectionRK4], fieldset=fieldset, ptype=pset._ptype), MoveEast, MoveNorth],
            fieldset=fieldset,
            ptype=pset._ptype,
        )
        assert kernels_mixed.funcname == "AdvectionRK4MoveEastMoveNorth"


def test_RK45Kernel_error_no_next_dt(fieldset):
    """Tests that kernel throws error if Particle class does not have next_dt for RK45"""
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    with pytest.raises(ValueError, match='ParticleClass requires a "next_dt" for AdvectionRK45 Kernel.'):
        Kernel(kernels=[AdvectionRK45], fieldset=fieldset, ptype=pset._ptype)


def test_kernel_signature(fieldset):
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    def good_kernel(particles, fieldset):
        pass

    def version_3_kernel(particle, fieldset, time):
        pass

    def version_3_kernel_without_time(particle, fieldset):
        pass

    def kernel_switched_args(fieldset, particle):
        pass

    def kernel_with_forced_kwarg(particles, *, fieldset=0):
        pass

    Kernel(kernels=[good_kernel], fieldset=fieldset, ptype=pset._ptype)

    with pytest.raises(ValueError, match="Kernel function must have 2 parameters, got 3"):
        Kernel(kernels=[version_3_kernel], fieldset=fieldset, ptype=pset._ptype)

    with pytest.raises(
        ValueError, match="Parameter 'particle' has incorrect name. Expected 'particles', got 'particle'"
    ):
        Kernel(kernels=[version_3_kernel_without_time], fieldset=fieldset, ptype=pset._ptype)

    with pytest.raises(
        ValueError, match="Parameter 'fieldset' has incorrect name. Expected 'particles', got 'fieldset'"
    ):
        Kernel(kernels=[kernel_switched_args], fieldset=fieldset, ptype=pset._ptype)

    with pytest.raises(
        ValueError,
        match="Parameter 'fieldset' has incorrect parameter kind. Expected POSITIONAL_OR_KEYWORD, got KEYWORD_ONLY",
    ):
        Kernel(kernels=[kernel_with_forced_kwarg], fieldset=fieldset, ptype=pset._ptype)
