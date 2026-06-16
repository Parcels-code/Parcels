import numpy as np
import pytest
import xarray as xr

from parcels import (
    AdvectionAnalytical,
    AdvectionDiffusionEM,
    AdvectionDiffusionM1,
    AdvectionEE,
    AdvectionRK4,
    AdvectionRK45,
    FieldSet,
    Particle,
    ParticleSet,
)
from tests.utils import TEST_DATA

kernel = {
    "EE": AdvectionEE,
    "RK4": AdvectionRK4,
    "RK45": AdvectionRK45,
    "AA": AdvectionAnalytical,
    "AdvDiffEM": AdvectionDiffusionEM,
    "AdvDiffM1": AdvectionDiffusionM1,
}


@pytest.fixture
def lon():
    xdim = 200
    return np.linspace(-170, 170, xdim, dtype=np.float32)


@pytest.fixture
def lat():
    ydim = 100
    return np.linspace(-80, 80, ydim, dtype=np.float32)


@pytest.fixture
def depth():
    zdim = 2
    return np.linspace(0, 30, zdim, dtype=np.float32)


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="When refactoring fieldfilebuffer croco support was dropped. This will be fixed in v4.")
def test_advection_2DCROCO():
    fieldset = FieldSet.from_modulefile(TEST_DATA / "fieldset_CROCO2D.py")

    runtime = 1e4
    X = np.array([40e3, 80e3, 120e3])
    Y = np.ones(X.size) * 100e3
    Z = np.zeros(X.size)
    pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=X, lat=Y, depth=Z)

    pset.execute([AdvectionRK4], runtime=runtime, dt=100)
    assert np.allclose(pset.depth, Z.flatten(), atol=1e-3)
    assert np.allclose(pset.lon_nextloop, [x + runtime for x in X], atol=1e-3)
