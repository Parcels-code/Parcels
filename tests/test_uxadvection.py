import numpy as np
import pytest

import parcels
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.kernels import (
    AdvectionEE,
    AdvectionRK2,
    AdvectionRK4,
)


@pytest.mark.parametrize("integrator", [AdvectionEE, AdvectionRK2, AdvectionRK4])
def test_ux_constant_flow_face_centered_2D(integrator):
    ds = datasets_unstructured["ux_constant_flow_face_centered_2D"]
    T = np.timedelta64(3600, "s")
    dt = np.timedelta64(300, "s")
    dt_s = 300.0

    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh="flat")
    pset = parcels.ParticleSet(fieldset, lon=[5.0], lat=[5.0])
    pfile = parcels.ParticleFile(store="test.zarr", outputdt=dt)
    pset.execute(integrator, runtime=T, dt=dt, output_file=pfile, verbose_progress=False)
    expected_lon = 8.6
    np.testing.assert_allclose(pset.lon, expected_lon, atol=1e-5)
