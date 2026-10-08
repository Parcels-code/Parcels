import numpy as np
import pandas as pd
import pytest

import parcels
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.kernels import (
    AdvectionEE,
    AdvectionRK2,
    AdvectionRK4,
)


@pytest.mark.parametrize("integrator", [AdvectionEE, AdvectionRK2, AdvectionRK4])
def test_ux_constant_flow_face_centered_2D(integrator, tmp_parquet):
    ds = datasets_unstructured["ux_constant_flow_face_centered_2D"]
    T = np.timedelta64(3600, "s")
    dt = np.timedelta64(300, "s")

    fieldset = parcels.FieldSet.from_ugrid_conventions(ds, mesh="flat")
    pset = parcels.ParticleSet(fieldset, x=[5.0], y=[5.0])
    pfile = parcels.ParticleFile(path=tmp_parquet, outputdt=dt)
    pset.execute(integrator, runtime=T, dt=dt, output_file=pfile, verbose_progress=False)
    expected_lon = 8.6
    np.testing.assert_allclose(pset.x, expected_lon, atol=1e-5)

    df = pd.read_parquet(tmp_parquet)
    np.testing.assert_allclose(df["x"].iloc[-1], expected_lon, atol=1e-5)
