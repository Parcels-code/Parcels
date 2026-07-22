import numpy as np
import pytest

from parcels import (
    AdvectionRK45,
    FieldSet,
    FieldSetWarning,
    KernelWarning,
    Particle,
    ParticleSet,
)
from tests.utils import TEST_DATA


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="From_pop is not supported during v4-alpha development. This will be reconsidered in v4.")
def test_fieldset_warning_pop():
    filenames = str(TEST_DATA / "POPtestdata_time.nc")
    variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
    dimensions = {"lon": "lon", "lat": "lat", "depth": "w_deps", "time": "time"}
    with pytest.warns(FieldSetWarning, match="General s-levels are not supported in B-grid.*"):
        # b-grid with s-levels and POP output in meters warning
        FieldSet.from_pop(filenames, variables, dimensions, mesh="flat")
