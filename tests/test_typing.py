import numpy as np
import pytest
import xarray as xr

from parcels._typing import (
    assert_valid_mesh,
)


def test_invalid_assert_valid_mesh():
    with pytest.raises(ValueError, match="Invalid value"):
        assert_valid_mesh("invalid option")

    ds = xr.Dataset({"A": (("a", "b"), np.arange(20).reshape(4, 5))})
    with pytest.raises(ValueError, match="Invalid input type"):
        assert_valid_mesh(ds)


def test_assert_valid_mesh():
    assert_valid_mesh("spherical")
