import numpy as np
import pytest
import xarray as xr

from parcels._datasets import utils
from parcels._datasets.structured.generic import datasets


@pytest.fixture
def nonzero_ds():
    """Small dataset with nonzero data_vars and non-index coords for replace_arrays_with_zeros tests.

    Uses 2D lon/lat as coords so they are regular (non-index) variables that can be zeroed.
    """
    import dask.array as da

    lon = np.array([[1.0, 2.0, 3.0, 4.0]] * 3)
    lat = np.array([[10.0] * 4, [20.0] * 4, [30.0] * 4])
    return xr.Dataset(
        {
            "U": (["y", "x"], da.from_array(np.ones((3, 4)), chunks=-1)),
            "V": (["y", "x"], da.from_array(np.full((3, 4), 2.0), chunks=-1)),
        },
        coords={
            "lon": (["y", "x"], da.from_array(lon, chunks=-1)),
            "lat": (["y", "x"], da.from_array(lat, chunks=-1)),
        },
    )


@pytest.mark.parametrize("ds", [pytest.param(v, id=k) for k, v in datasets.items()])
@pytest.mark.parametrize("except_for", [None, "coords"])
def test_replace_arrays_with_zeros(ds, except_for):
    utils.replace_arrays_with_zeros(ds, except_for=except_for)


def test_replace_arrays_with_zeros_none(nonzero_ds):
    """except_for=None: all data_vars and coords replaced with zeros."""
    result = utils.replace_arrays_with_zeros(nonzero_ds, except_for=None)

    for k in set(result.data_vars) | set(result.coords):
        assert np.all(result[k].values == 0), f"{k!r} should be zero"


def test_replace_arrays_with_zeros_coords(nonzero_ds):
    """except_for='coords': data_vars zeroed, coords preserved."""
    result = utils.replace_arrays_with_zeros(nonzero_ds, except_for="coords")

    for k in result.data_vars:
        assert np.all(result[k].values == 0), f"data_var {k!r} should be zero"

    np.testing.assert_array_equal(result["lon"].values, nonzero_ds["lon"].values)
    np.testing.assert_array_equal(result["lat"].values, nonzero_ds["lat"].values)


def test_replace_arrays_with_zeros_list(nonzero_ds):
    """except_for=[...]: listed variables preserved, others zeroed."""
    result = utils.replace_arrays_with_zeros(nonzero_ds, except_for=["U", "lon"])

    np.testing.assert_array_equal(result["U"].values, nonzero_ds["U"].values)
    np.testing.assert_array_equal(result["lon"].values, nonzero_ds["lon"].values)
    assert np.all(result["V"].values == 0), "V should be zero"
    assert np.all(result["lat"].values == 0), "lat should be zero"


def test_replace_arrays_with_zeros_does_not_mutate(nonzero_ds):
    """Original dataset is not modified."""
    original_U = nonzero_ds["U"].values.copy()
    original_lon = nonzero_ds["lon"].values.copy()
    utils.replace_arrays_with_zeros(nonzero_ds, except_for=None)
    np.testing.assert_array_equal(nonzero_ds["U"].values, original_U)
    np.testing.assert_array_equal(nonzero_ds["lon"].values, original_lon)


def test_replace_arrays_with_zeros_invalid_key(nonzero_ds):
    """Invalid key in except_for raises ValueError."""
    with pytest.raises(ValueError, match="not a valid item"):
        utils.replace_arrays_with_zeros(nonzero_ds, except_for=["nonexistent"])
