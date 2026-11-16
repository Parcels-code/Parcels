from datetime import timedelta

import numpy as np
import pytest

from parcels import Field, FieldSet
from parcels._core.model import StructuredModelData
from parcels._datasets.structured.generic import T as T_structured
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._python import NOTSET


def test_model_init_fail_on_float_time_dim():
    """Test that accessing time_interval fails when dataset has float time dimension.

    (users are expected to use timedelta64 or datetime).
    Time validation has moved from Field.__init__ to ModelData.time_interval.
    """
    ds = datasets_structured["ds_2d_left"].copy()
    ds["time"] = (
        ds["time"].dims,
        np.arange(0, T_structured, dtype="float64"),
        ds["time"].attrs,
    )

    model = StructuredModelData.from_sgrid_conventions(ds, vector_fields=NOTSET)
    with pytest.raises(
        ValueError,
        match=r"Are you sure that the time dimension on the xarray dataset is stored as timedelta, datetime or cftime datetime objects\?",
    ):
        _ = model.time_interval


def test_model_time_interval():
    """Test that field.time_interval delegates correctly to model.time_interval."""
    data = datasets_structured["ds_2d_left"]
    model = StructuredModelData.from_sgrid_conventions(data, vector_fields=NOTSET)
    field = Field(name="data_g", model=model)
    assert field.time_interval.left == np.datetime64("2000-01-01")
    assert field.time_interval.right == np.datetime64("2001-01-01")


def test_multi_model_time_interval():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]]
    fieldset = FieldSet.from_sgrid_conventions(ds1)

    ds2 = ds1.copy().rename({"U_A_grid": "U2", "V_A_grid": "V2"})
    ds2["time"] = (ds2["time"].dims, ds2["time"].data + np.timedelta64(timedelta(days=1)), ds2["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds2)

    ds3 = ds1.copy().rename({"U_A_grid": "U3", "V_A_grid": "V3"})
    ds3["time"] = (ds3["time"].dims, ds3["time"].data + np.timedelta64(timedelta(days=2)), ds3["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds3)

    fieldset.add_constant_field("constant_field", 1.0)

    assert len(fieldset.models) == 3
    assert fieldset.constant_model is not None
    assert fieldset.time_interval.left == np.datetime64("2000-01-03")
    assert fieldset.time_interval.right == np.datetime64("2001-01-01")


def test_multi_model_nonoverlapping_time_interval():
    ds1 = datasets_structured["ds_2d_left"][["U_A_grid", "V_A_grid", "grid"]]
    fieldset = FieldSet.from_sgrid_conventions(ds1)

    ds2 = ds1.copy().rename({"U_A_grid": "U2", "V_A_grid": "V2"})
    ds2["time"] = (ds2["time"].dims, ds2["time"].data + np.timedelta64(timedelta(days=1000)), ds2["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds2)

    ds3 = ds1.copy().rename({"U_A_grid": "U3", "V_A_grid": "V3"})
    ds3["time"] = (ds3["time"].dims, ds3["time"].data + np.timedelta64(timedelta(days=2000)), ds3["time"].attrs)
    fieldset += FieldSet.from_sgrid_conventions(ds3)

    fieldset.add_constant_field("constant_field", 1.0)

    assert len(fieldset.models) == 3
    assert fieldset.constant_model is not None
    assert fieldset.time_interval is None
