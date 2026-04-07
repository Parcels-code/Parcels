import json
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from parcels._datasets import utils
from parcels._datasets.structured.generic import datasets


def test_datetime_encoder_decoder_roundtrip():
    dt = datetime(2000, 1, 15, 12, 30, 45)
    data = {"timestamps": [dt, dt], "nested": {"time": dt}, "value": 42}
    encoded = json.dumps(data, cls=utils._DatetimeEncoder)
    decoded = json.loads(encoded, cls=utils._DatetimeDecoder)
    assert decoded == data


def test_from_xarray_dataset_dict():
    ds_expected = datasets["ds_2d_left"]
    d = ds_expected.to_dict(data=False)
    ds = utils.from_xarray_dataset_dict(d)

    assert list(ds.coords) == list(ds_expected.coords)
    assert list(ds.data_vars) == list(ds_expected.data_vars)

    for k in set(ds.coords) | set(ds.data_vars):
        assert ds[k].attrs == ds_expected[k].attrs, f"Attrs for {k!r} do not match"


def _replace_with_cf_time(ds) -> xr.Dataset:
    import cftime

    assert "time" in ds, "Dataset must have a dimension named 'time'"
    ntime = 12
    ntime = min(ntime, len(ds.time.values))
    ds = ds.isel(time=slice(None, ntime))

    dates = [cftime.DatetimeNoLeap(1, month, 1) for month in range(1, ntime + 1)]
    ds["time"] = dates
    return ds


@pytest.mark.parametrize("ds", [pytest.param(v, id=k) for k, v in datasets.items()])
def test_dataset_json_roundtrip(ds: xr.Dataset, tmp_path):
    path = tmp_path / "dataset-metadata.json"
    utils.dataset_to_json(ds, path)
    ds_parsed = utils.dataset_from_json(path)

    assert list(ds_parsed.coords) == list(ds.coords)
    assert list(ds_parsed.data_vars) == list(ds.data_vars)

    for k in set(ds.data_vars):
        assert ds_parsed[k].attrs == ds[k].attrs, f"Attrs for {k!r} do not match"

    for k in set(ds.coords):
        assert ds_parsed[k].attrs == ds[k].attrs, f"Attrs for {k!r} do not match"
        if isinstance(ds_parsed[k].dtype, np.dtypes.DateTime64DType):
            np.testing.assert_equal(ds_parsed[k].values, ds[k].values)
        else:
            np.testing.assert_allclose(ds_parsed[k].values, ds[k].values)


@pytest.mark.parametrize("ds", [pytest.param(_replace_with_cf_time(datasets["ds_2d_left"]), id="cftime-ds_2d_left")])
def test_dataset_json_errors_with_cftime(ds: xr.Dataset, tmp_path):
    path = tmp_path / "dataset-metadata.json"
    with pytest.raises(TypeError, match="Object of type Datetime.* is not JSON serializable"):
        utils.dataset_to_json(ds, path)
