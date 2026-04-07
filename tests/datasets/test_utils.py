import pytest
import xarray as xr

from parcels._datasets import utils
from parcels._datasets.structured.generic import datasets


def test_from_xarray_dataset_dict():
    ds_expected = datasets["ds_2d_left"]
    d = ds_expected.to_dict(data=False)
    ds = utils.from_xarray_dataset_dict(d)

    assert list(ds.coords) == list(ds_expected.coords)
    assert list(ds.data_vars) == list(ds_expected.data_vars)

    for k in set(ds.coords) | set(ds.data_vars):
        assert ds[k].attrs == ds_expected[k].attrs, f"Attrs for {k!r} do not match"


@pytest.mark.parametrize("ds", [pytest.param(v, id=k) for k, v in datasets.items()])
def test_dataset_json_roundtrip(ds: xr.Dataset, tmp_path):
    path = tmp_path / "dataset-metadata.json"
    utils.dataset_to_json(ds, path)
    ds_parsed = utils.dataset_from_json(path)  # noqa: F841
    # breakpoint()
