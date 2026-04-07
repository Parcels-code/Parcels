from parcels._datasets.structured.generic import datasets
from parcels._datasets.utils import from_xarray_dataset_dict


def test_from_xarray_dataset_dict():
    ds_expected = datasets["ds_2d_left"]
    d = ds_expected.to_dict(data=False)
    ds = from_xarray_dataset_dict(d)

    assert list(ds.coords) == list(ds_expected.coords)
    assert list(ds.data_vars) == list(ds_expected.data_vars)

    for k in set(ds.coords) | set(ds.data_vars):
        assert ds[k].attrs == ds_expected[k].attrs, f"Attrs for {k!r} do not match"
