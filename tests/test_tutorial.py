import pytest
import requests
import xarray as xr

import parcels.tutorial


@pytest.fixture(scope="function", autouse=True)
def tmp_path_parcels_example_data(monkeypatch, tmp_path):
    monkeypatch.setenv("PARCELS_EXAMPLE_DATA", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "url", [parcels.tutorial._ODIE.get_url(filename) for filename in parcels.tutorial._ODIE.registry.keys()]
)
def test_pooch_registry_url_reponse(url):
    response = requests.head(url)
    assert not (400 <= response.status_code < 600)


def test_open_dataset_non_existing():
    with pytest.raises(ValueError, match="Dataset.*not found"):
        parcels.tutorial.open_dataset("non_existing_dataset")


@pytest.mark.parametrize("name", parcels.tutorial.list_datasets())
def test_open_dataset(name):
    ds = parcels.tutorial.open_dataset(name)
    assert isinstance(ds, xr.Dataset)


@pytest.mark.parametrize("name", parcels.tutorial.list_datasets())
def test_dataset_keys(name):
    assert not name.endswith((".zarr", ".zip", ".nc")), "Dataset name should not have suffix"
