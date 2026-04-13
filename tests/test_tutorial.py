import pytest
import requests
import xarray as xr

from parcels.tutorial import (
    _ODIE,
    download_example_dataset,
    list_example_datasets,
    open_dataset,
)


@pytest.fixture(scope="function")
def tmp_path_parcels_example_data(monkeypatch, tmp_path):
    monkeypatch.setenv("PARCELS_EXAMPLE_DATA", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("url", [_ODIE.get_url(filename) for filename in _ODIE.registry.keys()])
def test_pooch_registry_url_reponse(url):
    response = requests.head(url)
    assert not (400 <= response.status_code < 600)


@pytest.mark.parametrize("dataset", list_example_datasets()[:1])
def test_download_example_dataset_folder_creation(dataset):
    dataset_folder_path = download_example_dataset(dataset)

    assert dataset_folder_path.exists()
    assert dataset_folder_path.name == dataset


def test_download_non_existing_example_dataset(tmp_path_parcels_example_data):
    with pytest.raises(ValueError):
        download_example_dataset("non_existing_dataset")


def test_download_example_dataset_no_data_home():
    # This test depends on your default data_home location and whether
    # it's okay to download files there. Be careful with this test in a CI environment.
    dataset = list_example_datasets()[0]
    dataset_folder_path = download_example_dataset(dataset)
    assert dataset_folder_path.exists()
    assert dataset_folder_path.name == dataset


@pytest.mark.parametrize("name", list_example_datasets(v4=True))
def test_open_dataset(name):
    ds = open_dataset(name)
    assert isinstance(ds, xr.Dataset)
