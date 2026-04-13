import abc
import os
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pooch
import xarray as xr

from parcels._v3to4 import patch_dataset_v4_compat

__all__ = ["download_example_dataset", "list_example_datasets"]

# When modifying existing datasets in a backwards incompatible way,
# make a new release in the repo and update the DATA_REPO_TAG to the new tag
_DATA_REPO_TAG = "main"

_DATA_URL = f"https://github.com/Parcels-code/parcels-data/raw/{_DATA_REPO_TAG}/data"

_DATA_HOME = os.environ.get("PARCELS_EXAMPLE_DATA")
if _DATA_HOME is None:
    _DATA_HOME = pooch.os_cache("parcels")


# Keys are the dataset names. Values are the filenames in the dataset folder. Note that
# you can specify subfolders in the dataset folder putting slashes in the filename list.
# e.g.,
# "my_dataset": ["file0.nc", "folder1/file1.nc", "folder2/file2.nc"]
# my_dataset/
# ├── file0.nc
# ├── folder1/
# │   └── file1.nc
# └── folder2/
#     └── file2.nc
#
# See instructions at https://github.com/Parcels-code/parcels-data for adding new datasets
_EXAMPLE_DATA_FILES: dict[str, list[str]] = {
    "MovingEddies_data": [
        "moving_eddiesP.nc",
        "moving_eddiesU.nc",
        "moving_eddiesV.nc",
    ],
    "MITgcm_example_data": ["mitgcm_UV_surface_zonally_reentrant.nc"],
    "OFAM_example_data": ["OFAM_simple_U.nc", "OFAM_simple_V.nc"],
    "Peninsula_data": [
        "peninsulaU.nc",
        "peninsulaV.nc",
        "peninsulaP.nc",
        "peninsulaT.nc",
    ],
    "GlobCurrent_example_data": [
        f"{date.strftime('%Y%m%d')}000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"
        for date in ([datetime(2002, 1, 1) + timedelta(days=x) for x in range(0, 365)] + [datetime(2003, 1, 1)])
    ],
    "CopernicusMarine_data_for_Argo_tutorial": [
        "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_uo-vo_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
        "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_so_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
        "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_thetao_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
    ],
    "DecayingMovingEddy_data": [
        "decaying_moving_eddyU.nc",
        "decaying_moving_eddyV.nc",
    ],
    "FESOM_periodic_channel": [
        "fesom_channel.nc",
        "u.fesom_channel.nc",
        "v.fesom_channel.nc",
        "w.fesom_channel.nc",
    ],
    "NemoCurvilinear_data": [
        "U_purely_zonal-ORCA025_grid_U.nc4",
        "V_purely_zonal-ORCA025_grid_V.nc4",
        "mesh_mask.nc4",
    ],
    "NemoNorthSeaORCA025-N006_data": [
        "ORCA025-N06_20000104d05U.nc",
        "ORCA025-N06_20000109d05U.nc",
        "ORCA025-N06_20000114d05U.nc",
        "ORCA025-N06_20000119d05U.nc",
        "ORCA025-N06_20000124d05U.nc",
        "ORCA025-N06_20000129d05U.nc",
        "ORCA025-N06_20000104d05V.nc",
        "ORCA025-N06_20000109d05V.nc",
        "ORCA025-N06_20000114d05V.nc",
        "ORCA025-N06_20000119d05V.nc",
        "ORCA025-N06_20000124d05V.nc",
        "ORCA025-N06_20000129d05V.nc",
        "ORCA025-N06_20000104d05W.nc",
        "ORCA025-N06_20000109d05W.nc",
        "ORCA025-N06_20000114d05W.nc",
        "ORCA025-N06_20000119d05W.nc",
        "ORCA025-N06_20000124d05W.nc",
        "ORCA025-N06_20000129d05W.nc",
        "coordinates.nc",
    ],
    "POPSouthernOcean_data": [
        "t.x1_SAMOC_flux.169000.nc",
        "t.x1_SAMOC_flux.169001.nc",
        "t.x1_SAMOC_flux.169002.nc",
        "t.x1_SAMOC_flux.169003.nc",
        "t.x1_SAMOC_flux.169004.nc",
        "t.x1_SAMOC_flux.169005.nc",
    ],
    "SWASH_data": [
        "field_0065532.nc",
        "field_0065537.nc",
        "field_0065542.nc",
        "field_0065548.nc",
        "field_0065552.nc",
        "field_0065557.nc",
    ],
    "WOA_data": [f"woa18_decav_t{m:02d}_04.nc" for m in range(1, 13)],
    "CROCOidealized_data": ["CROCO_idealized.nc"],
}


def _create_pooch_registry() -> dict[str, None]:
    """Collapses the mapping of dataset names to filenames into a pooch registry.

    Hashes are set to None for all files.
    """
    registry: dict[str, None] = {}
    for dataset, filenames in _EXAMPLE_DATA_FILES.items():
        for filename in filenames:
            registry[f"{dataset}/{filename}"] = None
    return registry


_POOCH_REGISTRY = _create_pooch_registry()
_ODIE = pooch.create(
    path=_DATA_HOME,
    base_url=_DATA_URL,
    registry=_POOCH_REGISTRY,
)


class _ParcelsDataset(abc.ABC):
    @abc.abstractmethod
    def open_dataset(self) -> xr.Dataset: ...


class _V3Dataset(_ParcelsDataset):
    def __init__(self, path_relative_to_root: str, pre_decode_cf_callable=None):
        self.path_relative_to_root = path_relative_to_root  # glob is allowed

        # Function to apply to the dataset before the decoding the CF variables
        self.pup = _ODIE
        self.pre_decode_cf_callable: None | Callable[[xr.Dataset], xr.Dataset] = pre_decode_cf_callable
        self.v3_dataset_name = path_relative_to_root.split("/")[0]

    def open_dataset(self) -> xr.Dataset:
        self.download_relevant_files()
        with xr.set_options(use_new_combine_kwarg_defaults=True):
            ds = xr.open_mfdataset(Path(self.pup.path) / self.path_relative_to_root, decode_cf=False)

        if self.pre_decode_cf_callable is not None:
            ds = self.pre_decode_cf_callable(ds)

        ds = xr.decode_cf(ds)
        return ds

    def download_relevant_files(self) -> None:
        for file in self.pup.registry:
            if self.v3_dataset_name in file:
                self.pup.fetch(file)
        return


class _ZarrZipDataset(_ParcelsDataset):
    def __init__(self, path_relative_to_root):
        self.pup = _ODIE
        self.path_relative_to_root = path_relative_to_root

    def open_dataset(self) -> xr.Dataset:
        self.pup.fetch(self.path_relative_to_root)
        return xr.open_zarr(Path(self.pup.path) / self.path_relative_to_root)


def _preprocess_drop_time_from_mesh1(ds: xr.Dataset) -> xr.Dataset:
    # For some reason on the mesh "NemoNorthSeaORCA025-N006_data/coordinates.nc" there are time dimensions. These dimension also has broken cf-time metadata
    # this fixes that
    return ds.isel(time=0).drop(["time", "time_steps"])


def _preprocess_drop_time_from_mesh2(ds: xr.Dataset) -> xr.Dataset:
    # For some reason on the mesh "NemoCurvilinear_data_zonal/mesh_mask" there is a time dimension.
    return ds.isel(time=0).drop(["time"])


def _preprocess_set_cf_calendar_360_day(ds: xr.Dataset) -> xr.Dataset:
    # For some reason "WOA_data/woa18_decav_t*_04.nc" looks to be simulation data using CF time (i.e., months of 30 days), however the calendar attribute isn't set.
    ds.time.attrs.update({"calendar": "360_day"})
    return ds


# The first here is a human readable key, the latter the path to load the netcdf data
# (after refactor the latter open path will disappear, and will just be `open_zarr(f'{ds_key}.zip')`)
# fmt: off
_DATASET_KEYS_AND_CONFIGS: dict[str, _V3Dataset] = dict([
    ("MovingEddies_data/P", _V3Dataset("MovingEddies_data/moving_eddiesP.nc")),
    ("MovingEddies_data/U", _V3Dataset("MovingEddies_data/moving_eddiesU.nc")),
    ("MovingEddies_data/V", _V3Dataset("MovingEddies_data/moving_eddiesV.nc")),
    ("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant", _V3Dataset("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant.nc")),
    ("OFAM_example_data/U", _V3Dataset("OFAM_example_data/OFAM_simple_U.nc")),
    ("OFAM_example_data/V", _V3Dataset("OFAM_example_data/OFAM_simple_V.nc")),
    ("Peninsula_data/U", _V3Dataset("Peninsula_data/peninsulaU.nc")),
    ("Peninsula_data/V", _V3Dataset("Peninsula_data/peninsulaV.nc")),
    ("Peninsula_data/P", _V3Dataset("Peninsula_data/peninsulaP.nc")),
    ("Peninsula_data/T", _V3Dataset("Peninsula_data/peninsulaT.nc")),
    ("GlobCurrent_example_data/data", _V3Dataset("GlobCurrent_example_data/*000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc")),
    ("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-cur_anfc", _V3Dataset("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_uo-vo_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc")),
    ("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-so_anfc", _V3Dataset("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_so_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc")),
    ("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-thetao_anfc", _V3Dataset("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_thetao_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc")),
    ("DecayingMovingEddy_data/U", _V3Dataset("DecayingMovingEddy_data/decaying_moving_eddyU.nc")),
    ("DecayingMovingEddy_data/V", _V3Dataset("DecayingMovingEddy_data/decaying_moving_eddyV.nc")),
    ("FESOM_periodic_channel/fesom_channel", _V3Dataset("FESOM_periodic_channel/fesom_channel.nc")),
    ("FESOM_periodic_channel/u.fesom_channel", _V3Dataset("FESOM_periodic_channel/u.fesom_channel.nc")),
    ("FESOM_periodic_channel/v.fesom_channel", _V3Dataset("FESOM_periodic_channel/v.fesom_channel.nc")),
    ("FESOM_periodic_channel/w.fesom_channel", _V3Dataset("FESOM_periodic_channel/w.fesom_channel.nc")),
    ("NemoCurvilinear_data_zonal/U", _V3Dataset("NemoCurvilinear_data/U_purely_zonal-ORCA025_grid_U.nc4")),
    ("NemoCurvilinear_data_zonal/V", _V3Dataset("NemoCurvilinear_data/V_purely_zonal-ORCA025_grid_V.nc4")),
    ("NemoCurvilinear_data_zonal/mesh_mask", _V3Dataset("NemoCurvilinear_data/mesh_mask.nc4", _preprocess_drop_time_from_mesh2)),
    ("NemoNorthSeaORCA025-N006_data/U", _V3Dataset("NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05U.nc")),
    ("NemoNorthSeaORCA025-N006_data/V", _V3Dataset("NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05V.nc")),
    ("NemoNorthSeaORCA025-N006_data/W", _V3Dataset("NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05W.nc")),
    ("NemoNorthSeaORCA025-N006_data/mesh_mask", _V3Dataset("NemoNorthSeaORCA025-N006_data/coordinates.nc", _preprocess_drop_time_from_mesh1)),
    # "POPSouthernOcean_data/t.x1_SAMOC_flux.16900*.nc", # TODO v4: In v3 but should be in v4 https://github.com/Parcels-code/Parcels/issues/2571#issuecomment-4214476973
    ("SWASH_data/data", _V3Dataset("SWASH_data/field_00655*.nc")),
    ("WOA_data/data", _V3Dataset("WOA_data/woa18_decav_t*_04.nc", _preprocess_set_cf_calendar_360_day)),
    ("CROCOidealized_data/data", _V3Dataset("CROCOidealized_data/CROCO_idealized.nc")),
])
# fmt: on


def list_example_datasets(v4=False) -> list[str]:  # TODO: Remove v4 flag when migrating to open_dataset
    """List the available example datasets.

    Use :func:`download_example_dataset` to download one of the datasets.

    Returns
    -------
    datasets : list of str
        The names of the available example datasets.
    """
    if v4:
        return list(_DATASET_KEYS_AND_CONFIGS.keys())
    return list(set(v.path_relative_to_root.split("/")[0] for v in _DATASET_KEYS_AND_CONFIGS.values()))


def download_example_dataset(dataset: str):
    """Load an example dataset from the parcels website.

    This function provides quick access to a small number of example datasets
    that are useful in documentation and testing in parcels.

    The location where the data is downloaded can be set using the environment variable PARCELS_EXAMPLE_DATA .

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.

    Returns
    -------
    dataset_folder : Path
        Path to the folder containing the downloaded dataset files.
    """
    # Dev note: `dataset` is assumed to be a folder name with netcdf files
    if dataset not in _EXAMPLE_DATA_FILES:
        raise ValueError(
            f"Dataset {dataset!r} not found. Available datasets are: " + ", ".join(_EXAMPLE_DATA_FILES.keys())
        )

    cache_folder = Path(_ODIE.path)
    dataset_folder = cache_folder / dataset

    for file_name in _ODIE.registry:
        if file_name.startswith(dataset):
            should_patch = dataset == "GlobCurrent_example_data"
            _ODIE.fetch(file_name, processor=_v4_compat_patch if should_patch else None)

    return dataset_folder


def open_dataset(name: str):
    try:
        dataset_config = _DATASET_KEYS_AND_CONFIGS[name]
    except KeyError as e:
        raise ValueError(
            f"Dataset {name!r} not found. Available datasets are: " + ", ".join(list_example_datasets(v4=True))
        ) from e
    assert not name.endswith((".zarr", ".zip", ".nc")), (
        "Dataset name should not have suffix"
    )  # TODO: Move to test_tutorial

    return dataset_config.open_dataset()


def _v4_compat_patch(fname, action, pup):
    """
    Patch the GlobCurrent example dataset to be compatible with v4.

    See https://www.fatiando.org/pooch/latest/processors.html#creating-your-own-processors
    """
    if action == "fetch":
        return fname
    xr.load_dataset(fname).pipe(patch_dataset_v4_compat).to_netcdf(fname)
    return fname
