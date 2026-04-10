import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pooch
import xarray as xr
import zarr

from parcels._v3to4 import patch_dataset_v4_compat

__all__ = ["download_example_dataset", "list_example_datasets"]

# When modifying existing datasets in a backwards incompatible way,
# make a new release in the repo and update the DATA_REPO_TAG to the new tag
_DATA_REPO_TAG = "main"

_DATA_URL = f"https://github.com/Parcels-code/parcels-data/raw/{_DATA_REPO_TAG}/data"

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


@dataclass
class DatasetNCtoZarrConfig:
    path_relative_to_root: str

    # Function to apply to the dataset before the decoding the CF variables
    pre_decode_cf_callable: None | Callable[[xr.Dataset], xr.Dataset] = None


# The first here is a human readable key, the latter the path to load the netcdf data
# (after refactor the latter open path will disappear, and will just be `open_zarr(f'{ds_key}.zip')`)
# fmt: off
_DATASET_KEYS_AND_CONFIGS: dict[str, DatasetNCtoZarrConfig] = dict([
    ("MovingEddies_data/P", DatasetNCtoZarrConfig("MovingEddies_data/moving_eddiesP.nc")),
    ("MovingEddies_data/U", DatasetNCtoZarrConfig("MovingEddies_data/moving_eddiesU.nc")),
    ("MovingEddies_data/V", DatasetNCtoZarrConfig("MovingEddies_data/moving_eddiesV.nc")),
    ("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant", DatasetNCtoZarrConfig("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant.nc")),
    ("OFAM_example_data/U", DatasetNCtoZarrConfig("OFAM_example_data/OFAM_simple_U.nc")),
    ("OFAM_example_data/V", DatasetNCtoZarrConfig("OFAM_example_data/OFAM_simple_V.nc")),
    ("Peninsula_data/U", DatasetNCtoZarrConfig("Peninsula_data/peninsulaU.nc")),
    ("Peninsula_data/V", DatasetNCtoZarrConfig("Peninsula_data/peninsulaV.nc")),
    ("Peninsula_data/P", DatasetNCtoZarrConfig("Peninsula_data/peninsulaP.nc")),
    ("Peninsula_data/T", DatasetNCtoZarrConfig("Peninsula_data/peninsulaT.nc")),
    ("GlobCurrent_example_data/data.nc", DatasetNCtoZarrConfig("GlobCurrent_example_data/*000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc")),
    ("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-cur_anfc", DatasetNCtoZarrConfig("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_uo-vo_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc")),
    ("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-so_anfc", DatasetNCtoZarrConfig("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m_so_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc")),
    ("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-thetao_anfc", DatasetNCtoZarrConfig("CopernicusMarine_data_for_Argo_tutorial/cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_thetao_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc")),
    ("DecayingMovingEddy_data/U", DatasetNCtoZarrConfig("DecayingMovingEddy_data/decaying_moving_eddyU.nc")),
    ("DecayingMovingEddy_data/V", DatasetNCtoZarrConfig("DecayingMovingEddy_data/decaying_moving_eddyV.nc")),
    ("FESOM_periodic_channel/fesom_channel", DatasetNCtoZarrConfig("FESOM_periodic_channel/fesom_channel.nc")),
    ("FESOM_periodic_channel/u.fesom_channel", DatasetNCtoZarrConfig("FESOM_periodic_channel/u.fesom_channel.nc")),
    ("FESOM_periodic_channel/v.fesom_channel", DatasetNCtoZarrConfig("FESOM_periodic_channel/v.fesom_channel.nc")),
    ("FESOM_periodic_channel/w.fesom_channel", DatasetNCtoZarrConfig("FESOM_periodic_channel/w.fesom_channel.nc")),
    ("NemoCurvilinear_data_zonal/U", DatasetNCtoZarrConfig("NemoCurvilinear_data/U_purely_zonal-ORCA025_grid_U.nc4")),
    ("NemoCurvilinear_data_zonal/V", DatasetNCtoZarrConfig("NemoCurvilinear_data/V_purely_zonal-ORCA025_grid_V.nc4")),
    ("NemoCurvilinear_data_zonal/mesh_mask", DatasetNCtoZarrConfig("NemoCurvilinear_data/mesh_mask.nc4")),
    ("NemoNorthSeaORCA025-N006_data/U", DatasetNCtoZarrConfig("NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05U.nc")),
    ("NemoNorthSeaORCA025-N006_data/V", DatasetNCtoZarrConfig("NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05V.nc")),
    ("NemoNorthSeaORCA025-N006_data/W", DatasetNCtoZarrConfig("NemoNorthSeaORCA025-N006_data/ORCA025-N06_200001*05W.nc")),
    ("NemoNorthSeaORCA025-N006_data/mesh_mask", DatasetNCtoZarrConfig("NemoNorthSeaORCA025-N006_data/coordinates.nc")),
    # "POPSouthernOcean_data/t.x1_SAMOC_flux.16900*.nc", # TODO v4: In v3 but should be in v4 https://github.com/Parcels-code/Parcels/issues/2571#issuecomment-4214476973
    ("SWASH_data/data", DatasetNCtoZarrConfig("SWASH_data/field_00655*.nc")),
    ("WOA_data/data", DatasetNCtoZarrConfig("WOA_data/woa18_decav_t*_04.nc")),
    ("CROCOidealized_data/data", DatasetNCtoZarrConfig("CROCOidealized_data/CROCO_idealized.nc")),
])
# fmt: on


def _create_pooch_registry() -> dict[str, None]:
    """Collapses the mapping of dataset names to filenames into a pooch registry.

    Hashes are set to None for all files.
    """
    registry: dict[str, None] = {}
    for dataset, filenames in _EXAMPLE_DATA_FILES.items():
        for filename in filenames:
            registry[f"{dataset}/{filename}"] = None
    return registry


POOCH_REGISTRY = _create_pooch_registry()


def _get_pooch(data_home=None):
    if data_home is None:
        data_home = os.environ.get("PARCELS_EXAMPLE_DATA")
    if data_home is None:
        data_home = pooch.os_cache("parcels")

    return pooch.create(
        path=data_home,
        base_url=_DATA_URL,
        registry=POOCH_REGISTRY,
    )


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


def download_example_dataset(dataset: str, data_home=None):
    """Load an example dataset from the parcels website.

    This function provides quick access to a small number of example datasets
    that are useful in documentation and testing in parcels.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    data_home : pathlike, optional
        The directory in which to cache data. If not specified, the value
        of the ``PARCELS_EXAMPLE_DATA`` environment variable, if any, is used.
        Otherwise the default location is assigned by :func:`get_data_home`.

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
    odie = _get_pooch(data_home=data_home)

    cache_folder = Path(odie.path)
    dataset_folder = cache_folder / dataset

    for file_name in odie.registry:
        if file_name.startswith(dataset):
            should_patch = dataset == "GlobCurrent_example_data"
            odie.fetch(file_name, processor=_v4_compat_patch if should_patch else None)

    return dataset_folder


# Just creating a temp folder to help during the migration
_TMP_ZARR_FOLDER = Path("../parcels-data/data-zarr")


def open_dataset(name: str, code_path: Literal["nc", "zarr"] = "nc"):  # TODO: Remove code_path arg
    try:
        cfg = _DATASET_KEYS_AND_CONFIGS[name]
    except KeyError as e:
        raise ValueError(
            f"Dataset {name!r} not found. Available datasets are: " + ", ".join(list_example_datasets(v4=True))
        ) from e

    open_dataset_kwargs = dict(decode_timedelta=False, decode_cf=False)
    open_dataset_kwargs = dict(decode_cf=False)
    # assert not dataset.endswith((".zarr", ".zip", ".nc")), "Dataset name should not have suffix"
    download_dataset_stem, rest = cfg.path_relative_to_root.split("/", maxsplit=1)
    folder = download_example_dataset(download_dataset_stem)

    with xr.set_options(use_new_combine_kwarg_defaults=True):
        # return f"{folder}/{rest}"
        ds = xr.open_mfdataset(f"{folder}/{rest}", **open_dataset_kwargs)

    if cfg.pre_decode_cf_callable is not None:
        ds = cfg.pre_decode_cf_callable(ds)

    ds = xr.decode_cf(ds)

    if code_path == "nc":
        return ds
    path = _TMP_ZARR_FOLDER / f"{name}.zip"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with zarr.storage.ZipStore(path, mode="w") as store:
            ds.to_zarr(store)
    return xr.open_zarr(path, **open_dataset_kwargs)


def _v4_compat_patch(fname, action, pup):
    """
    Patch the GlobCurrent example dataset to be compatible with v4.

    See https://www.fatiando.org/pooch/latest/processors.html#creating-your-own-processors
    """
    if action == "fetch":
        return fname
    xr.load_dataset(fname).pipe(patch_dataset_v4_compat).to_netcdf(fname)
    return fname
