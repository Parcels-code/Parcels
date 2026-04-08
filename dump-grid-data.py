import datetime
import os
from pathlib import Path

import intake
import zarr

from parcels._datasets import utils

cat = intake.open_catalog("../parcels-benchmarks/data/surf-data/parcels-benchmarks/catalog.yml")


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


datasets = {
    # "example-dataset-from-parcels": datasets_sgrid["ds_2d_padded_low"],
    # "fesom_mesh": cat.fesom_baroclinic_gyre_mesh.to_dask(),
    "moi_mesh": cat.moi_mesh.to_dask().set_coords(["glamf", "glamu"]),
}

zarrs_folder = Path("zarrs")
zarrs_folder.mkdir(exist_ok=True)
for k, ds in datasets.items():
    path = zarrs_folder / f"{datetime.datetime.now().isoformat()}-{k}.zip"
    ds = ds.pipe(utils.strip_datavars)
    nbytes_uncompressed_full_dataset = ds.nbytes
    nbytes_uncompressed_coords = 0

    for c in ds.coords:
        nbytes_uncompressed_coords += ds.coords[c].nbytes

    ds.to_zarr(zarr.storage.ZipStore(path))

    nbytes_compressed = os.path.getsize(path)

    print(r"Summary for dataset {k!r}")
    print("=========================")
    print(f"Original dataset uncompressed size: {sizeof_fmt(nbytes_uncompressed_full_dataset):>8}")
    print(f"Coords dataset uncompressed size: {sizeof_fmt(nbytes_uncompressed_coords):>8}")
    print(f"Compressed Zarr with coordinates: {sizeof_fmt(nbytes_compressed):>8}")
    print("---")
    print("Compressed dataset is:")
    print(f"  -{nbytes_compressed / nbytes_uncompressed_full_dataset:.1%} of original")
    print(f"  -{nbytes_compressed / nbytes_uncompressed_coords:.1%} of coordinate only")

# print()
# print("Compressed JSON representation of datasets")
# print("==========================================")
# for k, ds in datasets.items():
#     path = Path(f"jsons/{datetime.datetime.now().isoformat()}-{k}.json.gz")
#     utils.dataset_to_json(ds, path, compressed=True)
#     print(f"Dataset {k} JSON representation is size: {sizeof_fmt(os.path.getsize(path)):>8}")
