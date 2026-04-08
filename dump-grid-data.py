import datetime
import os
from pathlib import Path

import intake
import zarr

from parcels._datasets import utils
from parcels._datasets.structured.generic import datasets_sgrid

cat = intake.open_catalog("../parcels-benchmarks/data/surf-data/parcels-benchmarks/catalog.yml")


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


datasets = {
    "example-dataset-from-parcels": (datasets_sgrid["ds_2d_padded_low"], []),
    "fesom_mesh": (cat.fesom_baroclinic_gyre_mesh.to_dask(), []),
    "moi_mesh": (cat.moi_mesh.to_dask(), ["glamf", "glamu"]),
}

zarrs_folder = Path("zarrs")
zarrs_folder.mkdir(exist_ok=True)
for k, (ds, except_for) in datasets.items():
    path = zarrs_folder / f"{datetime.datetime.now().isoformat()}-{k}.zip"
    ds = ds.pipe(utils.replace_arrays_with_zeros, except_for=except_for)
    nbytes_uncompressed_full_dataset = ds.nbytes
    nbytes_uncompressed_trimmed = 0

    for c in ds.coords:
        nbytes_uncompressed_trimmed += ds[c].nbytes
    for d in ds.data_vars:
        if d in except_for:
            nbytes_uncompressed_trimmed += ds[d].nbytes

    ds.to_zarr(zarr.storage.ZipStore(path))

    nbytes_compressed = os.path.getsize(path)

    print(f"Summary for dataset {k!r}")
    print("=========================")
    print(f"Original dataset uncompressed size: {sizeof_fmt(nbytes_uncompressed_full_dataset):>8}")
    print(f"Trimmed dataset uncompressed size: {sizeof_fmt(nbytes_uncompressed_trimmed):>8}")
    print(f"Compressed Zarr with coordinates: {sizeof_fmt(nbytes_compressed):>8}")
    print("---")
    print("Timmed compressed dataset is:")
    print(f"  -{nbytes_compressed / nbytes_uncompressed_full_dataset:.1%} of original")
    print(f"  -{nbytes_compressed / nbytes_uncompressed_trimmed:.1%} of trimmed uncompressed")
