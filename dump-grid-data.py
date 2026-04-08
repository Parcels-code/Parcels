import datetime
import os
from pathlib import Path

import intake

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
    "example-dataset-from-parcels": datasets_sgrid["ds_2d_padded_low"],
    "fesom_mesh": cat.fesom_baroclinic_gyre_mesh.to_dask(),
    "moi_mesh": cat.moi_mesh.to_dask().set_coords(["glamf", "glamu"]),
}

jsons_folder = Path("jsons")
jsons_folder.mkdir(exist_ok=True)
print("Uncompressed JSON representation of datasets")
print("============================================")
for k, ds in datasets.items():
    path = Path(f"jsons/{datetime.datetime.now().isoformat()}-{k}.json")
    utils.dataset_to_json(ds, path)
    print(f"Dataset {k} JSON representation is size: {sizeof_fmt(os.path.getsize(path)):>8}")
print()
print("Compressed JSON representation of datasets")
print("==========================================")
for k, ds in datasets.items():
    path = Path(f"jsons/{datetime.datetime.now().isoformat()}-{k}.json.gz")
    utils.dataset_to_json(ds, path, compressed=True)
    print(f"Dataset {k} JSON representation is size: {sizeof_fmt(os.path.getsize(path)):>8}")
