---
file_format: mystnb
kernelspec:
  name: python3
---

# Participating in the issue tracker

We love hearing from our community!
We want to be able to support you in your workflows, and learn about how you use Parcels.
In open source projects, getting feedback from users is hard - you posting
issues and participating in the issue tracker is really useful for us and
helps future development and squash bugs.

Parcels provides issue templates that you can use when posting issues.
Following these templates provides structure and ensures that we have all the necessary information we need to help you.

## "Parcels doesn't work with my input dataset"

Parcels is designed to work with a large range of input datasets.

When extending support for various input datasets, or trying to debug problems
that only occur with specific datasets, having access to your dataset (or a
close representation of it) is very valuable.

This could include information such as:

- the nature of the array variables (e.g., via CF compliant metadata)
- descriptions about the origin of the dataset, or additional comments
- the shapes and data types of the arrays
- the grid topology (coordinates and key variables)

This also allows us to see if your metadata is broken/non-compliant with standards - where we can then suggest fixes for you (and maybe we can tell the data provider!).
Since version 4 of Parcels we rely much more on metadata to discover information about your input data.

Sharing a compact representation of your dataset often provides enough information to solve your problem, without having to share the full dataset (which may be very large or contain sensitive data).

Parcels makes this easy by replacing irrelevant array data with zeros and saving the result as a compressed Zarr zip store, which is typically small enough to attach directly to a GitHub issue.

### Step 1. Users

As a user with access to your dataset, you would do:

```{code-cell}
:tags: [hide-cell]

import xarray as xr
from parcels._datasets.structured.generic import datasets
datasets['ds_2d_left'].to_netcdf("my_dataset.nc")
```

```{code-cell}
import os

import xarray as xr
import zarr

from parcels._datasets.utils import replace_arrays_with_zeros

# load your dataset
ds = xr.open_dataset("my_dataset.nc")  # or xr.open_zarr(...), etc.

# Replace all data arrays with zeros, keeping coordinate metadata.
# This keeps array shapes and metadata while removing actual data.
#
# You can customise `except_for` to also retain actual values for specific variables:
#   except_for='coords'         — keep coordinate arrays (useful for grid topology)
#   except_for=['lon', 'lat']   — keep a specific list of variables
ds_trimmed = replace_arrays_with_zeros(ds)  # default: except_for=None

# Save to a zipped Zarr store - replace `my_dataset` with a more informative name
with zarr.storage.ZipStore("my_dataset.zip", mode='w') as store:
    ds_trimmed.to_zarr(store)

# Check the file size (aim for < 25 MB so it can be attached to a GitHub issue)
size_mb = os.path.getsize("my_dataset.zip") / 1e6
print(f"Zip store size: {size_mb:.1f} MB")
```

Then attach the zip file written above alongside your issue.

If the file is larger than 25 MB, try passing `except_for=None` (the default)
to ensure all arrays are zeroed out. If it is still too large, consider
subsetting your dataset to a smaller spatial or temporal region before saving.

### Step 2. Maintainers and developers

As developers looking to inspect the dataset, we would do:

```{code-cell}
import xarray as xr
import zarr

ds = xr.open_zarr(zarr.storage.ZipStore("my_dataset.zip", mode="r"))
ds
```

```{code-cell}
:tags: [hide-cell]

del ds
from pathlib import Path
Path("my_dataset.zip").unlink()
Path("my_dataset.nc").unlink()

```

From there we can take a look at the structure and metadata of your dataset!
This also makes it straightforward for us to add this dataset to our test suite.
