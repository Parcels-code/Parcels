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

Parcels provides issue templates for when posting issues.
Following these templates provides structure and ensures that we have all the necessary information we need to help you.

## "Parcels doesn't work with my input dataset"

Parcels is designed to work with a large range of input datasets.

When extending support for various input datasets, or trying to debug problems
that only occur with specific datasets, having the dataset metadata is very valuable.

This metadata could include information such as:

- CF compliant metadata
- descriptions about the origin of the dataset, or additional comments
- the shapes and data types of the arrays

This also allows us to see if your metadata is broken/non-compliant with standards - where we can then suggest fixes for you (and maybe we can tell the data provider!).
Since version 4 of Parcels we rely much more on metadata to discover information about your input data.

Sharing this metadata often provides enough debugging information to solve your problem, instead of having to share a whole dataset.

Sharing dataset metadata is made easy in Parcels.

### Step 1. Users

As a user with access to your dataset, you would do:

```{code-cell}
import json

import xarray as xr

# defining an example dataset to illustrate
# (you would use `xr.open_dataset(...)` instead)
ds = xr.Dataset(attrs={"description": "my dataset"})

output_file = "my_dataset.json"
with open(output_file, "w") as f:
    json.dump(ds.to_dict(data=False), f)  # write your dataset to a JSON excluding array data
```

Then attach the produced JSON file alongside your issue

### Step 2. Maintainers and developers

As developers looking to inspect the metadata, we would do:

```{code-cell}
from parcels._datasets.utils import from_xarray_dataset_dict

with open(output_file) as f:
    d = json.load(f)
ds = from_xarray_dataset_dict(d)
```

From there we can take a look the metadata of your dataset!
