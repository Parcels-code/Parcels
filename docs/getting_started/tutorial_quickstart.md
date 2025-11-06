---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart tutorial

```{note}
TODO: Completely rewrite examples/parcels_tutorial.ipynb to be this quickstart tutorial. Decide which file format and notebook testing to do so this file is checked, which is not in the "examples" folder
```

Welcome to the **Parcels** quickstart tutorial, in which we will go through all the necessary steps to run a simulation. The code in this notebook can be used as a starting point to run Parcels in your own environment. Along the way we will familiarize ourselves with some specific classes and methods. If you are ever confused about one of these, or how they relate to each other, we have a [concepts overview](concepts_overview.md) to describe how we think about them. Let's dive in!

## Import

```{code-cell}
import numpy as np
import xarray as xr
import parcels
```

## Input: `FieldSet`

Load the CopernicusMarine data in the Agulhas region from the example_datasets

```{code-cell}
example_dataset_folder = parcels.download_example_dataset(
    "CopernicusMarine_data_for_Argo_tutorial"
)

ds = xr.open_mfdataset(f"{example_dataset_folder}/*.nc", combine="by_coords")
ds.load()  # load the dataset into memory

fieldset = parcels.FieldSet.from_copernicusmarine(ds)
```

## Input: `ParticleSet`

```{code-cell}
# Particle locations and initial time
npart = 10  # number of particles to be released
lon = np.repeat(32, npart)
lat = np.linspace(-32.5, -30.5, npart)
time = np.repeat(ds.time.values[0], npart)
z = np.repeat(ds.depth.values[0], npart)

pset = parcels.ParticleSet(
    fieldset=fieldset, pclass=parcels.Particle, lon=lon, lat=lat, time=time, z=z
)
```

## Compute: `Kernel`

```{code-cell}
kernels = parcels.kernels.AdvectionEE
```

## Prepare output: `ParticleFile`

```{code-cell}
output_file = parcels.ParticleFile("Output.zarr", outputdt=np.timedelta64(1, "h"))
```

## Run Simulation: `ParticleSet.execute()`

```{code-cell}
pset.execute(
    kernels,
    runtime=np.timedelta64(5, "h"),
    dt=np.timedelta64(5, "m"),
    output_file=output_file,
)
```

## Read output

```{code-cell}
data_xarray = xr.open_zarr("Output.zarr")
data_xarray
```
