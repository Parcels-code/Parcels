---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart tutorial

```{note}
TODO: Completely rewrite examples/parcels_tutorial.ipynb to be this quickstart tutorial. Decide which file format and notebook testing to do so this file is checked, which is not in the "examples" folder
```

Welcome to the **Parcels** quickstart tutorial, in which we will go through all the necessary steps to run a simulation. The code in this notebook can be used as a starting point to run Parcels in your own environment. Along the way we will familiarize ourselves with some specific classes and methods. If you are ever confused about one of these and want to read more, we have a [concepts overview](concepts_overview.md) discussing them in more detail. Let's dive in!

## Imports
Parcels depends on `xarray`, expecting inputs in the form of [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) and writing output files that can be read with xarray.

```{code-cell}
import numpy as np
import xarray as xr
import parcels
```

## Input 1: `FieldSet`
A Parcels simulation of Lagrangian trajectories of virtual particles requires two inputs; the first is a set of hydrodynamics fields in which the particles are tracked. This set of vectorfields, with `U`, `V` (and `W`) flow velocities, can be read in to a `parcels.FieldSet` object from many types of models or observations. Here we provide an example using a subset of the [Global Ocean Physics Reanalysis](https://doi.org/10.48670/moi-00021) from the Copernicus Marine Service.

```{code-cell}
example_dataset_folder = parcels.download_example_dataset(
    "CopernicusMarine"
)

ds_in = xr.open_mfdataset(f"{example_dataset_folder}/*.nc", combine="by_coords")
ds_in.load()  # load the dataset into memory

fieldset = parcels.FieldSet.from_copernicusmarine(ds_in)
```

The reanalysis contains `U`, `V`, potential temperature (`thetao`) and salinity (`so`):

```{code-cell}
ds_in
```
The subset contains a region of the Agulhas current along the southeastern coast of Africa:
```{code-cell}
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
ds_in
```

## Input 2: `ParticleSet`
Now that we have read in the hydrodynamic data, we need to provide our second input: the virtual particles for which we will calculate the trajectories. We need to define the initial time and position and read those into a `parcels.ParticleSet` object, which also needs to know about the `FieldSet` in which the particles "live". Finally, we need to specify the type of `parcels.Particle` we want to use. The default particles have `time`, `lon`, `lat`, and `z` to keep track of, but with Parcels you can easily build your own particles to mimic plastic or an [ARGO float](../user_guide/tutorial_Argofloats.ipynb), adding variables such as size, temperature, or age.

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
After setting up the input data, we need to specify how to calculate the advection of the particles. These calculations, or numerical integrations, will be performed by what we call a `Kernel`, operating on each particle in the `ParticleSet`. The most common calculation is the advection of particles through the velocity field. Parcels comes with a number of standard kernels, from which we will use the Euler-forward advection kernel:

```{note}
TODO: link to list of included kernels
```
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
