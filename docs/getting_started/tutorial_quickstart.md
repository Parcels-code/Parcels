---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart tutorial

Welcome to the **Parcels** quickstart tutorial, in which we will go through all the necessary steps to run a simulation. The code in this notebook can be used as a starting point to run Parcels in your own environment. Along the way we will familiarize ourselves with some specific classes and methods. If you are ever confused about one of these and want to read more, we have a [concepts overview](concepts_overview.md) discussing them in more detail. Let's dive in!

## Imports

Parcels depends on `xarray`, expecting inputs in the form of [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) and writing output files that can be read with xarray.

```{code-cell}
import numpy as np
import xarray as xr
import parcels
```

## Input flow fields: `FieldSet`

A Parcels simulation of Lagrangian trajectories of virtual particles requires two inputs; the first is a set of hydrodynamics fields in which the particles are tracked. Here we provide an example using a subset of the [Global Ocean Physics Reanalysis](https://doi.org/10.48670/moi-00021) from the Copernicus Marine Service.

```{code-cell}
example_dataset_folder = parcels.download_example_dataset(
    "CopernicusMarine_data_for_Argo_tutorial"
)

ds_fields = xr.open_mfdataset(f"{example_dataset_folder}/*.nc", combine="by_coords")
ds_fields.load()  # load the dataset into memory
ds_fields
```

As we can see, the reanalysis dataset contains eastward velocity `uo`, northward velocity `vo`, potential temperature (`thetao`) and salinity (`so`) fields. 

These hydrodynamic fields need to be stored in a `parcels.FieldSet` object. Parcels provides tooling to parse many types of models or observations into such a `parcels.FieldSet` object. Here, we use `FieldSet.from_copernicusmarine()`, which recognizes the standard names of a velocity field:

```{code-cell}
fieldset = parcels.FieldSet.from_copernicusmarine(ds_fields)
```

The subset contains a region of the Agulhas current along the southeastern coast of Africa:

```{code-cell}
temperature = ds_fields.isel(time=0, depth=0).thetao.plot(cmap="magma")
velocity = ds_fields.isel(time=0, depth=0).plot.quiver(x="longitude", y="latitude", u="uo", v="vo")
```

## Input virtual particles: `ParticleSet`

Now that we have created a `parcels.FieldSet` object from the hydrodynamic data, we need to provide our second input: the virtual particles for which we will calculate the trajectories.

We need to create a `parcels.ParticleSet` object with the particles' initial time and position. The `parcels.ParticleSet` object also needs to know about the `FieldSet` in which the particles "live". Finally, we need to specify the type of `parcels.Particle` we want to use. The default particles have `time`, `z`, `lat`, and `lon`, but you can easily add other `Variables` such as size, temperature, or age to create your own particles to mimic plastic or an [ARGO float](../user_guide/examples/tutorial_Argofloats.ipynb).

```{code-cell}
# Particle locations and initial time
npart = 10  # number of particles to be released
# release particles in a line along a meridian
lat = np.linspace(-32.5, -30.5, npart)
lon = np.repeat(32, npart)
time = np.repeat(ds_fields.time.values[0], npart) # at initial time of input data
z = np.repeat(ds_fields.depth.values[0], npart) # at the first depth (surface)

pset = parcels.ParticleSet(
    fieldset=fieldset, pclass=parcels.Particle, time=time, z=z, lat=lat, lon=lon
)
```

```{code-cell}
temperature = ds_fields.isel(time=0, depth=0).thetao.plot(cmap="magma")
velocity = ds_fields.isel(time=0, depth=0).plot.quiver(x="longitude", y="latitude", u="uo", v="vo")
ax = temperature.axes
ax.scatter(lon,lat,s=40,c='w',edgecolors='r');
```

## Compute: `Kernel`

After setting up the input data and particle start locations and times, we need to specify what calculations to do with the particles. These calculations, or numerical integrations, will be performed by what we call a `Kernel`, operating on all particles in the `ParticleSet`. The most common calculation is the advection of particles through the velocity field. Parcels comes with a number of standard kernels, from which we will use the Runge-Kutta advection kernel `AdvectionRK2`:

```{note}
TODO: link to a list of included kernels
```

```{code-cell}
kernels = [parcels.kernels.AdvectionEE]
```

## Prepare output: `ParticleFile`

Before starting the simulation, we must define where and how frequent we want to write the output of our simulation. We can define this in a `ParticleFile` object:

```{code-cell}
output_file = parcels.ParticleFile("output-quickstart.zarr", outputdt=np.timedelta64(1, "h"))
```

The output files are in `.zarr` [format](https://zarr.readthedocs.io/en/stable/), which can be read by `xarray`. See the [Parcels output tutorial](./tutorial_output.ipynb) for more information on the zarr format. We want to choose the `outputdt` argument so that it captures the smallest timescales of our interest.

## Run Simulation: `ParticleSet.execute()`

Finally, we can run the simulation by _executing_ the `ParticleSet` using the specified list of `kernels`. Additionally, we need to specify:

- the `runtime`: for how long we want to simulate particles.
- the `dt`: the timestep with which to perform the numerical integration in the `kernels`. Depending on the numerical integration scheme, the accuracy of our simulation will depend on `dt`. Read [this notebook](https://github.com/Parcels-code/10year-anniversary-session2/blob/8931ef69577dbf00273a5ab4b7cf522667e146c5/advection_and_windage.ipynb) to learn more about numerical accuracy.

```{note}
TODO: add Michaels 10-years Parcels notebook to the user guide
```

```{code-cell}
:tags: [hide-output]
pset.execute(
    kernels,
    runtime=np.timedelta64(1, "D"),
    dt=np.timedelta64(5, "m"),
    output_file=output_file,
)
```

## Read output

To start analyzing the trajectories computed by **Parcels**, we can open the `ParticleFile` using `xarray`:

```{code-cell}
ds_particles = xr.open_zarr("output-quickstart.zarr")
ds_particles
```

The 10 particle trajectories are stored along the `trajectory` dimension, and each trajectory contains 25 observations (initial values + 24 hourly timesteps) along the `obs` dimension. The [working with Parcels output tutorial](./tutorial_output.ipynb) provides more detail about the dataset and how to analyse it.

Let's verify that Parcels has computed the advection of the virtual particles!

```{code-cell}
import matplotlib.pyplot as plt

# plot positions and color particles by number of observation
scatter = plt.scatter(ds_particles.lon.T, ds_particles.lat.T, c=np.repeat(ds_particles.obs.values,npart))
plt.scatter(lon,lat,facecolors="none",edgecolors='r') # starting positions
plt.xlim(31,33)
plt.ylabel("Latitude [deg N]")
plt.ylim(-33,-30)
plt.colorbar(scatter, label="Observation number")
plt.show()
```

That looks good! The virtual particles released in a line along the 32nd meridian (dark blue) have been advected by the flow field.

## Running a simulation backwards in time

Now that we know how to run a simulation, we can easily run another and change one of the settings. We can trace back the particles from their current to their original position by running the simulation backwards in time. To do so, we can simply make `dt` < 0.

```{note}
We have not edited the `ParticleSet`, which means that the new simulation will start with the particles at their current location!
```

```{code-cell}
:tags: [hide-output]
# set up output file
output_file = parcels.ParticleFile("output-backwards.zarr", outputdt=np.timedelta64(1, "h"))

# execute simulation in backwards time
pset.execute(
    kernels,
    runtime=np.timedelta64(1, "D"),
    dt=-np.timedelta64(5, "m"),
    output_file=output_file,
)
```

When we check the output, we can see that the particles have returned to their original position!

```{code-cell}
ds_particles_back = xr.open_zarr("output-backwards.zarr")

scatter = plt.scatter(ds_particles_back.lon.T, ds_particles_back.lat.T, c=np.repeat(ds_particles_back.obs.values,npart))
plt.scatter(lon,lat,facecolors="none",edgecolors='r') # starting positions
plt.xlabel("Longitude [deg E]")
plt.xlim(31,33)
plt.ylabel("Latitude [deg N]")
plt.ylim(-33,-30)
plt.colorbar(scatter, label="Observation number")
plt.show()
```

Using Euler forward advection, the final positions are equal to the original positions with an accuracy of 2 decimals:

```{code-cell}
# testing that final location == original location
np.testing.assert_almost_equal(ds_particles_back['lat'].values[:,-1],ds_particles['lat'].values[:,0], 2)
np.testing.assert_almost_equal(ds_particles_back['lon'].values[:,-1],ds_particles['lon'].values[:,0], 2)
```
