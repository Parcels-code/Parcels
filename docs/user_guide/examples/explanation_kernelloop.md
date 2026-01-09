---
file_format: mystnb
kernelspec:
  name: python3
---

# 📖 The Parcels Kernel loop

On this page we discuss Parcels' execution loop, and what happens under the hood when you combine multiple Kernels.

This is not very relevant when you only use the built-in Advection kernels, but can be important when you are writing and combining your own Kernels!

## Background

When you run a Parcels simulation (i.e. a call to `pset.execute()`), the Kernel loop is the main part of the code that is executed. This part of the code loops through time and executes the Kernels for all particle.

In order to make sure that the displacements of a particle in the different Kernels can be summed, all Kernels add to a _change_ in position (`particles.dlon`, `particles.dlat`, and `particles.dz`). This is important, because there are situations where movement kernels would otherwise not commute. Take the example of advecting particles by currents _and_ winds. If the particle would first be moved by the currents and then by the winds, the result could be different from first moving by the winds and then by the currents. Instead, by summing the _changes_ in position, the ordering of the Kernels has no consequence on the particle displacement.

## Basic implementation

Below is a structured overview of how the Kernel loop is implemented. Note that this is for `time` and `lon` only, but the process for `lon` is also applied to `lat` and `z`.

1. Initialise an extra Variable `particles.dlon=0`

2. Within the Kernel loop, for each particle:
   1. Update `particles.lon += particles.dlon`

   2. Update `particles.time += particles.dt` (except for on the first iteration of the Kernel loop)<br>

   3. Set variable `particles.dlon = 0`

   4. For each Kernel in the list of Kernels:
      1. Execute the Kernel

      2. Update `particles.dlon` by adding the change in longitude, if needed

   5. If `outputdt` is a multiple of `particles.time`, write `particles.lon` and `particles.time` to zarr output file

Besides having commutable Kernels, the main advantage of this implementation is that, when using Field Sampling with e.g. `particles.temp = fieldset.Temp[particles.time, particles.z, particles.lat, particles.lon]`, the particle location stays the same throughout the entire Kernel loop. Additionally, this implementation ensures that the particle location is the same as the location of the sampled field in the output file.

## Example with currents and winds

Below is a simple example of some particles at the surface of the ocean. We create an idealised zonal wind flow that will "push" a particle that is already affected by the surface currents. The Kernel loop ensures that these two forces act at the same time and location.

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import parcels

# Load the CopernicusMarine data in the Agulhas region from the example_datasets
example_dataset_folder = parcels.download_example_dataset(
    "CopernicusMarine_data_for_Argo_tutorial"
)

ds_fields = xr.open_mfdataset(f"{example_dataset_folder}/*.nc", combine="by_coords")
ds_fields.load()  # load the dataset into memory

# Create an idealised wind field and add it to the dataset
tdim, ydim, xdim = (len(ds_fields.time),len(ds_fields.latitude), len(ds_fields.longitude))
ds_fields["UWind"] = xr.DataArray(
    data=np.ones((tdim, ydim, xdim)) * np.sin(ds_fields.latitude.values)[None, :, None],
    coords=[ds_fields.time, ds_fields.latitude, ds_fields.longitude])

ds_fields["VWind"] = xr.DataArray(
    data=np.zeros((tdim, ydim, xdim)),
    coords=[ds_fields.time, ds_fields.latitude, ds_fields.longitude])

fieldset = parcels.FieldSet.from_copernicusmarine(ds_fields)

# Set unit converters for custom wind fields
fieldset.UWind.units = parcels.GeographicPolar()
```

Now we define a wind kernel that uses a forward Euler method to apply the wind forcing. Note that we update the `particles.dlon` and `particles.dlat` variables, rather than `particles.lon` and `particles.lat` directly.

```{code-cell}
def wind_kernel(particles, fieldset):
    particles.dlon += (
        fieldset.UWind[particles] * particles.dt
    )
    particles.dlat += (
        fieldset.VWind[particles] * particles.dt
    )
```

First run a simulation where we apply kernels as `[AdvectionRK4, wind_kernel]`

```{code-cell}
:tags: [hide-output]
npart = 10
z = np.repeat(ds_fields.depth[0].values, npart)
lons = np.repeat(31, npart)
lats = np.linspace(-32.5, -30.5, npart)

pset = parcels.ParticleSet(fieldset, pclass=parcels.Particle, z=z, lat=lats, lon=lons)
output_file = parcels.ParticleFile(
    store="advection_then_wind.zarr", outputdt=np.timedelta64(6,'h')
)
pset.execute(
    [parcels.kernels.AdvectionRK4, wind_kernel],
    runtime=np.timedelta64(5,'D'),
    dt=np.timedelta64(1,'h'),
    output_file=output_file,
)
```

Then also run a simulation where we apply the kernels in the reverse order as `[wind_kernel, AdvectionRK4]`

```{code-cell}
:tags: [hide-output]
pset_reverse = parcels.ParticleSet(
    fieldset, pclass=parcels.Particle, z=z, lat=lats,  lon=lons
)
output_file_reverse = parcels.ParticleFile(
    store="wind_then_advection.zarr", outputdt=np.timedelta64(6,"h")
)
pset_reverse.execute(
    [wind_kernel, parcels.kernels.AdvectionRK4],
    runtime=np.timedelta64(5,"D"),
    dt=np.timedelta64(1,"h"),
    output_file=output_file_reverse,
)
```

Finally, plot the trajectories to show that they are identical in the two simulations.

```{code-cell}
# Plot the resulting particle trajectories overlapped for both cases
advection_then_wind = xr.open_zarr("advection_then_wind.zarr")
wind_then_advection = xr.open_zarr("wind_then_advection.zarr")
plt.plot(wind_then_advection.lon.T, wind_then_advection.lat.T, "-")
plt.plot(advection_then_wind.lon.T, advection_then_wind.lat.T, "--", c="k", alpha=0.7)
plt.show()
```

```{warning}
It is better not to update `particles.lon` directly in a Kernel, as it can interfere with the loop above. Assigning a value to `particles.lon` in a Kernel will throw a warning.

Instead, update the local variable `particles.dlon`.
```
