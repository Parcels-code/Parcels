---
file_format: mystnb
kernelspec:
  name: python3
---

# 📖 Kernel loop

On this page we discuss how Parcels executes the Kernel loop, and what happens under the hood when you combine multiple Kernels.

This is not very relevant when you only use the built-in Advection Kernels, but can be important when you are writing and combining your own Kernels!

## Background

When you run a Parcels simulation (i.e. a call to `pset.execute()`), the Kernel loop is the main part of the code that is executed. This part of the code loops through time and executes the Kernels for all particle.

In order to make sure that the displacements of a particle in the different Kernels can be summed, all Kernels add to a _change_ in position (`particles.dx`, `particles.dy`, and `particles.dz`). This is important, because there are situations where movement Kernels would otherwise not commute. Take the example of advecting particles by currents _and_ winds. If the particle would first be moved by the currents and then by the winds, the result could be different from first moving by the winds and then by the currents. Instead, by summing the _changes_ in position, the ordering of the Kernels has no consequence on the particle displacement.

## Basic implementation

Below is a structured overview of how the Kernel loop is implemented. Note that this is for `time` and `x` only, but the process for `y` and `z` is also applied to `y` and `z`.

1. Initialise an extra Variable `particles.dx=0`

2. Within the Kernel loop, for each particle:
   1. Update `particles.x += particles.dx`

   2. Update `particles.time += particles.dt` (except for on the first iteration of the Kernel loop)<br>

   3. Set variable `particles.dx = 0`

   4. For each Kernel in the list of Kernels:
      1. Execute the Kernel

      2. Update `particles.dx` by adding the change in x, if needed

   5. If `outputdt` is a multiple of `particles.time`, write `particles.x` and `particles.time` to zarr output file

Besides having commutable Kernels, the main advantage of this implementation is that, when using Field Sampling with e.g. `particles.temp = fieldset.Temp[particles.time, particles.z, particles.y, particles.x]`, the particle location stays the same throughout the entire Kernel loop. Additionally, this implementation ensures that the particle location is the same as the location of the sampled field in the output file.

## Example with currents and winds

Below is a simple example of some particles at the surface of the ocean. We create an idealised zonal wind flow that will "push" a particle that is already affected by the surface currents. The Kernel loop ensures that these two forces act at the same time and location.

```{note}
Advecting particles by a combination of flow fields can be done with two separate kernels, as is shown below. However, it can also be done by summing the fields directly using `xarray` operations - provided the fields are defined on the same grid and have compatible dimensions. See the [manipulating field data tutorial](tutorial_manipulating_field_data.ipynb) for an example of that approach.
```

```{code-cell}
:tags: [hide-output]
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import parcels
import parcels.tutorial

# Load the CopernicusMarine data in the Agulhas region from the example_datasets
ds_fields = parcels.tutorial.open_dataset("CopernicusMarine_data_for_Argo_tutorial/data")
ds_fields.load()  # load the dataset into memory

# Create an idealised wind field and add it to the dataset
tdim, ydim, xdim = (len(ds_fields.time),len(ds_fields.latitude), len(ds_fields.longitude))
ds_fields["UWind"] = xr.DataArray(
    data=np.ones((tdim, ydim, xdim)) * np.sin(ds_fields.latitude.values)[None, :, None],
    coords=[ds_fields.time, ds_fields.latitude, ds_fields.longitude])

ds_fields["VWind"] = xr.DataArray(
    data=np.zeros((tdim, ydim, xdim)),
    coords=[ds_fields.time, ds_fields.latitude, ds_fields.longitude])

fields = {
    "U": ds_fields["uo"],
    "V": ds_fields["vo"],
    "UWind": ds_fields["UWind"],
    "VWind": ds_fields["VWind"],
}
ds_fset = parcels.convert.copernicusmarine_to_sgrid(fields=fields)
fieldset = parcels.FieldSet.from_sgrid_conventions(
    ds_fset,
    vector_fields={
        "UV": ("U", "V"),
        "Wind": ("UWind", "VWind")
    }
)
```

Now we define a wind Kernel that uses a forward Euler method to apply the wind forcing. Note that we update the `particles.dx` and `particles.dy` variables, rather than `particles.x` and `particles.y` directly.

```{code-cell}
def wind_kernel(particles, fieldset):
    uwind, vwind = fieldset.Wind[particles]
    particles.dx += uwind * particles.dt
    particles.dy += vwind * particles.dt
```

First run a simulation where we apply Kernels as `[AdvectionRK2, wind_kernel]`

```{code-cell}
:tags: [hide-output]
npart = 10
z = np.repeat(ds_fields.depth[0].values, npart)
lons = np.repeat(31, npart)
lats = np.linspace(-32.5, -30.5, npart)

pset = parcels.ParticleSet(fieldset, pclass=parcels.Particle, z=z, y=lats, x=lons)
output_file = parcels.ParticleFile(
    path="advection_then_wind.parquet", outputdt=np.timedelta64(6,'h')
)
pset.execute(
    [parcels.kernels.AdvectionRK2, wind_kernel],
    runtime=np.timedelta64(5,'D'),
    dt=np.timedelta64(1,'h'),
    output_file=output_file,
)
```

Then also run a simulation where we apply the Kernels in the reverse order as `[wind_kernel, AdvectionRK2]`

```{code-cell}
:tags: [hide-output]
pset_reverse = parcels.ParticleSet(
    fieldset, pclass=parcels.Particle, z=z, y=lats, x=lons
)
output_file_reverse = parcels.ParticleFile(
    path="wind_then_advection.parquet", outputdt=np.timedelta64(6,"h")
)
pset_reverse.execute(
    [wind_kernel, parcels.kernels.AdvectionRK2],
    runtime=np.timedelta64(5,"D"),
    dt=np.timedelta64(1,"h"),
    output_file=output_file_reverse,
)
```

Finally, plot the trajectories to show that they are identical in the two simulations.

```{code-cell}
# Plot the resulting particle trajectories overlapped for both cases
advection_then_wind = parcels.read_particlefile("advection_then_wind.parquet")
wind_then_advection = parcels.read_particlefile("wind_then_advection.parquet")

fig, ax = plt.subplots(figsize=(5, 3))
for traj in wind_then_advection.partition_by("particle_id", maintain_order=True):
    ax.plot(traj["x"], traj["y"], "-")
for traj in advection_then_wind.partition_by("particle_id", maintain_order=True):
    ax.plot(traj["x"], traj["y"], "--", c="k", alpha=0.7)
plt.show()
```

```{warning}
It is better not to update `particles.x` directly in a Kernel, as it can interfere with the loop above. Assigning a value to `particles.x` in a Kernel will throw a warning.

Instead, update the local variable `particles.dx`.
```
