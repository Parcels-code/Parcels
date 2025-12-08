---
file_format: mystnb
kernelspec:
  name: python3
---

# Parcels concepts

Parcels is a set of Python classes and methods to create particle tracking simulations. Here, we will explain the basic concepts defined by the most important classes and functions. This overview can be useful to start understanding the different components we use in Parcels, and to structure the code in a simulation script.

A Parcels simulation is generally built up from four different components:

1. [**FieldSet**](#1-fieldset). The input dataset of gridded fields (e.g. ocean current velocity, temperature) in which virtual particles are defined.
2. [**ParticleSet**](#2-particleset). The dataset of virtual particles. These always contain time, z, lat, and lon, for which initial values must be defined, and may contain other variables.
3. [**Kernels**](#3-kernels). Kernels perform some specific operation on the particles every time step (e.g. interpolate the temperature from the temperature field to the particle location).
4. [**Execute**](#4-execution). Execute the simulation. The core method which integrates the operations defined in Kernels for a given time and timestep, and writes output to a ParticleFile.

We discuss each component in more detail below. The subsections titled **"Learn how to"** link to more detailed [how-to guide notebooks](../user_guide/index.md) and more detailed _explanations_ of Parcels functionality are included under **"Read more about"** subsections. The full list of classes and methods is in the [API reference](../reference.md). If you want to learn by doing, check out the [quickstart tutorial](./tutorial_quickstart.md) to start creating your first Parcels simulation.

```{image} ../_static/concepts_diagram.svg
:alt: Parcels concepts diagram
:width: 100%
```

## 1. FieldSet

Parcels provides a framework to simulate particles **within a set of fields**, such as flow velocities and temperature. To start a parcels simulation we must define this dataset with the **`parcels.FieldSet`** class.

The input dataset from which to create a `parcels.FieldSet` can be an [`xarray.Dataset`](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataset) with output from a hydrodynamic model or reanalysis. Such a dataset usually contains a number of gridded variables (e.g. `"U"`), which in Parcels become `parcels.Field` objects. A set of `parcels.Field` objects is stored in a `parcels.FieldSet` in an analoguous way to how `xarray.DataArray` objects combine to make an `xarray.Dataset`.

For several common input datasets, such as the Copernicus Marine Service analysis products, Parcels has a specific method to read and parse the data correctly:

```python
dataset = xr.open_mfdataset("insert_copernicus_data_files.nc")
fieldset = parcels.FieldSet.from_copernicusmarine(dataset)
```

In some cases we might want to combine `parcels.Field`s from different sources in the same `parcels.FieldSet`, such as ocean currents from one dataset and Stokes drift from another. This is possible in Parcels by adding each `parcels.Field` separately:

```python
dataset1 = xr.dataset("insert_current_data_files.nc")
dataset2 = xr.dataset("insert_stokes_data_files.nc")

Ucurrent = parcels.Field(name="Ucurrent", data=dataset1["Ucurrent"], grid=parcels.XGrid.from_dataset(dataset1), interp_method=parcels.interpolators.XLinear)
Ustokes = parcels.Field(name="Ustokes", data=dataset2["Ustokes"], grid=parcels.XGrid.from_dataset(dataset2), interp_method=parcels.interpolators.XLinear)

fieldset = parcels.FieldSet([Ucurrent, Ustokes])
```

### Grid

Each `parcels.Field` is defined on a grid. With Parcels we can simulate particles in fields on both structured (**`parcels.XGrid`**) and unstructured (**`parcels.UxGrid`**) grids. The grid is defined by the coordinates of grid cell nodes, edges, and faces. `parcels.XGrid` objects are based on [`xgcm.Grid`](https://xgcm.readthedocs.io/en/latest/grids.html), while `parcels.UxGrid` objects are based on [`uxarray.Grid`](https://uxarray.readthedocs.io/en/stable/generated/uxarray.Grid.html#uxarray.Grid) objects.

#### Read more about grids

- [Grids explanation](../user_guide/examples/explanation_grids.md)

### Interpolation

To find the value of a `parcels.Field` at any particle location, Parcels uses interpolation. Depending on the variable, grid, and required accuracy, different interpolation methods may be appropriate. Parcels comes with a number of built-in **`parcels.interpolators`**:

```{code-cell}
import parcels

for interpolator in parcels.interpolators.__all__:
  print(f"{interpolator}")
```

### Read more about

- [Interpolation explanation](../user_guide/examples/explanation_interpolation.md)

### Learn how to

- [Use interpolation methods](../user_guide/examples/tutorial_interpolation.ipynb)

## 2. ParticleSet

Once the environment has a `parcels.FieldSet` object, you can start defining your particles in a **`parcels.ParticleSet`** object. This object requires:

1. The `parcels.FieldSet` object in which the particles will be released.
2. The type of `parcels.Particle`: A default `Particle` or a custom `Particle`-type with additional `Variable`s.
3. Initial conditions for each `Variable` defined in the `Particle`, most notably the release coordinates of `time`, `z`, `lat` and `lon`.

### Learn how to

- [Release particles at different times](../user_guide/examples/tutorial_delaystart.ipynb)

## 3. Kernels

**`parcels.Kernel`** objects are little snippets of code, which are applied to the particles in the `ParticleSet`, for every time step during a simulation. They define the computation or numerical integration done by Parcels, and can represent many processes such as advection, ageing, growth, or simply the sampling of a field.

Basic kernels are included in Parcels, among which several types of advection kernels:

```{code-cell}
for kernel in parcels.kernels.advection.__all__:
  print(f"{kernel}")
```

We can also write custom kernels, to add certain types of 'behaviour' to the particles. To do so we write a function with two arguments: `particles` and `fieldset`. We can then write any computation as a function of any variables defined in the `Particle` and any `Field` defined in the `FieldSet`. Kernels can then be combined by creating a `list` of the kernels. Note that the kernels are executed in order:

```python
# Create a custom kernel which displaces each particle southward

def NorthVel(particles, fieldset):
    vvel = -1e-4
    particles.dlat += vvel * particles.dt


# Create a custom kernel which keeps track of the particle age

def Age(particles, fieldset):
    particles.age += particles.dt

# define all kernels to be executed on particles using an (ordered) list
kernels = [Age, NorthVel, parcels.kernels.AdvectionRK4]
```

```{note}
Every Kernel must be a function with the following (and only those) arguments: `(particles, fieldset)`
```

```{warning}
It is advised _not_ to update the particle coordinates (`particles.time`, `particles.z`, `particles.lat`, or `particles.lon`) directly, as that can negatively interfere with the way that particle movements by different kernels are vectorially added. Use a change in the coordinates: `particles.dlon`, `particles.dlat`, `particles.dz`, and/or `particles.dt` instead, and be careful with `particles.dt`. See also the [kernel loop tutorial](https://docs.oceanparcels.org/en/latest/examples/tutorial_kernelloop.html).
```

```{warning}
We have to be careful with writing kernels for vector fields on Curvilinear grids. While Parcels automatically rotates the "U" and "V" field when necessary, this is not the case for other fields such as Stokes drift. [This guide](../user_guide/examples/tutorial_nemo_curvilinear.ipynb) describes how to use a curvilinear grid in Parcels.
```

### Read more about

- [The Kernel loop](../user_guide/examples/explanation_kernelloop.md)

### Learn how to

- [Sample fields like temperature](../user_guide/examples/tutorial_sampling.ipynb).
- [Mimic the behaviour of ARGO floats](../user_guide/examples/tutorial_Argofloats.ipynb).
- [Add diffusion to approximate subgrid-scale processes and unresolved physics](../user_guide/examples/tutorial_diffusion.ipynb).
- [Convert between units in m/s and degree/s](../user_guide/examples/tutorial_unitconverters.ipynb).

## 4. Execution

The execution of the simulation, given the `FieldSet`, `ParticleSet`, and `Kernels` defined in the previous steps, is done using the method **`parcels.ParticleSet.execute()`**. This method requires the following arguments:

1. The kernels to be executed.
2. The `runtime` defining how long the execution loop runs. Alternatively, you may define the `endtime` at which the execution loop stops.
3. The timestep `dt` at which to execute the kernels.
4. (Optional) The `ParticleFile` object to write the output to.

### Output

To analyse the particle data generated in the simulation, we need to define a `parcels.ParticleFile` and add it as an argument to `parcels.ParticleSet.execute()`. The output will be written in a [zarr format](https://zarr.readthedocs.io/en/stable/), which can be opened as an `xarray.Dataset`. The dataset will contain the particle data with at least `time`, `z`, `lat` and `lon`, for each particle at timesteps defined by the `outputdt` argument.

There are many ways to analyze particle output, and although we provide [a short tutorial to get started](./tutorial_output.ipynb), we recommend writing your own analysis code and checking out other projects such as [trajan](https://opendrift.github.io/trajan/index.html) and [Lagrangian Diagnostics](https://lagrangian-diags.readthedocs.io/en/latest/).

#### Learn how to

- [choose an appropriate timestep](../user_guide/examples/tutorial_numerical_accuracy.ipynb)
- [work with Parcels output](./tutorial_output.ipynb)
