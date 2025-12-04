---
file_format: mystnb
kernelspec:
  name: python3
---
# Parcels concepts

Parcels is a set of Python classes and methods to create particle tracking simulations. Here, we will explain the basic concepts defined by the most important classes and functions. This overview can be useful to start understanding the names for different components we use in Parcels, and to structure and make appropriate use of the code in a simulation script.

A Parcels simulation is generally built up from four different components:
1. [**FieldSet**](#1-fieldset). The input dataset of gridded fields (e.g. ocean current velocity, temperature) in which virtual particles are defined.
2. [**ParticleSet**](#2-particleset). The dataset of virtual particles. These always contain time, z, lat, and lon, for which initial values must be defined, and may contain other variables.
3. [**Kernels**](#3-kernels). Kernels perform some specific operation on the particles every time step (e.g. interpolate the temperature from the temperature field to the particle location).
4. [**Execute**](#4-execution). Execute the simulation. The core method which integrates the operations defined in Kernels for a given time and timestep, and writes output to a ParticleFile.

We discuss each component in more detail below and link to more detailed [how-to guides](../user_guide/index.md) and the full list of classes and methods in the [API reference](../reference.md). If you want to learn by doing, check out the [quickstart tutorial](./tutorial_quickstart.md) to start creating your first Parcels simulation. 

![png](../user_guide/explanations/images/parcels_user_diagram.png)

## 1. FieldSet

Parcels provides a framework to simulate particles **within a set of fields**, such as flow velocities and temperature. To start a parcels simulation we must define this dataset with the `FieldSet` class.

The input dataset from which to create a `FieldSet` can be an `xarray.Dataset` with output from a hydrodynamic model or reanalysis.

### Grid
read more about [grids](../user_guide/explanations/explanation_grids.md)

### Interpolation
read more about [interpolation](../user_guide/explanations/explanation_interpolation.md)

## 2. ParticleSet


## 3. Kernels
read more about [Kernels](../user_guide/explanations/explanation_kernelloop.md)

## 4. Execution

### Output
ParticleFile