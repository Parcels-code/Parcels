# User guide

The core of our user guide is a series of Jupyter notebooks which document how to implement specific Lagrangian simulations with the flexibility of **Parcels**.

Before diving into these advanced _how-to_ guides (🖥️), we suggest users get started by reading the explanation (📖) of the core concepts and trying the tutorials (🎓).

For a description of the specific classes and functions, check out the [API reference](../reference/parcels/index). To discover other community resources, check out our [Community](../community/index.md) page.

## Installation

```{toctree}
:caption: Installation instructions
:name: installation
:titlesonly:
getting_started/installation.md
```

```{note}
If you have code that uses Parcels v3, you can migrate to Parcels v4 using [this migration guide](v4-migration.md)
```

## Getting started

```{toctree}
:caption: Getting Started
:name: getting-started
:titlesonly:
getting_started/tutorial_quickstart.md
getting_started/tutorial_output.ipynb
getting_started/explanation_concepts.md
```

## Set up FieldSets

```{toctree}
:caption: Set up FieldSets
:name: setup-fieldsets
:titlesonly:
examples/explanation_grids.md
examples/tutorial_nemo.ipynb
examples/tutorial_croco_3D.ipynb
examples/tutorial_mitgcm.ipynb
examples/tutorial_fesom.ipynb
examples/tutorial_schism.ipynb
examples/tutorial_velocityconversion.ipynb
examples/tutorial_nestedgrids.ipynb
examples/tutorial_manipulating_field_data.ipynb
```

<!-- examples/documentation_indexing.ipynb -->
<!-- examples/tutorial_timevaryingdepthdimensions.ipynb -->

## Create ParticleSets

```{toctree}
:caption: Create ParticleSets
:name: create-particlesets
:titlesonly:
examples/tutorial_delaystart.ipynb
```

## Write Kernels

```{toctree}
:caption: Write Kernels
:name: write-kernels
:titlesonly:

examples/explanation_kernelloop.md
examples/tutorial_sampling.ipynb
examples/tutorial_statuscodes.ipynb
examples/tutorial_write_in_kernel.ipynb
```

## Set interpolation methods

```{toctree}
:caption: Set interpolation methods
:name: interpolation-methods
:titlesonly:

examples/explanation_interpolation.md
examples/tutorial_interpolation.ipynb
```

<!-- examples/tutorial_particle_field_interaction.ipynb -->
<!-- examples/tutorial_analyticaladvection.ipynb -->
<!-- examples/tutorial_kernelloop.ipynb -->

## Run a simulation

```{toctree}
:caption: Run a simulation
:name: run-simulation
:titlesonly:
examples/explanation_performance.md
examples/tutorial_dt_integrators.ipynb
```

<!-- examples/tutorial_peninsula_AvsCgrid.ipynb -->
<!-- examples/documentation_advanced_zarr.ipynb -->
<!-- examples/documentation_LargeRunsOutput.ipynb -->

<!-- ```{toctree}
:caption: Other tutorials
:name: tutorial-other

``` -->

<!-- examples/documentation_stuck_particles.ipynb -->
<!-- examples/documentation_unstuck_Agrid.ipynb -->
<!-- examples/documentation_geospatial.ipynb -->

## Example Kernels

```{toctree}
:caption: Example Kernels
:name: example-kernels
:titlesonly:
examples/tutorial_gsw_density.ipynb
examples/tutorial_Argofloats.ipynb
examples/tutorial_diffusion.ipynb
examples/tutorial_interaction.ipynb
```

<!-- examples/documentation_homepage_animation.ipynb -->

```{toctree}
:hidden:
:caption: Other
v3 to v4 migration guide <v4-migration>
```
