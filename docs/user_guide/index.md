# User guide

```{toctree}
:hidden:
v3 to v4 migration guide <v4-migration>
Example scripts <additional_examples>
User contributions
```

The core of our user guide is a series of Jupyter notebooks which document how to implement specific Lagrangian simulations with the flexibility of **Parcels**. Before diving into these advanced _how-to_ guides, we suggest users get started by reading the explanation of the core concepts and trying the quickstart tutorial. For Kernels and examples written by users, check out the [parcels contributing repository](https://github.com/Parcels-code/parcels_contributions). For a description of the specific classes and functions, check out the [API reference](../reference.md).

```{note}
The tutorials written for Parcels v3 are currently being updated for Parcels v4. Shown below are only the notebooks which have been updated.
[Feel free to post a Discussion on GitHub](https://github.com/Parcels-code/Parcels/discussions/categories/ideas) if you feel like v4 needs a specific tutorial that wasn't in v3, or [post an issue](https://github.com/Parcels-code/Parcels/issues/new?template=01_feature.md) if you feel that the notebooks below can be improved!
```

## Getting started

```{nbgallery}
:caption: Getting started
:name: tutorial-overview

<!-- ../examples/tutorial_parcels_structure.ipynb -->
<!-- ../examples/parcels_tutorial.ipynb -->
<!-- ../examples/tutorial_output.ipynb -->
```

## How to:

```{note}
**Migrate from v3 to v4** using [this migration guide](v4-migration.md)
```

```{nbgallery}
:caption: Set up FieldSets
:name: tutorial-fieldsets

<!-- ../examples/documentation_indexing.ipynb -->
<!-- ../examples/tutorial_nemo_curvilinear.ipynb -->
<!-- ../examples/tutorial_nemo_3D.ipynb -->
<!-- ../examples/tutorial_croco_3D.ipynb -->
<!-- ../examples/tutorial_timevaryingdepthdimensions.ipynb -->
<!-- ../examples/tutorial_periodic_boundaries.ipynb -->
<!-- ../examples/tutorial_interpolation.ipynb -->
<!-- ../examples/tutorial_unitconverters.ipynb -->
```

```{nbgallery}
:caption: Create ParticleSets
:name: tutorial-particlesets

<!-- ../examples/tutorial_delaystart.ipynb -->
```

```{nbgallery}
:caption: Write a custom kernel
:name: tutorial-kernels

<!-- ../examples/tutorial_diffusion.ipynb -->
../examples/tutorial_sampling.ipynb
../examples/tutorial_gsw_density.ipynb
<!-- ../examples/tutorial_particle_field_interaction.ipynb -->
<!-- ../examples/tutorial_interaction.ipynb -->
<!-- ../examples/tutorial_analyticaladvection.ipynb -->
<!-- ../examples/tutorial_kernelloop.ipynb -->
```

```{nbgallery}
:caption: Other tutorials
:name: tutorial-other

<!-- ../examples/tutorial_peninsula_AvsCgrid.ipynb -->
<!-- ../examples/documentation_MPI.ipynb -->
<!-- ../examples/documentation_stuck_particles.ipynb -->
<!-- ../examples/documentation_unstuck_Agrid.ipynb -->
<!-- ../examples/documentation_LargeRunsOutput.ipynb -->
<!-- ../examples/documentation_geospatial.ipynb -->
<!-- ../examples/documentation_advanced_zarr.ipynb -->
```

```{nbgallery}
:caption: Worked examples
:name: tutorial-examples

../examples/tutorial_Argofloats.ipynb
<!-- ../examples/documentation_homepage_animation.ipynb -->
```
