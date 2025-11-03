# User guide

The core of our user guide is a series of Jupyter notebooks which document how to implement specific Lagrangian simulations with the flexibility of **Parcels**. Before diving into these advanced _how-to_ guides, we suggest users get started by reading the explanation of the core concepts and trying the quickstart tutorial. For Kernels and examples written by users, check out the [parcels contributing repository](https://github.com/Parcels-code/parcels_contributions). For a description of the specific classes and functions, check out the [API reference](../reference.md).

```{note}
The tutorials written for Parcels v3 are currently being updated for Parcels v4. Shown below are only the notebooks which have been updated.
[Feel free to post a Discussion on GitHub](https://github.com/Parcels-code/Parcels/discussions/categories/ideas) if you feel like v4 needs a specific tutorial that wasn't in v3, or [post an issue](https://github.com/Parcels-code/Parcels/issues/new?template=01_feature.md) if you feel that the notebooks below can be improved!
```

## Getting started

```{nbgallery}
:caption: Getting started
:name: getting-started

<!-- examples/explanation_parcels_concepts.md -->
<!-- examples/tutorial_quickstart.ipynb -->
<!-- examples/tutorial_output.ipynb -->
```

## How to:

```{note}
**Migrate from v3 to v4** using [this migration guide](v4-migration.md)
```

```{nbgallery}
:caption: Set up FieldSets
:name: how-to-fieldsets

<!-- examples/explanation_fieldset.md -->
<!-- examples/explanation_grid_indexing.md -->
<!-- examples/how-to_nemo_curvilinear.ipynb -->
<!-- examples/how-to_nemo_3D.ipynb -->
<!-- examples/how-to_croco_3D.ipynb -->
<!-- examples/how-to_timevaryingdepthdimensions.ipynb -->
<!-- examples/how-to_periodic_boundaries.ipynb -->
<!-- examples/how-to_interpolation.ipynb -->
<!-- examples/how-to_unitconverters.ipynb -->
```

```{nbgallery}
:caption: Create ParticleSets
:name: how-to-particlesets

<!-- examples/how-to_delaystart.ipynb -->
```

```{nbgallery}
:caption: Write a custom `Kernel`
:name: how-to-kernels


<!-- examples/explanation_kernelloop.md -->
<!-- examples/how-to_diffusion.ipynb -->
examples/how-to_sampling.ipynb
examples/how-to_gsw_density.ipynb
<!-- examples/how-to_particle_field_interaction.ipynb -->
<!-- examples/how-to_interaction.ipynb -->
<!-- examples/how-to_analyticaladvection.ipynb -->
```

```{nbgallery}
:caption: Write an `Interpolator`
:name: how-to-interpolators
```

```{nbgallery}
:caption: Run an accurate and efficient simulation
:name: how-to-simulation
```

```{nbgallery}
:caption: Other tutorials
:name: how-to-other

<!-- examples/explanation_peninsula_AvsCgrid.ipynb -->
<!-- examples/how-to_MPI.ipynb -->
<!-- examples/explanation_stuck_particles.ipynb -->
<!-- examples/how-to_unstuck_Agrid.ipynb -->
<!-- examples/how-to_LargeRunsOutput.ipynb -->
<!-- examples/how-to_geospatial.ipynb -->
<!-- examples/how-to_advanced_zarr.ipynb -->
```

```{nbgallery}
:caption: Worked examples
:name: how-to-examples

examples/how-to_Argofloats.ipynb
<!-- ../examples/how-to_homepage_animation.ipynb -->
```

```{toctree}
:hidden:
v3 to v4 migration guide <v4-migration>
Example scripts <additional_examples>
Community examples <https://github.com/Parcels-code/parcels_contributions>
```