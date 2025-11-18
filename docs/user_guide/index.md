# User guide

The core of our user guide is a series of Jupyter notebooks which document how to implement specific Lagrangian simulations with the flexibility of **Parcels**. Before diving into these advanced _how-to_ guides, we suggest users get started by reading the explanation of the core concepts and trying the quickstart tutorial. For a description of the specific classes and functions, check out the [API reference](../reference.md). To discover other community resources, check out our [Community](../community/index.md) page.

```{note}
The tutorials written for Parcels v3 are currently being updated for Parcels v4. Shown below are only the notebooks which have been updated.
[Feel free to post a Discussion on GitHub](https://github.com/Parcels-code/Parcels/discussions/categories/ideas) if you feel like v4 needs a specific tutorial that wasn't in v3, or [post an issue](https://github.com/Parcels-code/Parcels/issues/new?template=01_feature.md) if you feel that the notebooks below can be improved!
```

## Getting started

- [Quickstart Tutorial](../getting_started/tutorial_quickstart.md)
- [Output Tutorial](../getting_started/tutorial_output.ipynb)
- [Concepts Overview](../getting_started/concepts_overview.md)

## How to

```{note}
**Migrate from v3 to v4** using [this migration guide](v4-migration.md)
```

```{toctree}
:caption: Set up FieldSets
:titlesonly:

```

<!-- examples/documentation_indexing.ipynb -->
<!-- examples/tutorial_nemo_curvilinear.ipynb -->
<!-- examples/tutorial_nemo_3D.ipynb -->
<!-- examples/tutorial_croco_3D.ipynb -->
<!-- examples/tutorial_timevaryingdepthdimensions.ipynb -->
<!-- examples/tutorial_periodic_boundaries.ipynb -->
<!-- examples/tutorial_interpolation.ipynb -->
<!-- examples/tutorial_unitconverters.ipynb -->

```{toctree}
:caption: Create ParticleSets
:titlesonly:
examples/tutorial_delaystart.ipynb
```

```{toctree}
:caption: Write Kernels
:titlesonly:

examples/tutorial_sampling.ipynb
examples/tutorial_gsw_density.ipynb
examples/tutorial_Argofloats.ipynb
```

```{toctree}
:caption: Configure interpolation
:titlesonly:

examples/explanation_interpolation.md
examples/tutorial_interpolation.ipynb
```

<!-- examples/tutorial_diffusion.ipynb -->
<!-- examples/tutorial_particle_field_interaction.ipynb -->
<!-- examples/tutorial_interaction.ipynb -->
<!-- examples/tutorial_analyticaladvection.ipynb -->
<!-- examples/tutorial_kernelloop.ipynb -->

```{toctree}
:caption: Other tutorials
:name: tutorial-other

```

<!-- examples/tutorial_peninsula_AvsCgrid.ipynb -->
<!-- examples/documentation_stuck_particles.ipynb -->
<!-- examples/documentation_unstuck_Agrid.ipynb -->
<!-- examples/documentation_LargeRunsOutput.ipynb -->
<!-- examples/documentation_geospatial.ipynb -->
<!-- examples/documentation_advanced_zarr.ipynb -->

```{toctree}
:caption: Worked examples
```

<!-- examples/documentation_homepage_animation.ipynb -->

```{toctree}
:hidden:
:caption: Other
v3 to v4 migration guide <v4-migration>
Example scripts <additional_examples>
```
