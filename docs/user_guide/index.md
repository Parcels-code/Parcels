# User guide

The core of our user guide is a series of Jupyter notebooks which document how to implement specific Lagrangian simulations with the flexibility of **Parcels**. Before diving into these advanced _how-to_ guides, we suggest users get started by reading the explanation of the core concepts and trying the quickstart tutorial. For a description of the specific classes and functions, check out the [API reference](../reference.md). To discover other community resources, check out our [Community](../community/index.md) page.

```{note}
The tutorials written for Parcels v3 are currently being updated for Parcels v4. Shown below are only the notebooks which have been updated.
[Feel free to post a Discussion on GitHub](https://github.com/Parcels-code/Parcels/discussions/categories/ideas) if you feel like v4 needs a specific tutorial that wasn't in v3, or [post an issue](https://github.com/Parcels-code/Parcels/issues/new?template=01_feature.md) if you feel that the notebooks below can be improved!
```

## Getting started

- [Quickstart Tutorial](../getting_started/tutorial_quickstart.md)
- [Output Tutorial](../getting_started/tutorial_output.ipynb)
- [Concepts Explanation](../getting_started/explanation_concepts.md)

## How to

```{note}
**Migrate from v3 to v4** using [this migration guide](v4-migration.md)
```

```{toctree}
:caption: Set up FieldSets
:titlesonly:
how-to-guides/tutorial_nemo_curvilinear.ipynb
how-to-guides/tutorial_unitconverters.ipynb
explanations/explanation_grids.md
```

<!-- how-to-guides/documentation_indexing.ipynb -->
<!-- how-to-guides/tutorial_nemo_3D.ipynb -->
<!-- how-to-guides/tutorial_croco_3D.ipynb -->
<!-- how-to-guides/tutorial_timevaryingdepthdimensions.ipynb -->

```{toctree}
:caption: Create ParticleSets
:titlesonly:
how-to-guides/tutorial_delaystart.ipynb
```

```{toctree}
:caption: Write Kernels
:titlesonly:

explanations/explanation_kernelloop.md
how-to-guides/tutorial_sampling.ipynb
how-to-guides/tutorial_statuscodes.ipynb
how-to-guides/tutorial_gsw_density.ipynb
how-to-guides/tutorial_Argofloats.ipynb
```

```{toctree}
:caption: Set interpolation method
:titlesonly:

explanations/explanation_interpolation.md
how-to-guides/tutorial_interpolation.ipynb
```

<!-- how-to-guides/tutorial_diffusion.ipynb -->
<!-- how-to-guides/tutorial_particle_field_interaction.ipynb -->
<!-- how-to-guides/tutorial_interaction.ipynb -->
<!-- how-to-guides/tutorial_analyticaladvection.ipynb -->
<!-- how-to-guides/tutorial_kernelloop.ipynb -->

```{toctree}
:caption: Other tutorials
:name: tutorial-other

```

<!-- how-to-guides/tutorial_peninsula_AvsCgrid.ipynb -->
<!-- how-to-guides/documentation_stuck_particles.ipynb -->
<!-- how-to-guides/documentation_unstuck_Agrid.ipynb -->
<!-- how-to-guides/documentation_LargeRunsOutput.ipynb -->
<!-- how-to-guides/documentation_geospatial.ipynb -->
<!-- how-to-guides/documentation_advanced_zarr.ipynb -->

```{toctree}
:caption: Worked examples
```

<!-- how-to-guides/documentation_homepage_animation.ipynb -->

```{toctree}
:hidden:
:caption: Other
v3 to v4 migration guide <v4-migration>
Example scripts <additional_examples>
```
