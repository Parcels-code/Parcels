# User guide

The core of our user guide is a series of Jupyter notebooks which document how to implement specific Lagrangian simulations with the flexibility of Parcels.

Before diving into these advanced _how-to_ guides (🖥️), we suggest users get started by reading the explanation (📖) of the core concepts and trying the tutorials (🎓).

For a description of the specific classes and functions, check out the [API reference](../reference/parcels/index). To discover other community resources, check out our [Community](../community/index.md) page.

```{toctree}
:titlesonly:
:maxdepth: 3
installation.md
group_gettingstarted.md
group_convert_models.md
group_fieldsets.md
group_particlesets.md
group_kernels.md
group_interpolation.md
group_simulation.md
group_example_kernels.md
group_misc.md
```

```{note}
If you have code that uses Parcels v3, you can migrate to Parcels v4 using [🎓 the v3 to v4 migration guide](v4-migration.md)
```

## Getting started

```{toctree}
:caption: Getting Started
:name: getting-started
:titlesonly:
gettingstarted_quickstart.md
gettingstarted_output.ipynb
gettingstarted_concepts.md
```

## Converting model data to FieldSets

Below is a list of tutorials that show how to convert data from ocean (🌊) and atmospheric (🍃) models to Parcels FieldSets.

```{toctree}
:caption: Converting 🌊 ocean models
:name: converting-ocean-models
:titlesonly:
convert_copernicusmarine.md
convert_nemo.ipynb
convert_croco_3D.ipynb
convert_mitgcm.ipynb
convert_mom6.ipynb
convert_delft3d.ipynb
convert_fesom.ipynb
convert_schism.ipynb
```

```{toctree}
:caption: Converting 🍃 atmospheric models
:name: converting-atmospheric-models
:titlesonly:
convert_aifs.ipynb
```

```{toctree}
:caption: Converter concepts
:name: converter-concepts
:titlesonly:
convert_writing_functions.md
```

## Work with FieldSets

```{toctree}
:caption: Work with FieldSets
:name: work-with-fieldsets
:titlesonly:
fieldsets_grids.ipynb
fieldsets_sigma_coordinates.ipynb
fieldsets_velocityconversion.ipynb
fieldsets_nestedgrids.ipynb
fieldsets_manipulating_field_data.ipynb
fieldsets_xgcm_derived_fields.ipynb
```

## Create ParticleSets

```{toctree}
:caption: Create ParticleSets
:name: create-particlesets
:titlesonly:
particlesets_delaystart.ipynb
```

## Write Kernels

```{toctree}
:caption: Write Kernels
:name: write-kernels
:titlesonly:

kernels_loopexplanation.md
kernels_sampling.ipynb
kernels_statuscodes.ipynb
kernels_write_in_kernel.ipynb
```

## Set interpolation methods

```{toctree}
:caption: Set interpolation methods
:name: interpolation-methods
:titlesonly:

interpolation_explanation.md
interpolation_tutorial.ipynb
interpolation_peninsula_AvsCgrid.ipynb
interpolation_stuck_particles.ipynb
interpolation_unstuck_Agrid.ipynb
```

<!-- examples/tutorial_analyticaladvection.ipynb -->

## Run a simulation

```{toctree}
:caption: Run a simulation
:name: run-simulation
:titlesonly:
simulation_performance.md
simulation_dt_integrators.ipynb
```

## Example Kernels

```{toctree}
:caption: Example Kernels
:name: example-kernels
:titlesonly:
example_gsw_density.ipynb
example_Argofloats.ipynb
example_diffusion.ipynb
example_interaction.ipynb
```

## Other

```{toctree}
:caption: Other
:name: other
:titlesonly:
📖 v4 development <../v4/index>
🎓 v3 to v4 migration guide <v4-migration>
misc_homepage_animation.md
```

<!-- examples/documentation_geospatial.ipynb -->
