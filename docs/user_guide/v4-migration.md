---
html_theme.sidebar_secondary.remove: true
---

# Parcels v4 migration guide

This migration guide gives some tips if you want to migrate your Parcels v3 code to Parcels v4. The biggest changes are in the [Kernel API](#kernels) and the way that [FieldSets are created](#fieldset). The other changes are mostly small and should be easy to fix.

## Kernels

| #   | Description of change                                                                                                                                              | How to migrate                                                                                                                                                                               |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | The Kernel loop is 'vectorized': the input of a Kernel is a collection of particles                                                                                | Replace `if`-statements with `numpy.where` statements or [boolean indexing](examples/tutorial_Argofloats)                                                                                    |
| 2   | Functions should work on a collection of particles instead of a single particle                                                                                    | Use `numpy` functions instead of `math` functions                                                                                                                                            |
| 3   | The `time` argument in the Kernel signature is removed                                                                                                             | Use `particle.t`                                                                                                                                                                             |
| 4   | `particle.lon`, `particle.lat` and `particle.depth` are renamed                                                                                                    | Use `particles.x`, `particles.y` and `particles.z`                                                                                                                                           |
| 5   | `particle_dlon` `particle_dlat` and `particle_ddepth` are renamed                                                                                                  | Use `particles.dx`, `particles.dy` and `particles.dz`                                                                                                                                        |
| 6   | The `particle` argument in the Kernel signature is renamed to `particles`                                                                                          | Change the argument name in your Kernel signature to `particles`                                                                                                                             |
| 7   | `particle.delete()` is no longer valid                                                                                                                             | Use `particle.state = StatusCode.Delete`                                                                                                                                                     |
| 8   | Kernels are not concatenated under the hood, so can't access each others variables or states                                                                       | Use fieldset.context or particle data to share information between kernels                                                                                                                   |
| 9   | The `InteractionKernel` class is removed as normal Kernels now have access to _all_ particles                                                                      | Particle-particle interaction can be [performed within normal Kernels](examples/tutorial_interaction)                                                                                        |
| 10  | Automatic conversion from depth to sigma grids under the hood is removed                                                                                           | Explicitly use `convert_z_to_sigma_croco` in sampling kernels (such as the `AdvectionRK4_3D_CROCO` or `SampleOMegaCroco` kernels) when working with [CROCO data](examples/tutorial_croco_3D) |
| 11  | The default advection scheme is changed from RK4 to RK2 as it is [faster while the accuracy is comparable for most applications](examples/tutorial_dt_integrators) | Use `pset.execute(parcels.AdvectionRK2)` for advection                                                                                                                                       |

## FieldSet

| #   | Description of change                                                                      | How to migrate                                                                                          |
| --- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| 12  | `FieldSet.interp_method` doesn't accept a string (e.g. `"linear"` or `"nearest"`)          | Use an Interpolation function such as `parcels.interpolators.Linear` or `parcels.interpolators.Nearest` |
| 13  | `FieldSet.add_constant` is removed to reflect that this value no longer has to be constant | Use `FieldSet.add_context` to add a context variable to the FieldSet                                    |

## Particle

| #   | Description of change                | How to migrate                                                         |
| --- | ------------------------------------ | ---------------------------------------------------------------------- |
| 14  | `Particle.add_variable()` is removed | Use `Particle.add_variables()`, which also takes a list of `Variables` |

## ParticleSet

| #   | Description of change                                                                                            | How to migrate                                                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 15  | pset.execute() does not require Kernel objects                                                                   | Simply pass the function(s) as a list to pset.execute()                                                                                                |
| 16  | `repeatdt` is removed                                                                                            | See the [Delayed starts tutorial](examples/tutorial_delaystart.ipynb#release-particles-repeatedly) for how to implement repeated releases of particles |
| 17  | `lonlatdepth_dtype` is removed                                                                                   | Set the dtype of your particle variables directly                                                                                                      |
| 18  | ParticleSet.execute() expects `datatime`, `numpy.datetime64` or `numpy.timedelta.64` for `runtime` and `endtime` | Update `runtime` and `endtime` to use `numpy.datetime64` or `numpy.timedelta.64`                                                                       |
| 19  | `ParticleSet.from_field()`, `ParticleSet.from_line()`, `ParticleSet.from_list()` are removed                     | Use [`ParticleSet` constructor](examples/tutorial_delaystart)directly                                                                                  |

## ParticleFile

| #   | Description of change                                                        | How to migrate                                                                                                                                     |
| --- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 20  | ParticleFiles output is in [parquet format](getting_started/tutorial_output) | Read the output with `polars.read_parquet` or (to automatically handle cftime) `parcels.read_particlefile`                                         |
| 21  | `ParticleFile` is not a method of the `ParticleSet` class                    | Use `ParticleFile(...)` instead of `pset.ParticleFile(...)`                                                                                        |
| 22  | `ParticleFile` errors out if there's existing output                         | Remove the existing output file or add `mode="w"` to overwrite                                                                                     |
| 23  | The output file does not have a `trajectory` dimension                       | Use `particle_id` instead of `trajectory` in your code                                                                                             |
| 24  | `ParticleFile` does not have a `name` attribute                              | Use `ParticleFile.path`, which can be a string or a Path                                                                                           |
| 25  | `ParticleFile` does not have a `chunks` argument                             | Remove the `chunks` argument from your code                                                                                                        |
| 26  | `ParticleFile` does not have a `to_write` argument                           | Remove the `to_write` argument from your code                                                                                                      |
| 27  | Particles are not written when they are deleted                              | Call `ParticleFile.write()` [inside a kernel to write out](examples/tutorial_write_in_kernel.ipynb#writing-on-particle-deletion) deleted particles |

## Field

| #   | Description of change                                                           | How to migrate                                                                                                                                                           |
| --- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 28  | Calling `Field.eval()` directly gives a warning if any values are out of bounds | Use `Field.eval()` as before, but check for warnings                                                                                                                     |
| 29  | `Field.eval()` returns an array of floats (related to the vectorization)        | Use `Field.eval()` as before, but expect an array of floats instead of a single float                                                                                    |
| 30  | The `NestedField` class is removed                                              | See the [Nested Grids tutorial](examples/tutorial_nestedgrids) for how to set up Nested Grids in v4                                                                      |
| 31  | `applyConversion` is removed                                                    | Interpolation on VectorFields automatically converts from m/s to degrees/s for spherical meshes. Other conversion of units should be handled in Interpolators or Kernels |

## GridSet

| #   | Description of change | How to migrate                                              |
| --- | --------------------- | ----------------------------------------------------------- |
| 32  | `GridSet` is a list   | Change `fieldset.gridset.grids[0]` to `fieldset.gridset[0]` |

## UnitConverters

| #   | Description of change                | How to migrate                                                                                                                                                                   |
| --- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 33  | The `UnitConverter` class is removed | Remove any `UnitConverter` usage from your code and Interpolation functions should handle unit conversion internally, based on the value of `grid._mesh` ("spherical" or "flat") |
