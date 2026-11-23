# 📖 Parcels v4 philosophy

Supported by funding from the [WarmWorld](https://www.warmworld.de) [ELPHE](https://www.geomar.de/en/research/ongoing-projects/project-details/prj/376612?cHash=449a07d057ea0d890287bd2b8ca26e8b) project and an [NWO Vici project](https://www.nwo.nl/en/researchprogrammes/nwo-talent-programme/projects-vici/vici-2022), the Parcels team has worked on a near-complete rewrite of the Parcels codebase in 2025/2026.

The key goals of this update were

1. to support `Fields` on unstructured grids;
2. to allow for user-defined interpolation methods (somewhat similar to user-defined kernels);
3. to make the codebase more modular, easier to extend, and more maintainable;
4. to align Parcels more with other tools in the [Pangeo ecosystem](https://www.pangeo.io/#ecosystem), particularly by leveraging `xarray` more; and
5. to improve the performance of Parcels.

## Rationale for dropping JIT support in v4

Since Parcels v4, we have dropped support for Just-In-Time (JIT) compilation. This decision was made for the following reasons:

1. We want to leverage the power of `xarray` and `uxarray` for data handling and interpolation. These libraries are not compatible with the JIT compilation in v3.
2. We want to make the codebase more maintainable and easier to understand. The JIT compilation pre-v4 adds complexity to the codebase and makes it harder to debug and maintain.
3. We have quite a few features in Parcels (also v3) that only work in Scipy mode (particle-particle interaction, particle-field interaction, etc.).
4. We want users to write more flexible/complex kernels. JIT doesn't support calling functions, or using methods from `numpy` or `scipy`, while this is possible in Scipy mode.

Essentially, the only advantage of JIT in Parcels v3 was its speed. But we think we have some good ideas how to speed up Parcels itself without JIT compilation, such as relying more on vectorized operations. Now that the ecosystem for just-in-time compilation with python has matured in the last 10 years, we want to leverage other packages and methods (`cython`, `numba`, `jax`?) and Python internals for speed-up.
