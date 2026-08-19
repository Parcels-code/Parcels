---
title: "Parcels v4: An Xarray-aligned, flexible lagrangian simulation framework"
tags:
  - Python
  - Lagrangian modelling
  - Unstructured grids
  - Xarray
  - Pangeo
authors:
  - name: Nick Hodgskin
    orcid: 0009-0003-0778-3183
    corresponding: true
    affiliation: 1
  - name: Erik van Sebille
    orcid: 0000-0003-2041-0704
    affiliation: 1
  - name: Joe Schoonover
    orcid: 0000-0001-5650-7095
    affiliation: 2

affiliations:
  - name: Institute for Marine and Atmospheric Research, Utrecht University, the Netherlands
    index: 1
  - name: Fluid Numerics, Hickory, NC, USA
    index: 2
date: 17 August 2026
bibliography: paper.bib
---

# Summary

Parcels [@Lange2017; @Delandmeter2019] is a highly customisable Lagrangian simulation framework.
Version 4 of the software is a major update which overhauls the software internals to natively consume Xarray CITE dataset objects, and adds support for unstructured grid datasets.
This makes Parcels compatable with many new data formats (e.g., Zarr, Icechunk)
and execution modes (e.g., streaming data from cloud buckets, or data providers such as the [Copernicus Marine Data Store](https://marine.copernicus.eu/)), while also enabling simulations on different grid geometries.

# Statement of need

The first release of Parcels was in ... .
Since then, several factors have changed the climate modelling space.
A significant portion of the geospatial science community has shifted from creating their own scripts for manipulating NetCDF data, to using open source software - namely Xarray - for working with multidimensional climate data.
Major benefits of this software include (a) providing a single in-memory, metadata-rich representation of a full NetCDF-like dataset, (b) its flexibility to manipulate datasets in other data-formats (e.g., Zarr, HDF5), and (c) its abstractions allowing for easy data ingestion from network data sources.

Writing software that natively works from Xarray datasets provides a powerful
abstraction layer allowing downstream developers to create software that works across data formats and data ingestion paradigms.
This is particularly important as climate datasets are continually increasing in resolution and size, preventing local file based storage, and datasets are increasingly being stored in modern data formats such as Zarr.

Another interesting aspect is that climate modellers are increasingly providing model output on different grid geometries which have attractive features compared to conventional structured grids.
Running natively on these grid geometries, which can be represented as unstructured grids, without re-interpolation allows researchers to run simulations that capture the details from the original dataset.

Finally, some naming conventions and abstractions in the Parcels codebase
were previously oriented towards oceanographers.
These domain specific items have been removed in this version, and our documentation has been adapted so that the applicability of Parcels to other domains, such as atmospheric particle tracking, is more apparent.

# Software design

When working with output from a single circulation model, version 4 of Parcels assumes data (i.e., the field data and mesh data) is all contained in a single Xarray dataset object.
This object is either opened directly from disk/a data store, or is constructed by the user from the component Field and mesh files with help from a "converter" function.
These converters also attach relevant CF-convention and grid geometry metadata (SGRID metadata for structured data, UGRID metadata for unstructured data) to the dataset object, allowing the internals of Parcels to assume a certain dataset structure and metadata richness.
Below is a diagram illustrating the code path when
working with structured data.

TODO: insert mermaid diagram `converters.md`

When working with unstructured data, the main difference is the use of Uxarray Datasets instead of Xarray datasets, and using UGRID conventions instead of SGRID conventions.

If a user wants to run a simulation with fields from different models, they load each model data into its own FieldSet and then combine the FieldSets together into a single FieldSet.

TODO: Decide how in-detail we need to go for this paper (can't enumerate everything)

- class diagram of the code?

# Research impact statement

Parcels has been cited in over 200 peer reviewed scientic papers so far
mostly within the field of oceanography.
This software update
expands the reach of Parcels both to more users within oceanography,
and in other domains.

# AI usage disclosure

Large Language Models (Claude Opus 4.6, ChatGPT ... ) have been used in a guided manner for code generation, documentation, refactoring, and testing.
All LLM written code and documentation has been verified by the authors.
This manuscript was fully written by humans.

# Acknowledgements

This work has been funded via Vici ... and via a Warmworld grant ... .

# References
