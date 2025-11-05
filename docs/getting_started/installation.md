# Installation guide

## Basic Installation

The simplest way to install the Parcels code is to use Anaconda and the [Parcels conda-forge package](https://anaconda.org/conda-forge/parcels) with the latest release of Parcels. This package will automatically install all the requirements for a fully functional installation of Parcels. This is the "batteries-included" solution probably suitable for most users. Note that we support Python 3.10 and higher.

If you want to install the latest development version of Parcels and work with features that have not yet been officially released, you can follow the instructions for a [developer installation](#installation-for-developers).

The steps below are the installation instructions for Linux, macOS and Windows.

(step-1-above)=

**Step 1:** Install Anaconda's Miniconda following the steps at https://docs.anaconda.com/miniconda/. If you're on Linux /macOS, the following assumes that you installed Miniconda to your home directory.

**Step 2:** Start a terminal (Linux / macOS) or the Anaconda prompt (Windows). Activate the `base` environment of your Miniconda and create an environment containing Parcels, all its essential dependencies, `trajan` (a trajectory plotting dependency used in the notebooks) and the nice-to-have cartopy and jupyter packages:

```bash
conda activate base
conda create -n parcels -c conda-forge parcels trajan cartopy jupyter
```

```{note}
For some of the examples, `pytest` also needs to be installed. This can be quickly done with `conda install -n parcels pytest` which installs `pytest` directly into the newly created `parcels` environment.
```

**Step 3:** Activate the newly created Parcels environment:

```bash
conda activate parcels
```

```{note}
The next time you start a terminal and want to work with Parcels, activate the environment with `conda activate parcels`.
```

**Step 4:** Create a Jupyter Notebook or Python script to set up your first Parcels simulation! The [quickstart tutorial](tutorial_quickstart.md) is a great way to get started immediately. You can also first read about the core [Parcels concepts](concepts_overview.md) to familiarize yourself with the classes and methods you will use.

## Installation for developers

See the [development section in our contributing guide](../development/index.md#development) for development instructions.
