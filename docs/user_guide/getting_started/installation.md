# 📦 Installation guide

## Basic Installation

The simplest way to install the Parcels code is to use Anaconda and the [Parcels conda-forge package](https://anaconda.org/conda-forge/parcels) with the latest release of Parcels. This package will automatically install all the requirements for a fully functional installation of Parcels. This is the "batteries-included" solution probably suitable for most users. Note that we support Python 3.11 and higher.

If you want to install the latest development version of Parcels and work with features that have not yet been officially released, you can follow the instructions in the [development section in our contributing guide](../../development/index.md#development).

The steps below are the installation instructions for Linux, macOS and Windows.

(step-1-above)=

**Step 1:** Install Anaconda's Miniconda following the steps at https://docs.anaconda.com/miniconda/. If you're on Linux or macOS, the following assumes that you installed Miniconda to your home directory.

**Step 2:** Start a terminal (Linux / macOS) or the Anaconda prompt (Windows). Activate the `base` environment of your Miniconda and create an environment containing Parcels, all its essential dependencies, `trajan` (a trajectory plotting dependency used in the notebooks) and the nice-to-have cartopy and jupyter packages:

```bash
conda create -n parcelsv4-env -c conda-forge parcels trajan cartopy jupyter
```

**Step 3:** Activate the newly created Parcels environment:

```bash
conda activate parcelsv4-env
```

```{note}
The next time you start a terminal and want to work with Parcels, activate the environment with `conda activate parcelsv4-env`.
```

**Step 4:** Create a Jupyter Notebook or Python script to set up your first Parcels simulation! The [quickstart tutorial](tutorial_quickstart.md) is a great way to get started immediately. You can also first read about the core [Parcels concepts](explanation_concepts.md) to familiarize yourself with the classes and methods you will use.

## Installation for developers

See the [development section in our contributing guide](../../development/index.md#development) for development instructions.
