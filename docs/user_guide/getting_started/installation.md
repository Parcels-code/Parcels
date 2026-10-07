# 📦 Installation guide

We recommend installing Parcels (Python 3.11+) via the [conda-forge package](https://anaconda.org/conda-forge/parcels), which includes all requirements. These steps work on Linux, macOS and Windows.

**Step 1:** Install [Miniconda](https://docs.anaconda.com/miniconda/) (conda 4.2 or higher; [update conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-conda.html#updating-conda-to-the-current-version) if needed).

**Step 2:** In a terminal (Linux/macOS) or the Anaconda prompt (Windows), create an environment with Parcels, `trajan` (used for plotting in the tutorials), `cartopy` and `jupyter`:

```bash
conda create -n parcels-v4 -c conda-forge parcels trajan cartopy jupyter
```

**Step 3:** Activate the environment (do this each time you start a new terminal):

```bash
conda activate parcels-v4
```

**Step 4:** Get started with the [quickstart tutorial](tutorial_quickstart.md), or first read about the core [Parcels concepts](explanation_concepts.md).

## Installation for developers

To install the latest development version, see the [development section in our contributing guide](../../development/index.md#development).
