# 📖 Squeezing performance in Parcels

In many Parcels simulations, the bottle-neck in terms of performance is the retrieval of the hydrodynamic field data from disk. This is especially true for simulations with a relatively small number of particles, where the time spent on retrieving the fields can be much larger than the time spent on computing the particle trajectories.

In this tutorial, we will show how to squeeze performance in Parcels by using a few different techniques. Which technique works best for your case depends on the amount of field data you have and how you can store it on disk.

## Option 1: explicitly load the full FieldSet into memory

**Best for: small Datasets (less than a few GB)**

For relatively small Datasets (less than a few GB), it is possible to load the entire FieldSet into memory. This can be done by calling the `load()` method on the `xarray.Dataset` object:

```{code-cell}
ds = ds.load()
```

This will make Parcels use `numpy` functions in the interpolation routines, which are much faster than the `xarray` functions. However, this will also increase the memory usage of your simulation, so it is not always possible to use this option.

### Advantages and disadvantages

| Advantages                         | Disadvantages                                          |
| ---------------------------------- | ------------------------------------------------------ |
| Very fast and simple to implement. | Will only work if the entire Dataset fits into memory. |

## Option 2: use cached zarr files

**Best for: large Datasets (more than a few GB) and particles distributed over a small part of the domain**

If your Dataset is too large to fit into memory, but your particles are only distributed over a small part of the domain, it could be efficient to use cached zarr files. This can be done by using the (experimental) `zarr.CacheStore` in combination with the `parcels.open_raw_zarr()` function. This will make Parcels only load the chunks that are needed for the particles, and cache these chunks in memory for future use.

```{code-cell}
source_store = zarr.storage.LocalStore(filenames)
cache_store = zarr.storage.MemoryStore()

store = CacheStore(
    store=source_store, cache_store=cache_store, max_size=MAX_CACHE_SIZE
)
ds = parcels.open_raw_zarr(store)
```

### Advantages and disadvantages

| Advantages                                                                                                           | Disadvantages                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Parcels will only load the chunks that are needed for the particles, which can be much less than the entire Dataset. | The hydrodynamic data will have to be stored in zarr format, which may require additional storage space.          |
|                                                                                                                      | The fieldset data can't be changed after it is loaded, as dask operations are not supported on the raw zarr data. |

```{note}
In our performance testing, we have found that using zarr files saved without any compression can be considerably faster than using compressed zarr files. However, we are working on an upstream fix in to make caching compressed zarr files faster, so this may change in the future.
```

## Option 3: use `fieldset.to_windowed_arrays()`

**Best for: large Datasets (more than a few GB) and particles distributed over the entire domain**

If your Dataset is so large that it doesn't fit into memory, you can use the `fieldset.to_windowed_arrays()` method to make Parcels only hold two timeslices in memory. Note that this only works if the two timeslices still fit into memory.

The two timeslices (the current and the next) are fully loaded into memory, so this method is especially useful if your particles are distributed over the entire domain, as all data will then have to be accessed anyway.

```{code-cell}
fieldset.to_windowed_arrays()
```

### Advantages and disadvantages

| Advantages                                                                                   | Disadvantages                                                                                                                            |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Parcels will only hold two timeslices in memory, which is much less than the entire Dataset. | Parcels will still have to load the entire two timeslices into memory, which can be a lot of data if the Dataset is large.               |
| Hydrodynamic files do not have to be reformatted and stored.                                 | Inefficient when the Particles only sample a small part of the domain, as Parcels will still load the entire two timeslices into memory. |

## Option 4: use Dask

**Best for: large Datasets (more than a few GB) and small ParticleSets (less than a few hundred particles)**

If your Dataset is so large that it doesn't fit into memory, and you have very few particles, you can use Dask to perform the interpolation operations. In this case, you don't have to do any special setup, as Parcels will automatically use Dask if the `xarray.Dataset` is a Dask array.

### Advantages and disadvantages

| Advantages           | Disadvantages                                  |
| -------------------- | ---------------------------------------------- |
| Works out-of-the-box | Only performs well for very small ParticleSets |

```{note}
The long-term plan for Parcels development is to make this Option 4 work well for all cases. However, this will require significant work on Dask indexing.
```
