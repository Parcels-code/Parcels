import dask
import dask.array as da
import numpy as np
import xarray as xr

from parcels._chunk_cached_array import ChunkCachedArray, wrap_dataset


def test_chunk_cached_data_stays_lazy_until_explicit_materialization():
    loaded = []

    @dask.delayed
    def chunk(index):
        loaded.append(index)
        return np.arange(16 * index, 16 * index + 16).reshape(4, 4)

    array = da.concatenate([da.from_delayed(chunk(i), shape=(4, 4), dtype=int) for i in range(2)])
    dataset = wrap_dataset(xr.Dataset({"value": (("x", "y"), array)}), max_cache_bytes=1024)

    with dask.config.set(scheduler="synchronous"):
        data = dataset.value.data
        assert isinstance(data, da.Array)
        assert loaded == []

        np.testing.assert_array_equal(dataset.value.values, np.arange(32).reshape(8, 4))
        assert sorted(loaded) == [0, 1]

        loaded.clear()
        np.testing.assert_array_equal(np.asarray(dataset.value), np.arange(32).reshape(8, 4))
        assert sorted(loaded) == [0, 1]


def test_chunk_cached_vectorized_selection_reuses_cached_chunk():
    loaded = []

    @dask.delayed
    def chunk(index):
        loaded.append(index)
        return np.arange(16 * index, 16 * index + 16).reshape(4, 4)

    array = da.concatenate([da.from_delayed(chunk(i), shape=(4, 4), dtype=int) for i in range(2)])
    dataset = wrap_dataset(xr.Dataset({"value": (("x", "y"), array)}), max_cache_bytes=1024)
    assert isinstance(dataset.value.variable._data, ChunkCachedArray)
    indices = {"x": xr.DataArray([0, 1], dims="points"), "y": xr.DataArray([1, 2], dims="points")}

    with dask.config.set(scheduler="synchronous"):
        np.testing.assert_array_equal(dataset.value.isel(indices).data, [1, 6])
        assert loaded == [0]
        np.testing.assert_array_equal(dataset.value.isel(indices).data, [1, 6])
        assert loaded == [0]
