from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xarray.core.indexing import BasicIndexer, ExplicitlyIndexedNDArrayMixin, OuterIndexer, VectorizedIndexer
from xarray.namedarray.pycompat import is_duck_array

from .lru_cache import ByteBoundedLRUCache

if TYPE_CHECKING:
    import dask.array
    import xarray as xr


def wrap_dataset(ds: xr.Dataset, max_cache_bytes: int) -> xr.Dataset:
    """Replace all dask-backed data variables with ChunkCachedArray wrappers.

    Returns a shallow copy of the dataset. Each dask-backed data variable's
    internal ``variable._data`` is swapped for a ``ChunkCachedArray`` that
    caches chunks on vectorized indexing. Coordinate variables are loaded
    eagerly into memory to avoid dask task-graph overhead on every
    ``.isel()`` call.

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset (not modified).
    max_cache_bytes : int
        Maximum cache size in bytes, per variable.

    Returns
    -------
    xr.Dataset
        Copy with dask arrays wrapped in ChunkCachedArray.
    """
    from dask.base import is_dask_collection

    ds = ds.copy()
    # Load coordinates eagerly — they are small 1D arrays and keeping them
    # as dask arrays causes expensive task-graph construction on every .isel().
    for name in list(ds.coords):
        ds[name].load()
    for name in ds.data_vars:
        var = ds[name].variable
        if is_duck_array(var._data) and is_dask_collection(var._data):
            var._data = ChunkCachedArray(var._data, max_cache_bytes)  # type: ignore[assignment, arg-type]
    return ds


class ChunkCachedArray(ExplicitlyIndexedNDArrayMixin):
    """Chunk-level LRU cache on top of a dask array for vectorized indexing.

    Implements xarray's ExplicitlyIndexed protocol so it can be used as
    a drop-in replacement for the dask array in ``da.data``. Xarray's
    ``.isel()`` with vectorized indexers will route through ``_vindex_get``,
    which uses the chunk cache. Other indexing modes delegate to the
    underlying dask array.

    On each vectorized index:
      1. Maps global indices -> (chunk_coord, local_index) per dimension.
      2. Fetches missing chunks via dask_array.blocks[...].compute().
      3. Assembles the result from cached numpy arrays.
    """

    def __init__(self, dask_array: dask.array.Array, max_cache_bytes: int) -> None:
        self.array = dask_array
        self.cache = ByteBoundedLRUCache(max_cache_bytes)

        # Precompute chunk boundaries per dimension.
        # _boundaries[d] is a 1D array of cumulative chunk sizes, e.g., [0, 15, 30].
        self._boundaries: list[np.ndarray] = []
        for dim_chunks in dask_array.chunks:
            self._boundaries.append(np.concatenate(([0], np.cumsum(dim_chunks))))

    def get_duck_array(self):
        return self.array.compute()

    def _raw_vindex(self, *indices: np.ndarray) -> np.ndarray:
        """Vectorized indexing with chunk caching.

        Parameters
        ----------
        *indices : np.ndarray
            One 1D integer index array per dimension. All must have the same length N.

        Returns
        -------
        np.ndarray
            1D array of length N with the selected values.
        """
        ndim = len(self.array.chunks)
        assert len(indices) == ndim
        n_points = len(indices[0])

        # Step 1: Map global indices to chunk coords and local indices.
        # Normalize negative indices (e.g. -1 → last element) to positive,
        # matching standard numpy fancy-indexing semantics.
        indices = tuple(np.where(idx < 0, idx + self.array.shape[d], idx) for d, idx in enumerate(indices))
        chunk_ids = np.empty((ndim, n_points), dtype=np.intp)
        local_indices = np.empty((ndim, n_points), dtype=np.intp)
        for d in range(ndim):
            cid = np.searchsorted(self._boundaries[d], indices[d], side="right") - 1
            chunk_ids[d] = cid
            local_indices[d] = indices[d] - self._boundaries[d][cid]

        # Step 2: Group points by chunk using a structured array for vectorized grouping.
        # Encode each point's chunk coords as a single int for fast grouping.
        # Use np.ravel_multi_index on chunk_ids to get a flat chunk key per point.
        numblocks = np.array(self.array.numblocks, dtype=np.intp)
        flat_keys = np.ravel_multi_index(chunk_ids, numblocks)

        # Sort points by flat chunk key to group them.
        sort_order = np.argsort(flat_keys, kind="quicksort")
        sorted_flat_keys = flat_keys[sort_order]  # type: ignore[index]

        # Find group boundaries.
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(sorted_flat_keys)) + 1, [n_points]))

        out = np.empty(n_points, dtype=self.array.dtype)
        for g in range(len(boundaries) - 1):
            grp_slice = slice(boundaries[g], boundaries[g + 1])
            grp_indices = sort_order[grp_slice]

            # Recover the chunk key tuple from any point in this group.
            key = tuple(int(chunk_ids[d, grp_indices[0]]) for d in range(ndim))

            chunk_data = self.cache.get(key)
            if chunk_data is None:
                chunk_data = self.array.blocks[key].compute()
                self.cache.put(key, chunk_data)

            # Vectorized fancy-index: extract all points from this chunk at once.
            local_idx = tuple(local_indices[d, grp_indices] for d in range(ndim))
            out[grp_indices] = chunk_data[local_idx]

        return out

    # --- ExplicitlyIndexed protocol ---

    def _vindex_get(self, indexer: VectorizedIndexer):
        key = indexer.tuple
        return self._raw_vindex(*key)

    def _oindex_get(self, indexer: OuterIndexer):
        # Delegate to dask for orthogonal indexing
        return self.array[indexer.tuple]

    def __getitem__(self, indexer):
        if isinstance(indexer, VectorizedIndexer):
            return self._vindex_get(indexer)
        if isinstance(indexer, OuterIndexer):
            return self._oindex_get(indexer)
        if isinstance(indexer, BasicIndexer):
            return self.array[indexer.tuple]
        return self.array[indexer]
