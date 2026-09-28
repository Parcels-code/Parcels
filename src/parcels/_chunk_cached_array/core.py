from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _is_equal_length_1d(key: tuple[Any, ...]) -> bool:
    """Return whether every indexer is a 1D integer array of the same length."""
    if not key:
        return False
    length: int | None = None
    for item in key:
        if not isinstance(item, np.ndarray) or item.ndim != 1:
            return False
        item_length = int(item.shape[0])
        if length is None:
            length = item_length
        elif item_length != length:
            return False
    return True


def _raise_if_out_of_bounds(key: tuple[Any, ...], shape: tuple[int, ...]) -> None:
    """Raise IndexError if an integer indexer points outside its axis.

    Parameters
    ----------
    key : tuple
        Per-axis indexer. Slices are ignored; integer arrays are checked.
    shape : tuple of int
        Length of each axis of the array being indexed.
    """
    for axis, item in enumerate(key):
        if not isinstance(item, np.ndarray):
            continue
        size = shape[axis]
        out_of_bounds = (item < -size) | (item >= size)
        if item.size and np.any(out_of_bounds):
            bad = int(np.ravel(item)[int(np.flatnonzero(out_of_bounds)[0])])
            raise IndexError(f"index {bad} is out of bounds for axis {axis} with size {size}")


class ChunkCachedArray(ExplicitlyIndexedNDArrayMixin):
    """Chunk-level LRU cache on top of a dask array for vectorized indexing.

    Implements xarray's ExplicitlyIndexed protocol so it can be used as
    a drop-in replacement for the dask array in ``da.data``. Xarray's
    ``.isel()`` with vectorized indexers will route through ``_vindex_get``,
    which uses the chunk cache. Other indexing modes delegate to the
    underlying dask array.

    On each vectorized index:
      1. Expands mixed slice and array keys into flat per-dimension indices.
         Integer indexers are broadcast, and slice axes follow those broadcast
         axes (xarray's vectorized-indexing order). Equal-length 1D integer
         keys skip this step.
      2. Maps global indices -> (chunk_coord, local_index) per dimension.
      3. Fetches missing chunks via dask_array.blocks[...].compute().
      4. Assembles the result from cached numpy arrays, reshaping mixed keys.
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
            1D array of length N with the selected values. An empty selection
            returns an empty array without fetching a chunk.

        Raises
        ------
        IndexError
            If an index is smaller than ``-size`` or greater than or equal to
            ``size`` on its axis.
        """
        ndim = len(self.array.chunks)
        assert len(indices) == ndim
        _raise_if_out_of_bounds(indices, tuple(int(size) for size in self.array.shape))
        n_points = len(indices[0])
        if n_points == 0:
            return np.empty(0, dtype=self.array.dtype)

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
        sorted_flat_keys = flat_keys[sort_order]  # type: ignore[call-overload]

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
        """Gather values for an xarray vectorized indexer.

        Parameters
        ----------
        indexer : VectorizedIndexer
            One slice or integer array per dimension. Equal-length 1D integer
            keys are gathered directly. Mixed keys are broadcast, each slice is
            expanded with ``slice.indices`` onto an axis after those broadcast
            axes, and the gathered values are reshaped to that result.

        Returns
        -------
        np.ndarray
            Selected values in xarray's vectorized-indexing order.
        """
        key = indexer.tuple
        if _is_equal_length_1d(key):
            return self._raw_vindex(*key)

        # Reject before expanding slices or touching a chunk. Valid negative
        # indices stay negative here; ``_raw_vindex`` normalizes them.
        _raise_if_out_of_bounds(key, tuple(int(size) for size in self.array.shape))
        flat_indices, result_shape = self._flatten_mixed_vindex(key)
        if 0 in result_shape:
            return np.empty(result_shape, dtype=self.array.dtype)
        return self._raw_vindex(*flat_indices).reshape(result_shape)

    def _flatten_mixed_vindex(self, key: tuple[Any, ...]) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
        """Broadcast integer indexers and expand slices to flat coordinates.

        Parameters
        ----------
        key : tuple
            One slice or integer array per dimension.

        Returns
        -------
        tuple
            ``(flat_indices, result_shape)``. ``flat_indices`` is empty when
            ``result_shape`` contains a zero-size axis. Otherwise it holds one
            1D coordinate array per dimension, raveled in ``result_shape`` order:
            the broadcast integer-index shape, then one axis per slice.
        """
        shape = tuple(int(size) for size in self.array.shape)
        if len(key) != len(shape):
            raise ValueError(f"vectorized indexer length {len(key)} does not match ndim {len(shape)}")

        array_shapes: list[tuple[int, ...]] = []
        slice_lengths: list[int] = []
        normalized_slices: list[tuple[int, int, int]] = []
        for axis, item in enumerate(key):
            if isinstance(item, np.ndarray):
                array_shapes.append(tuple(int(size) for size in item.shape))
            elif isinstance(item, slice):
                start, stop, step = item.indices(shape[axis])
                normalized_slices.append((start, stop, step))
                slice_lengths.append(len(range(start, stop, step)))
            else:
                raise TypeError(f"unsupported vectorized indexer type: {type(item)!r}")

        broadcast_shape = np.broadcast_shapes(*array_shapes) if array_shapes else ()
        result_shape = broadcast_shape + tuple(slice_lengths)
        if 0 in result_shape:
            return (), result_shape

        # Place each slice on its own axis after the broadcast index axes.
        n_slices = len(slice_lengths)
        expanded: list[np.ndarray] = []
        slice_axis = 0
        for item in key:
            if isinstance(item, np.ndarray):
                reshaped = np.reshape(item, tuple(int(size) for size in item.shape) + (1,) * n_slices)
                expanded.append(np.broadcast_to(reshaped, broadcast_shape + (1,) * n_slices))
                continue
            start, stop, step = normalized_slices[slice_axis]
            positions = np.arange(start, stop, step)
            slice_shape = (
                (1,) * (len(broadcast_shape) + slice_axis) + (int(positions.size),) + (1,) * (n_slices - slice_axis - 1)
            )
            expanded.append(positions.reshape(slice_shape))
            slice_axis += 1

        flat_indices = tuple(np.reshape(idx, -1) for idx in np.broadcast_arrays(*expanded))
        return flat_indices, result_shape

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
