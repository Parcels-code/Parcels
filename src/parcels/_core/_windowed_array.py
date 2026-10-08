"""Transparent rolling time-window cache for lazy (dask-backed) field data.

Assumptions / current limits:
  * ``time`` is the leading dimension of the field (true for both the SGRID and
    UGRID ingestion paths; the structured path transposes to ``(time, ...)``).
  * Valid while the requested time indices stay within the resident window
    (i.e. all particles share the clock). A sample that requests time indices
    spanning more than the retained levels would force reloads.
  * The clock is assumed monotonic but may run in either direction: forward
    (``dt > 0``) or backward (``dt < 0``). Eviction keeps only the levels each
    ``isel`` actually requests, which is symmetric in time -- so direction never
    enters the logic and no integration-direction flag is needed.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from dask.base import is_dask_collection

from parcels._core.warnings import FieldSetWarning

# xarray / uxarray ``isel`` keyword arguments that are NOT dimension indexers.
_NON_INDEXER_KWARGS = frozenset({"drop", "missing_dims", "ignore_grid"})


class WindowedArray:
    """Wrap a lazy DataArray so ``isel`` loads/caches/evicts time levels as NumPy."""

    def __init__(self, data: xr.DataArray, time_dim: str = "time", max_levels: int | None = None):
        if data.dims[0] != time_dim:
            raise ValueError(f"WindowedArray expects {time_dim!r} as the leading dimension, got {data.dims}")
        self._data = data
        self._tdim = time_dim
        self._max = max_levels
        self._cache: xr.DataArray = xr.DataArray(
            np.empty((0, *data.shape[1:]), dtype=data.dtype),
            dims=data.dims,
            coords={time_dim: np.empty(0, dtype=np.intp)},
        )
        # diagnostics
        self.loads = 0
        self.bytes_read = 0
        self._slab_bytes = int(np.prod(data.isel({time_dim: 0}).shape)) * data.dtype.itemsize

    # -- transparency: forward everything we don't override -------------------
    def __getattr__(self, name):
        # __getattr__ only fires for misses; reach _data without recursing.
        return getattr(object.__getattribute__(self, "_data"), name)

    def __repr__(self):
        return (
            f"WindowedArray(time_dim={self._tdim!r}, cached_levels={self._cache[self._tdim].values.tolist()}, "
            f"loads={self.loads})\n{self._data!r}"
        )

    # -- window management ----------------------------------------------------
    def _read_level(self, lvl: int) -> np.ndarray:
        """Bulk, sequential read of one time level into NumPy (the dask->NumPy step)."""
        return np.asarray(self._data.isel({self._tdim: int(lvl)}).values)

    def _ensure(self, levels: np.ndarray) -> None:
        if self._max is not None and levels.size > self._max:
            # If isel requests more levels to be loaded than self._max, then the
            # request must be granted. Otherwise an indexing error will occur when
            # isel attempts to index into the cache. This can cause large memory
            # overhead, potentially beyond the cap set by self._max.
            # The most likely reason for this to occur is non-synchronous particle clocks.
            warnings.warn(
                f"The windowed array cache is attempting to hold {levels.size} time levels "
                f"which exceeds max_level={self._max}; the cache will hold {levels.size} to maintain "
                f"simulation accuracy. This may cause significant memory usage or an OOM error. "
                f"This most likely occured due to non-synchronous particle clockes. Raise max "
                f"levels or narrow the spread of particle times.",
                FieldSetWarning,
                stacklevel=3,
            )

        lo, hi = int(np.min(levels)), int(np.max(levels))
        coord = self._cache[self._tdim].values
        keep = (coord >= lo) & (coord <= hi)

        if self._max is not None:
            cached_in_span = np.flatnonzero(keep)
            non_required = np.array([i for i in cached_in_span if coord[i] not in levels], dtype=int)

            spare = max(self._max - levels.size, 0)
            n_drop = max(non_required.size - spare, 0)

            keep[non_required[:n_drop]] = False

        keep_idxs = np.flatnonzero(keep)
        if keep_idxs.size < coord.size:
            self._cache = self._cache.isel({self._tdim: keep_idxs})

        for lvl in levels:
            lvl = int(lvl)
            if lvl in self._cache[self._tdim].values:
                continue

            slab = self._read_level(lvl)
            self.loads += 1
            self.bytes_read += self._slab_bytes
            slab_as_xr = xr.DataArray(slab[None], dims=self._data.dims, coords={self._tdim: [lvl]})

            # The cache is ordered based on time index
            coord = self._cache[self._tdim].values
            pos = int(np.searchsorted(coord, lvl))

            self._cache = xr.concat(
                [
                    self._cache.isel({self._tdim: slice(0, pos)}),
                    slab_as_xr,
                    self._cache.isel({self._tdim: slice(pos, None)}),
                ],
                dim=self._tdim,
            )

    # -- intercepted indexing -------------------------------------------------
    def isel(self, indexers: dict | None = None, **kwargs):
        sel = dict(indexers) if indexers is not None else {}
        sel.update({k: v for k, v in kwargs.items() if k not in _NON_INDEXER_KWARGS})

        # no time selection, therefore there is no interaction with the cache
        if self._tdim not in sel:
            return self._data.isel(indexers, **kwargs)

        t_ind = sel[self._tdim]
        t_vals = np.asarray(t_ind.values if isinstance(t_ind, xr.DataArray) else t_ind)
        levels = np.unique(t_vals)

        if levels.size == 0:
            # empty selection (e.g. a kernel evaluating an empty particle subset):
            # trim the time dimension since a sized zero slice was selected
            return self._cache.isel(sel).drop_vars(self._tdim)
        else:
            self._ensure(levels)

            # re-assign the time indices requested to the cache indices
            cached_lvls = self._cache[self._tdim].values
            cache_indxs = np.searchsorted(cached_lvls, t_vals)
            sel[self._tdim] = xr.DataArray(cache_indxs, dims=getattr(t_ind, "dims", ()))

            return self._cache.isel(sel)  # return the requested isel directly from the cached DataArray


def maybe_windowed(data: xr.DataArray, max_levels: int | None = None):
    """Wrap dask-backed, field data in a ``WindowedArray``; else pass through.

    NumPy-backed fields (already resident) and fields without a leading ``time``
    dimension are returned unchanged, so existing eager workflows are unaffected.
    Already-wrapped data is returned unchanged.
    """
    if isinstance(data, WindowedArray):
        return data
    if data.dims and data.dims[0] == "time" and is_dask_collection(data.data):
        return WindowedArray(data, max_levels=max_levels)
    elif data.dims and data.dims[0] == "mockT" and is_dask_collection(data.data):
        return data.compute()
    return data
