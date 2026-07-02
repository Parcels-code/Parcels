"""Transparent rolling time-window cache for lazy (dask-backed) field data.

Assumptions / current limits:
  * ``time`` is the leading dimension of the field (true for both the SGRID and
    UGRID ingestion paths; the structured path transposes to ``(time, ...)``).
  * Valid while the requested time indices stay within the resident window
    (i.e. all particles share the clock). A sample that requests time indices
    spanning more than the retained levels would force reloads.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from dask import is_dask_collection

# xarray / uxarray ``isel`` keyword arguments that are NOT dimension indexers.
_NON_INDEXER_KWARGS = frozenset({"drop", "missing_dims", "ignore_grid"})


class WindowedArray:
    """Wrap a lazy DataArray so ``isel`` loads/caches/evicts time levels as NumPy."""

    def __init__(self, data: xr.DataArray, time_dim: str = "time", max_levels: int | None = None):
        if data.dims[0] != time_dim:
            raise ValueError(f"WindowedArray expects {time_dim!r} as the leading dimension, got {data.dims}")
        self._data = data
        self._tdim = time_dim
        self._cache: dict[int, np.ndarray] = {}  # time index -> NumPy slab (remaining dims)
        self._max = max_levels
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
            f"WindowedArray(time_dim={self._tdim!r}, cached_levels={sorted(self._cache)}, "
            f"loads={self.loads})\n{self._data!r}"
        )

    # -- window management ----------------------------------------------------
    def _read_level(self, lvl: int) -> np.ndarray:
        """Bulk, sequential read of one time level into NumPy (the dask->NumPy step)."""
        return np.asarray(self._data.isel({self._tdim: int(lvl)}).values)

    def _ensure(self, levels: np.ndarray) -> None:
        for lvl in levels:
            lvl = int(lvl)
            if lvl not in self._cache:
                self._cache[lvl] = self._read_level(lvl)
                self.loads += 1
                self.bytes_read += self._slab_bytes
        # retire stale levels (the clock only moves forward across the window)
        lo = int(np.min(levels))
        for old in [k for k in self._cache if k < lo]:
            del self._cache[old]
        if self._max is not None and len(self._cache) > self._max:
            for old in sorted(self._cache)[: len(self._cache) - self._max]:
                del self._cache[old]

    # -- intercepted indexing -------------------------------------------------
    def isel(self, indexers: dict | None = None, **kwargs):
        sel = dict(indexers) if indexers is not None else {}
        sel.update({k: v for k, v in kwargs.items() if k not in _NON_INDEXER_KWARGS})

        # no time selection -> nothing to window; preserve control kwargs
        if self._tdim not in sel:
            return self._data.isel(indexers, **kwargs)

        t_ind = sel[self._tdim]
        t_vals = np.asarray(t_ind.values if isinstance(t_ind, xr.DataArray) else t_ind)
        levels = np.unique(t_vals)
        self._ensure(levels)

        # stack the resident levels into one small NumPy block; remap to local indices
        block = np.stack([self._cache[int(lvl)] for lvl in levels])  # (nlevels, *rest)
        nda = xr.DataArray(block, dims=self._data.dims)  # NumPy-backed, original dim order
        local = np.searchsorted(levels, t_vals)
        sel[self._tdim] = xr.DataArray(local, dims=getattr(t_ind, "dims", ()))
        return nda.isel(sel)  # plain vectorised gather in NumPy (no ignore_grid needed)


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
    return data
