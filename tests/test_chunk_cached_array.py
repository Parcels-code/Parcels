"""Tests for ChunkCachedArray vectorized indexing, including mixed slices."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import is_dask_collection
from dask.callbacks import Callback

from parcels._chunk_cached_array import ChunkCachedArray, wrap_dataset

# Dask drops a trailing "-<digits>" token when it names fused block tasks, so the
# callback key contains this whole string only when the name has no such suffix.
SOURCE_NAME = "parcelschunksource"
DIMS = ("time", "depth", "lat", "lon")
SHAPE = (5, 4, 7, 6)
CHUNKS = (2, 3, 3, 4)


def _points(*values: int, dims: str | tuple[str, ...] = "points") -> xr.DataArray:
    return xr.DataArray(np.array(values, dtype=np.int64), dims=dims)


def _dataset(
    shape: tuple[int, ...] = SHAPE,
    chunks: tuple[int, ...] = CHUNKS,
    dims: tuple[str, ...] = DIMS,
) -> tuple[xr.Dataset, xr.Dataset, ChunkCachedArray]:
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    darr = da.from_array(values, chunks=chunks, name=SOURCE_NAME)
    wrapped = wrap_dataset(xr.Dataset({"data": (dims, darr)}), max_cache_bytes=int(values.nbytes))
    reference = xr.Dataset({"data": (dims, values)})
    cached = wrapped["data"].variable._data
    assert isinstance(cached, ChunkCachedArray)
    return wrapped, reference, cached


def _assert_same_isel(wrapped: xr.Dataset, reference: xr.Dataset, indexers: dict) -> xr.DataArray:
    got = wrapped["data"].isel(indexers)
    ref = reference["data"].isel(indexers)
    assert tuple(got.dims) == tuple(ref.dims)
    assert not is_dask_collection(got.data)
    np.testing.assert_array_equal(np.asarray(got.data), np.asarray(ref.data))
    return got


class _ChunkComputeCounter(Callback):
    """Count dask tasks whose key names the source array."""

    def __init__(self, array_name: str) -> None:
        super().__init__()
        self.array_name = array_name
        self.all_keys: list[object] = []

    def _pretask(self, key, dsk, state) -> None:
        self.all_keys.append(key)

    @property
    def chunk_computes(self) -> int:
        return sum(self.array_name in (key if isinstance(key, str) else str(key)) for key in self.all_keys)


@contextmanager
def _counting() -> Iterator[_ChunkComputeCounter]:
    counter = _ChunkComputeCounter(SOURCE_NAME)
    with dask.config.set(scheduler="sync"), counter:
        yield counter


@pytest.mark.parametrize(
    "indexers",
    [
        pytest.param(
            {
                "time": slice(None),
                "depth": slice(1, 4),
                "lat": _points(0, 6, 3, 1),
                "lon": _points(0, 5, 2, 4),
            },
            id="slices-before-full-and-bounded",
        ),
        pytest.param(
            {
                "time": _points(0, 4, 2, 1),
                "depth": slice(0, 4, 2),
                "lat": slice(None, None, -1),
                "lon": _points(5, 0, 3, 1),
            },
            id="slices-between-stepped-and-reverse",
        ),
        pytest.param(
            {
                "time": _points(4, 0, -1, 2),
                "depth": _points(3, 0, 1, -4),
                "lat": slice(1, 7, 2),
                "lon": slice(None, None, -1),
            },
            id="slices-after-bounded-and-reverse",
        ),
    ],
)
def test_mixed_isel_slice_position_matches_numpy(indexers):
    """Full, bounded, stepped, and reverse slices keep xarray's axis order."""
    wrapped, reference, _ = _dataset()
    _assert_same_isel(wrapped, reference, indexers)


def test_isel_omits_nonsingleton_dimension():
    """An omitted axis longer than one is a full slice in vectorized order."""
    wrapped, reference, _ = _dataset()
    indexers = {
        "time": _points(0, 4, 2, -1),
        "depth": _points(0, 3, 1, 2),
        "lon": _points(5, 0, -2, 4),
    }
    got = _assert_same_isel(wrapped, reference, indexers)
    assert "lat" in got.dims
    assert got.sizes["lat"] == SHAPE[DIMS.index("lat")]


@pytest.mark.parametrize("explicit_full_slice", [False, True], ids=["omitted", "explicit-full-slice"])
def test_isel_singleton_dimension_matches_numpy(explicit_full_slice):
    """A size-1 axis omitted from isel is the reported mixed-slice key."""
    dims = ("time", "mockZ", "N", "M")
    wrapped, reference, _ = _dataset((4, 1, 5, 6), (2, 1, 2, 3), dims)
    indexers = {
        "time": _points(0, 3, -1, 1),
        "N": _points(0, 4, 2, -2),
        "M": _points(5, 0, 3, 1),
    }
    if explicit_full_slice:
        indexers["mockZ"] = slice(None)
    got = _assert_same_isel(wrapped, reference, indexers)
    assert got.sizes["mockZ"] == 1


def test_broadcast_multidimensional_indexers_with_leading_slice():
    """Size-1 index arrays broadcast, and a leading slice stays at the end."""
    wrapped, reference, _ = _dataset()
    indexers = {
        "time": slice(None, None, -1),
        "depth": xr.DataArray(np.array([[0, 3], [1, 2], [3, 0]], dtype=np.int64), dims=("i", "j")),
        "lat": xr.DataArray(np.array([0, 6], dtype=np.int64), dims="j"),
        "lon": xr.DataArray(np.array([[5, 1], [0, 4], [2, 3]], dtype=np.int64), dims=("i", "j")),
    }
    got = _assert_same_isel(wrapped, reference, indexers)
    assert tuple(got.dims) == ("time", "i", "j")
    assert got.sizes["time"] == SHAPE[0]


def test_all_array_vindex_preserves_duplicates_order_and_negative_indices():
    """Equal-length 1D keys keep point order, duplicates, and valid negatives."""
    wrapped, reference, _ = _dataset()
    indexers = {
        "time": _points(4, 0, -1, 2, 4, -5),
        "depth": _points(3, 0, -4, 1, 3, 2),
        "lat": _points(6, 1, 0, -3, 6, 2),
        "lon": _points(5, 0, -2, 4, 5, 1),
    }
    got = _assert_same_isel(wrapped, reference, indexers)
    assert tuple(got.dims) == ("points",)
    assert got.shape == (6,)


def test_mixed_vindex_duplicates_negative_indices_and_uneven_chunks():
    """Mixed keys select duplicate, unordered, and final short-chunk points."""
    wrapped, reference, _ = _dataset()
    indexers = {
        "time": _points(4, 0, -1, 2, 4),
        "depth": slice(None, None, -1),
        "lat": _points(6, 0, -7, 3, 1),
        "lon": slice(0, 6, 2),
    }
    _assert_same_isel(wrapped, reference, indexers)


@pytest.mark.parametrize(
    "indexers",
    [
        pytest.param(
            {
                "time": _points(),
                "depth": _points(),
                "lat": _points(),
                "lon": _points(),
            },
            id="empty-point-arrays",
        ),
        pytest.param(
            {
                "time": _points(),
                "depth": slice(None),
                "lat": slice(1, 4),
                "lon": _points(),
            },
            id="empty-points-with-slices",
        ),
        pytest.param(
            {
                "time": _points(0, 2, 4),
                "depth": slice(2, 2),
                "lat": slice(None),
                "lon": _points(1, 5, 0),
            },
            id="zero-length-slice",
        ),
        pytest.param(
            {
                "time": _points(),
                "depth": slice(1, 1),
                "lat": slice(0, 0),
                "lon": _points(),
            },
            id="empty-points-and-slices",
        ),
    ],
)
def test_empty_selections_match_numpy_without_loading_chunks(indexers):
    """Zero-length slices and empty point arrays do not fetch a chunk."""
    wrapped, reference, cached = _dataset()
    with _counting() as counter:
        got = _assert_same_isel(wrapped, reference, indexers)
    assert got.size == 0
    assert counter.chunk_computes == 0, counter.all_keys
    assert cached.cache.current_bytes == 0


@pytest.mark.parametrize(
    "indexers",
    [
        pytest.param(
            {
                "time": _points(0),
                "depth": _points(0),
                "lat": _points(0),
                "lon": _points(6),
            },
            id="all-array-too-large",
        ),
        pytest.param(
            {
                "time": _points(-6),
                "depth": _points(0),
                "lat": _points(0),
                "lon": _points(0),
            },
            id="all-array-too-negative",
        ),
        pytest.param(
            {
                "time": _points(5),
                "depth": slice(None),
                "lat": _points(0),
                "lon": _points(0),
            },
            id="mixed-too-large",
        ),
        pytest.param(
            {
                "time": _points(0),
                "depth": _points(-5),
                "lat": slice(0, 3),
                "lon": _points(1),
            },
            id="mixed-too-negative",
        ),
    ],
)
def test_out_of_bounds_indices_raise_before_chunk_lookup(indexers):
    """Indices outside ``[-size, size)`` raise IndexError and load nothing."""
    wrapped, _, cached = _dataset()
    with _counting() as counter, pytest.raises(IndexError, match="out of bounds"):
        wrapped["data"].isel(indexers)
    assert counter.chunk_computes == 0, counter.all_keys
    assert cached.cache.current_bytes == 0


def test_zero_slice_step_raises_without_loading_chunks():
    """A slice step of zero is invalid and must not fetch a chunk."""
    wrapped, _, cached = _dataset()
    indexers = {
        "time": _points(0, 1),
        "depth": slice(0, 4, 0),
        "lat": _points(0, 2),
        "lon": slice(None),
    }
    with _counting() as counter, pytest.raises(ValueError):
        wrapped["data"].isel(indexers)
    assert counter.chunk_computes == 0, counter.all_keys
    assert cached.cache.current_bytes == 0


def test_repeated_mixed_selection_loads_only_referenced_chunks():
    """The first mixed selection loads referenced chunks; the repeat loads none.

    ``depth`` chunks are ``(2, 2, 1)`` on a length-5 axis, so slice ``0:3``
    touches chunk 0 (indices 0, 1) and chunk 1 (index 2). Paired with time
    indices 0 and 2 (chunks 0 and 1) and lat indices inside chunk 0, the
    selection is exactly four chunks of the eighteen in the array.
    """
    shape = (6, 5, 4)
    chunks = (2, 2, 2)
    wrapped, reference, cached = _dataset(shape, chunks, ("time", "depth", "lat"))
    total_chunks = 3 * 3 * 2
    indexers = {
        "time": _points(0, 2),
        "depth": slice(0, 3),
        "lat": _points(0, 1),
    }
    with _counting() as first:
        _assert_same_isel(wrapped, reference, indexers)
    assert first.chunk_computes == 4, first.all_keys
    assert first.chunk_computes < total_chunks
    assert cached.cache.current_bytes > 0

    cached_bytes = cached.cache.current_bytes
    with _counting() as second:
        _assert_same_isel(wrapped, reference, indexers)
    assert second.chunk_computes == 0, second.all_keys
    assert cached.cache.current_bytes == cached_bytes
