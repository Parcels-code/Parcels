import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import pyarrow.parquet as pq
import xarray as xr

if TYPE_CHECKING:
    import io
    from pathlib import Path


def _decode_dict_to_utf8(d: dict[Any, Any]) -> dict[Any, Any]:
    ret = {}
    for key, item in d.items():
        if isinstance(key, bytes):
            key = key.decode("utf8")
        if isinstance(item, dict):
            item = _decode_dict_to_utf8(item)
        if isinstance(item, bytes):
            item = item.decode("utf8")
        ret[key] = item
    return ret


def particlefile_to_v3_zarr(from_parquet: str | Path | io.BytesIO, to_zarr: str | os.PathLike) -> None:
    """Convert a v4 particle file (parquet) to v3-style zarr output.

    Reads the parquet file, renames columns to v3 conventions
    (``particle_id`` -> ``trajectory``, ``t`` -> ``time``, ``x`` -> ``lon``,
    ``y`` -> ``lat``), and reshapes the data into a 2D
    ``(trajectory, obs)`` zarr store.

    Parameters
    ----------
    from_parquet : Path
        Path to the input parquet file.
    to_zarr : Path
        Path to the output zarr store. Must have a ``.zarr`` suffix.

    Raises
    ------
    ValueError
        If ``to_zarr`` does not have a ``.zarr`` suffix.

    Notes
    -----
    This is not a lazy operation — the entire parquet file is read into memory
    and pivoted before writing to zarr. For large particle files this may
    require significant memory. Performance improvements are welcome via PRs.
    """
    to_zarr = Path(to_zarr)
    if to_zarr.suffix != ".zarr":
        raise ValueError(f"Parameter `to_zarr` must have a '.zarr' suffix. Got {to_zarr=}.")
    df = pl.read_parquet(from_parquet)
    table = pq.read_table(from_parquet)

    # TODO: Check for available memory here and fail as a safeguard?

    # Rename columns to v3 conventions
    rename_map = {"particle_id": "trajectory", "t": "time", "x": "lon", "y": "lat"}
    try:
        df = df.rename(rename_map)
    except pl.exceptions.ColumnNotFoundError as e:
        e.add_note(
            f"Expected to have all columns {list(rename_map)} in the output parquet. Got columns {list(df.columns)}."
        )
        raise e

    metadata_per_field = {}
    for parquet_var in table.schema.names:
        zarr_var = rename_map.get(parquet_var, parquet_var)  # default to parquet name
        metadata_per_field[zarr_var] = table.field(parquet_var).metadata or {}

    # Group by trajectory, sort by time, and assign observation index
    df = df.sort("trajectory", "time")
    df = df.with_columns(
        pl.col("time").cum_count().over("trajectory").alias("obs") - 1,
    )

    # Pivot to (trajectory, obs) dimensions
    trajectories = df["trajectory"].unique().sort()
    data_vars = [c for c in df.columns if c not in ("trajectory", "obs")]

    ds_dict = {}
    for var in data_vars:
        pivoted = df.pivot(on="obs", index="trajectory", values=var, sort_columns=True)
        value_cols = [c for c in pivoted.columns if c != "trajectory"]
        ds_dict[var] = (
            ["trajectory", "obs"],
            pivoted.select(value_cols).to_numpy(),
            _decode_dict_to_utf8(metadata_per_field[var]),
        )

    ds = xr.Dataset(
        ds_dict,
        coords={"trajectory": trajectories.to_numpy()},
        attrs=_decode_dict_to_utf8(table.schema.metadata),
    )
    ds = ds.assign_coords({"obs": ds["obs"]})

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Consolidated metadata is currently not part in the Zarr format 3 specification."
        )
        ds.to_zarr(to_zarr)
