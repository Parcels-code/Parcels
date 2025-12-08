"""
Provides helpers and utils for working with SGrid conventions, as well as data objects
useful for representing the SGRID metadata model in code.

This code is best read alongside the SGrid conventions documentation:
https://sgrid.github.io/sgrid/

Note this code doesn't aim to completely cover the SGrid conventions, but aim to
cover SGrid to the extent to which Parcels is concerned.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, overload

import xarray as xr

RE_DIM_DIM_PADDING = r"(\w+):(\w+)\s*\(padding:\s*(\w+)\)"

Dim = str


class Padding(enum.Enum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"
    BOTH = "both"


class SGridMetadataProtocol(Protocol):
    def to_attrs(self) -> dict[str, str | int]: ...
    def from_attrs(cls, d: dict[str, Hashable]) -> Self: ...


class Grid2DMetadata(SGridMetadataProtocol):
    def __init__(
        self,
        cf_role: Literal["grid_topology"],
        topology_dimension: Literal[2],
        node_dimensions: tuple[Dim, Dim],
        face_dimensions: tuple[DimDimPadding, DimDimPadding],
        vertical_dimensions: None | tuple[DimDimPadding] = None,
    ):
        if cf_role != "grid_topology":
            raise ValueError(f"cf_role must be 'grid_topology', got {cf_role!r}")

        if topology_dimension != 2:
            raise ValueError("topology_dimension must be 2 for a 2D grid")

        if not (
            isinstance(node_dimensions, tuple)
            and len(node_dimensions) == 2
            and all(isinstance(nd, str) for nd in node_dimensions)
        ):
            raise ValueError("node_dimensions must be a tuple of 2 dimensions for a 2D grid")

        if not (
            isinstance(face_dimensions, tuple)
            and len(face_dimensions) == 2
            and all(isinstance(fd, DimDimPadding) for fd in face_dimensions)
        ):
            raise ValueError("face_dimensions must be a tuple of 2 DimDimPadding for a 2D grid")

        if vertical_dimensions is not None:
            if not (
                isinstance(vertical_dimensions, tuple)
                and len(vertical_dimensions) == 1
                and isinstance(vertical_dimensions[0], DimDimPadding)
            ):
                raise ValueError("vertical_dimensions must be a tuple of 1 DimDimPadding for a 2D grid")

        # Required attributes
        self.cf_role = cf_role
        self.topology_dimension = topology_dimension
        self.node_dimensions = node_dimensions
        self.face_dimensions = face_dimensions

        #! Optional attributes aren't really important to Parcels, can be added later if needed
        # Optional attributes
        # # With defaults (set in init)
        # edge1_dimensions: tuple[Dim, DimDimPadding]
        # edge2_dimensions: tuple[DimDimPadding, Dim]

        # # Without defaults
        # node_coordinates: None | Any = None
        # edge1_coordinates: None | Any = None
        # edge2_coordinates: None | Any = None
        # face_coordinate: None | Any = None

        #! Important optional attribute for 2D grids with vertical layering
        self.vertical_dimensions = vertical_dimensions

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Grid2DMetadata):
            return NotImplemented
        return (
            self.cf_role == other.cf_role
            and self.topology_dimension == other.topology_dimension
            and self.node_dimensions == other.node_dimensions
            and self.face_dimensions == other.face_dimensions
            and self.vertical_dimensions == other.vertical_dimensions
        )

    @classmethod
    def from_attrs(cls, attrs):
        try:
            return cls(
                cf_role=attrs["cf_role"],
                topology_dimension=attrs["topology_dimension"],
                node_dimensions=load_mappings(attrs["node_dimensions"]),
                face_dimensions=load_mappings(attrs["face_dimensions"]),
                vertical_dimensions=maybe_load_mappings(attrs.get("vertical_dimensions")),
            )
        except Exception as e:
            raise SGridParsingException(f"Failed to parse Grid2DMetadata from {attrs=!r}") from e

    def to_attrs(self) -> dict[str, str | int]:
        d = dict(
            cf_role=self.cf_role,
            topology_dimension=self.topology_dimension,
            node_dimensions=dump_mappings(self.node_dimensions),
            face_dimensions=dump_mappings(self.face_dimensions),
        )
        if self.vertical_dimensions is not None:
            d["vertical_dimensions"] = dump_mappings(self.vertical_dimensions)
        return d


class Grid3DMetadata(SGridMetadataProtocol):
    def __init__(
        self,
        cf_role: Literal["grid_topology"],
        topology_dimension: Literal[3],
        node_dimensions: tuple[Dim, Dim, Dim],
        volume_dimensions: tuple[DimDimPadding, DimDimPadding, DimDimPadding],
    ):
        if cf_role != "grid_topology":
            raise ValueError(f"cf_role must be 'grid_topology', got {cf_role!r}")

        if topology_dimension != 3:
            raise ValueError("topology_dimension must be 3 for a 3D grid")

        if not (
            isinstance(node_dimensions, tuple)
            and len(node_dimensions) == 3
            and all(isinstance(nd, str) for nd in node_dimensions)
        ):
            raise ValueError("node_dimensions must be a tuple of 3 dimensions for a 3D grid")

        if not (
            isinstance(volume_dimensions, tuple)
            and len(volume_dimensions) == 3
            and all(isinstance(fd, DimDimPadding) for fd in volume_dimensions)
        ):
            raise ValueError("face_dimensions must be a tuple of 2 DimDimPadding for a 2D grid")

        # Required attributes
        self.cf_role = cf_role
        self.topology_dimension = topology_dimension
        self.node_dimensions = node_dimensions
        self.volume_dimensions = volume_dimensions

        # ! Optional attributes aren't really important to Parcels, can be added later if needed
        # Optional attributes
        # # With defaults (set in init)
        # edge1_dimensions: tuple[DimDimPadding, Dim, Dim]
        # edge2_dimensions: tuple[Dim, DimDimPadding, Dim]
        # edge3_dimensions: tuple[Dim, Dim, DimDimPadding]
        # face1_dimensions: tuple[Dim, DimDimPadding, DimDimPadding]
        # face2_dimensions: tuple[DimDimPadding, Dim, DimDimPadding]
        # face3_dimensions: tuple[DimDimPadding, DimDimPadding, Dim]

        # # Without defaults
        # node_coordinates
        # edge *i_coordinates*
        # face *i_coordinates*
        # volume_coordinates

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Grid3DMetadata):
            return NotImplemented
        return (
            self.cf_role == other.cf_role
            and self.topology_dimension == other.topology_dimension
            and self.node_dimensions == other.node_dimensions
            and self.volume_dimensions == other.volume_dimensions
        )

    @classmethod
    def from_attrs(cls, attrs):
        try:
            return cls(
                cf_role=attrs["cf_role"],
                topology_dimension=attrs["topology_dimension"],
                node_dimensions=load_mappings(attrs["node_dimensions"]),
                volume_dimensions=load_mappings(attrs["volume_dimensions"]),
            )
        except Exception as e:
            raise SGridParsingException(f"Failed to parse Grid3DMetadata from {attrs=!r}") from e

    def to_attrs(self) -> dict[str, str | int]:
        return dict(
            cf_role=self.cf_role,
            topology_dimension=self.topology_dimension,
            node_dimensions=dump_mappings(self.node_dimensions),
            volume_dimensions=dump_mappings(self.volume_dimensions),
        )


@dataclass
class DimDimPadding:
    dim1: str
    dim2: str
    padding: Padding

    def __repr__(self) -> str:
        return f"DimDimPadding(dim1={self.dim1!r}, dim2={self.dim2!r}, padding={self.padding!r})"

    def __str__(self) -> str:
        return f"{self.dim1}:{self.dim2} (padding:{self.padding.value})"

    @classmethod
    def load(cls, s: str) -> Self:
        match = re.match(RE_DIM_DIM_PADDING, s)
        if not match:
            raise ValueError(f"String {s!r} does not match expected format for DimDimPadding")
        dim1 = match.group(1)
        dim2 = match.group(2)
        padding = Padding(match.group(3).lower())
        return cls(dim1, dim2, padding)


def dump_mappings(parts: Iterable[DimDimPadding | Dim]) -> str:
    """Takes in a list of edge-node-padding tuples and serializes them into a string
    according to the SGrid convention.
    """
    ret = []
    for part in parts:
        ret.append(str(part))
    return " ".join(ret)


@overload
def maybe_dump_mappings(parts: None) -> None: ...
@overload
def maybe_dump_mappings(parts: Iterable[DimDimPadding | Dim]) -> str: ...


def maybe_dump_mappings(parts):
    if parts is None:
        return None
    return dump_mappings(parts)


def load_mappings(s: str) -> tuple[DimDimPadding | Dim, ...]:
    """Takes in a string indicating the mappings of dims and dim-dim-padding
    and returns a tuple with this data destructured.

    Treats `:` and `: ` equivalently (in line with the convention).
    """
    if not isinstance(s, str):
        raise ValueError(f"Expected string input, got {s!r} of type {type(s)}")

    s = s.replace(": ", ":")
    ret = []
    while s:
        # find next part
        match = re.match(RE_DIM_DIM_PADDING, s)
        if match and match.start() == 0:
            # match found at start, take that as next part
            part = match.group(0)
            s_new = s[match.end() :].lstrip()
        else:
            # no DimDimPadding match at start, assume just a Dim until next space
            part, *s_new = s.split(" ", 1)
            s_new = "".join(s_new)

        assert s != s_new, f"Parsing did not advance, stuck at {s!r}"

        parsed: DimDimPadding | Dim
        try:
            parsed = DimDimPadding.load(part)
        except ValueError as e:
            e.add_note(f"Failed to parse part {part!r} from {s!r} as a dimension dimension padding string")
            try:
                # Not a DimDimPadding, assume it's just a Dim
                assert ":" not in part, f"Part {part!r} from {s!r} not a valid dim (contains ':')"
                parsed = part
            except AssertionError as e2:
                raise e2 from e

        ret.append(parsed)
        s = s_new

    return tuple(ret)


@overload
def maybe_load_mappings(s: None) -> None: ...
@overload
def maybe_load_mappings(s: Hashable) -> tuple[DimDimPadding | Dim, ...]: ...


def maybe_load_mappings(s):
    if s is None:
        return None
    return load_mappings(s)


SGRID_PADDING_TO_XGCM_POSITION = {
    Padding.LOW: "right",
    Padding.HIGH: "left",
    Padding.BOTH: "inner",
    Padding.NONE: "outer",
    # "center" position is not used in SGrid, in SGrid this would just be the edges/faces themselves
}


class SGridParsingException(Exception):
    """Exception raised when parsing SGrid attributes fails."""

    pass


def parse_grid_attrs(attrs: dict[str, Hashable]) -> Grid2DMetadata | Grid3DMetadata:
    grid: Grid2DMetadata | Grid3DMetadata
    try:
        grid = Grid2DMetadata.from_attrs(attrs)
    except Exception as e:
        e.add_note("Failed to parse as 2D SGrid, trying 3D SGrid")
        try:
            grid = Grid3DMetadata.from_attrs(attrs)
        except Exception as e2:
            e2.add_note("Failed to parse as 3D SGrid")
            raise SGridParsingException("Failed to parse SGrid metadata as either 2D or 3D grid") from e2
    return grid


def get_grid_topology(ds: xr.Dataset) -> xr.DataArray | None:
    """Extracts grid topology DataArray from an xarray Dataset."""
    for var_name in ds.variables:
        if ds[var_name].attrs.get("cf_role") == "grid_topology":
            return ds[var_name]
    return None


def parse_sgrid(ds: xr.Dataset):
    # Function similar to that provided in `xgcm.metadata_parsers.
    # Might at some point be upstreamed to xgcm directly
    try:
        grid_topology = get_grid_topology(ds)
        assert grid_topology is not None, "No grid_topology variable found in dataset"
        grid = parse_grid_attrs(grid_topology.attrs)

    except Exception as e:
        raise SGridParsingException(f"Error parsing {grid_topology=!r}") from e

    if isinstance(grid, Grid2DMetadata):
        dimensions = grid.face_dimensions + (grid.vertical_dimensions or ())
    else:
        assert isinstance(grid, Grid3DMetadata)
        dimensions = grid.volume_dimensions

    xgcm_coords = {}
    for dim_dim_padding, axis in zip(dimensions, "XYZ", strict=False):
        xgcm_position = SGRID_PADDING_TO_XGCM_POSITION[dim_dim_padding.padding]
        xgcm_coords[axis] = {"center": dim_dim_padding.dim2, xgcm_position: dim_dim_padding.dim1}

    return (ds, {"coords": xgcm_coords})
