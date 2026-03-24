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
from typing import Any, Literal, Protocol, Self, cast, overload

import xarray as xr

from parcels._python import repr_from_dunder_dict

RE_DIM_DIM_PADDING = r"(\w+):(\w+)\s*\(padding:\s*(\w+)\)"

Dim = str


class Padding(enum.Enum):
    NONE = "none"
    LOW = "low"
    HIGH = "high"
    BOTH = "both"


SGRID_PADDING_TO_XGCM_POSITION = {
    Padding.LOW: "right",
    Padding.HIGH: "left",
    Padding.BOTH: "inner",
    Padding.NONE: "outer",
    # "center" position is not used in SGrid, in SGrid this would just be the edges/faces themselves
}


class AttrsSerializable(Protocol):
    def to_attrs(self) -> dict[str, str | int]: ...

    @classmethod
    def from_attrs(cls, d: dict[str, Hashable]) -> Self: ...


# Note that - for some optional attributes in the SGRID spec - these IDs are not available
# hence this isn't full coverage
_ID_FETCHERS_GRID2DMETADATA = {
    "node_dimension1": lambda meta: meta.node_dimensions[0],
    "node_dimension2": lambda meta: meta.node_dimensions[1],
    "face_dimension1": lambda meta: meta.face_dimensions[0].dim1,
    "face_dimension2": lambda meta: meta.face_dimensions[1].dim1,
    "type1": lambda meta: meta.face_dimensions[0].padding,
    "type2": lambda meta: meta.face_dimensions[1].padding,
}

_ID_FETCHERS_GRID3DMETADATA = {
    "node_dimension1": lambda meta: meta.node_dimensions[0],
    "node_dimension2": lambda meta: meta.node_dimensions[1],
    "node_dimension3": lambda meta: meta.node_dimensions[2],
    "face_dimension1": lambda meta: meta.volume_dimensions[0].dim1,
    "face_dimension2": lambda meta: meta.volume_dimensions[1].dim1,
    "face_dimension3": lambda meta: meta.volume_dimensions[2].dim1,
    "type1": lambda meta: meta.volume_dimensions[0].padding,
    "type2": lambda meta: meta.volume_dimensions[1].padding,
    "type3": lambda meta: meta.volume_dimensions[2].padding,
}


class Grid2DMetadata(AttrsSerializable):
    def __init__(
        self,
        cf_role: Literal["grid_topology"],
        topology_dimension: Literal[2],
        node_dimensions: tuple[Dim, Dim],
        face_dimensions: tuple[DimDimPadding, DimDimPadding],
        node_coordinates: None | tuple[Dim, Dim] = None,
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

        if node_coordinates is not None:
            if not (
                isinstance(node_coordinates, tuple)
                and len(node_coordinates) == 2
                and all(isinstance(nd, str) for nd in node_coordinates)
            ):
                raise ValueError("node_coordinates must be a tuple of 2 dimensions for a 2D grid")

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

        # Optional attributes
        self.node_coordinates = node_coordinates
        self.vertical_dimensions = vertical_dimensions

        #! Some optional attributes aren't really important to Parcels, can be added later if needed
        # Optional attributes
        # # With defaults (set in init)
        # edge1_dimensions: tuple[Dim, DimDimPadding]
        # edge2_dimensions: tuple[DimDimPadding, Dim]

        # # Without defaults
        # edge1_coordinates: None | Any = None
        # edge2_coordinates: None | Any = None
        # face_coordinate: None | Any = None

    def __repr__(self) -> str:
        return repr_from_dunder_dict(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Grid2DMetadata):
            return NotImplemented
        return self.to_attrs() == other.to_attrs()

    @classmethod
    def from_attrs(cls, attrs):  # type: ignore[override]
        try:
            return cls(
                cf_role=attrs["cf_role"],
                topology_dimension=attrs["topology_dimension"],
                node_dimensions=cast(tuple[Dim, Dim], load_mappings(attrs["node_dimensions"])),
                face_dimensions=cast(tuple[DimDimPadding, DimDimPadding], load_mappings(attrs["face_dimensions"])),
                node_coordinates=maybe_load_mappings(attrs.get("node_coordinates")),
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
        if self.node_coordinates is not None:
            d["node_coordinates"] = dump_mappings(self.node_coordinates)
        if self.vertical_dimensions is not None:
            d["vertical_dimensions"] = dump_mappings(self.vertical_dimensions)
        return d

    def rename(self, names_dict: dict[str, str]) -> Self:
        return cast(Self, _metadata_rename(self, names_dict))

    def get_value_by_id(self, id: str) -> str:
        """In the SGRID specification for 2D grids, different parts of the spec are identified by different "ID"s.

        Easily extract the value for a given ID.

        Example
        -------
        # Get padding 2
        >>> get_name_from_id("type2")
        "low"
        """
        return _ID_FETCHERS_GRID2DMETADATA[id](self)


class Grid3DMetadata(AttrsSerializable):
    def __init__(
        self,
        cf_role: Literal["grid_topology"],
        topology_dimension: Literal[3],
        node_dimensions: tuple[Dim, Dim, Dim],
        volume_dimensions: tuple[DimDimPadding, DimDimPadding, DimDimPadding],
        node_coordinates: None | tuple[Dim, Dim, Dim] = None,
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

        if node_coordinates is not None:
            if not (
                isinstance(node_coordinates, tuple)
                and len(node_coordinates) == 3
                and all(isinstance(nd, str) for nd in node_coordinates)
            ):
                raise ValueError("node_coordinates must be a tuple of 3 dimensions for a 3D grid")

        # Required attributes
        self.cf_role = cf_role
        self.topology_dimension = topology_dimension
        self.node_dimensions = node_dimensions
        self.volume_dimensions = volume_dimensions

        # Optional attributes
        self.node_coordinates = node_coordinates

        # ! Some optional attributes aren't really important to Parcels, can be added later if needed
        # Optional attributes
        # # With defaults (set in init)
        # edge1_dimensions: tuple[DimDimPadding, Dim, Dim]
        # edge2_dimensions: tuple[Dim, DimDimPadding, Dim]
        # edge3_dimensions: tuple[Dim, Dim, DimDimPadding]
        # face1_dimensions: tuple[Dim, DimDimPadding, DimDimPadding]
        # face2_dimensions: tuple[DimDimPadding, Dim, DimDimPadding]
        # face3_dimensions: tuple[DimDimPadding, DimDimPadding, Dim]

        # # Without defaults
        # edge *i_coordinates*
        # face *i_coordinates*
        # volume_coordinates

    def __repr__(self) -> str:
        return repr_from_dunder_dict(self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Grid3DMetadata):
            return NotImplemented
        return self.to_attrs() == other.to_attrs()

    @classmethod
    def from_attrs(cls, attrs):  # type: ignore[override]
        try:
            return cls(
                cf_role=attrs["cf_role"],
                topology_dimension=attrs["topology_dimension"],
                node_dimensions=cast(tuple[Dim, Dim, Dim], load_mappings(attrs["node_dimensions"])),
                volume_dimensions=cast(
                    tuple[DimDimPadding, DimDimPadding, DimDimPadding], load_mappings(attrs["volume_dimensions"])
                ),
                node_coordinates=maybe_load_mappings(attrs.get("node_coordinates")),
            )
        except Exception as e:
            raise SGridParsingException(f"Failed to parse Grid3DMetadata from {attrs=!r}") from e

    def to_attrs(self) -> dict[str, str | int]:
        d = dict(
            cf_role=self.cf_role,
            topology_dimension=self.topology_dimension,
            node_dimensions=dump_mappings(self.node_dimensions),
            volume_dimensions=dump_mappings(self.volume_dimensions),
        )
        if self.node_coordinates is not None:
            d["node_coordinates"] = dump_mappings(self.node_coordinates)
        return d

    def rename(self, dims_dict: dict[str, str]) -> Self:
        return cast(Self, _metadata_rename(self, dims_dict))

    def get_value_by_id(self, id: str) -> str:
        """In the SGRID specification for 3D grids, different parts of the spec are identified by different "ID"s.

        Easily extract the value for a given ID.

        Example
        -------
        # Get padding 2
        >>> get_name_from_id("type2")
        "low"
        """
        return _ID_FETCHERS_GRID3DMETADATA[id](self)


@dataclass
class DimDimPadding:
    """A data class representing a dimension-dimension-padding triplet for SGrid metadata.

    This triplet can represent different relations depending on context within the standard
    For example - for "face_dimensions" this can show the relation between an edge (dim1) and a node
    (dim2).
    """

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

        assert s != s_new, f"SGrid parsing did not advance, stuck at {s!r}"

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

        coords = {}
        for pos, dim in [("center", dim_dim_padding.dim1), (xgcm_position, dim_dim_padding.dim2)]:
            # only include dimensions in dataset (ignore dimensions in metadata that may not exist - e.g., due to `.isel`)
            if dim in ds.dims:
                coords[pos] = dim
        xgcm_coords[axis] = coords

    return (ds, {"coords": xgcm_coords})


def rename(ds: xr.Dataset, name_dict: dict[str, str]) -> xr.Dataset:
    grid_da = get_grid_topology(ds)
    if grid_da is None:
        raise ValueError(
            "No variable found in dataset with 'cf_role' attribute set to 'grid_topology'. This doesn't look to be an SGrid dataset - please make your dataset conforms to SGrid conventions."
        )

    ds = ds.rename(name_dict)

    # Update the metadata
    grid = parse_grid_attrs(grid_da.attrs)
    ds[grid_da.name].attrs = grid.rename(name_dict).to_attrs()
    return ds


def get_unique_names(grid: Grid2DMetadata | Grid3DMetadata) -> set[str]:
    dims = set()
    dims.update(set(grid.node_dimensions))

    for key, value in grid.__dict__.items():
        if key in ("cf_role", "topology_dimension") or value is None:
            continue
        assert isinstance(value, tuple), (
            f"Expected sgrid metadata attribute to be represented as a tuple, got {value!r}. This is an internal error to Parcels - please post an issue if you encounter this."
        )
        for item in value:
            if isinstance(item, DimDimPadding):
                dims.add(item.dim1)
                dims.add(item.dim2)
            else:
                assert isinstance(item, str)
                dims.add(item)
    return dims


def to_ascii(grid: Grid2DMetadata | Grid3DMetadata) -> str:
    """Return an ASCII diagram of the staggered grid structure.

    Parameters
    ----------
    grid : Grid2DMetadata | Grid3DMetadata

    Returns
    -------
    str
    """
    if isinstance(grid, Grid2DMetadata):
        return _grid2d_to_ascii(grid)
    return _grid3d_to_ascii(grid)


def grid_plot(grid: Grid2DMetadata | Grid3DMetadata):
    """Return a matplotlib Figure showing the staggered grid point positions.

    For 2D grids, a single panel is produced.  For 3D grids, three panels
    show the XY, XZ, and YZ cross-sections.

    Requires ``matplotlib`` to be installed.

    Parameters
    ----------
    grid : Grid2DMetadata | Grid3DMetadata

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(grid, Grid2DMetadata):
        return _grid2d_plot(grid)
    return _grid3d_plot(grid)


def _grid2d_to_ascii(grid: Grid2DMetadata) -> str:
    fd = grid.face_dimensions
    nd = grid.node_dimensions
    lines = [
        "Grid2DMetadata",
        f"  X-axis:  face={fd[0].dim1!r}  node={nd[0]!r}  padding={fd[0].padding.value}",
        f"  Y-axis:  face={fd[1].dim1!r}  node={nd[1]!r}  padding={fd[1].padding.value}",
    ]
    if grid.vertical_dimensions:
        vd = grid.vertical_dimensions[0]
        lines.append(f"  Z-axis:  face={vd.dim1!r}  node={vd.dim2!r}  padding={vd.padding.value}")
    if grid.node_coordinates:
        lines.append(f"  Coordinates: {grid.node_coordinates[0]}, {grid.node_coordinates[1]}")
    if grid.vertical_dimensions:
        vd = grid.vertical_dimensions[0]

        def _z(base: str, sym: str) -> str:
            return base.ljust(28) + sym

        lines += [
            "",
            "  Staggered grid layout (symbolic 3x3 nodes):",
            "",
            _z("    ↑ Y", "↑ Z"),
            _z("    |", "|"),
            _z("    n --u-- n --u-- n", "w"),
            _z("    |       |       |", "|"),
            _z("    v   ·   v   ·   v", "·"),
            _z("    |       |       |", "|"),
            _z("    n --u-- n --u-- n", "w"),
            _z("    |       |       |", "|"),
            _z("    v   ·   v   ·   v", "·"),
            _z("    |       |       |", "|"),
            "    n --u-- n --u-- n --→ X w",
            "",
            f"  n = node  ({nd[0]}, {nd[1]})",
            f"  u = x-face  ({fd[0].dim1})",
            f"  v = y-face  ({fd[1].dim1})",
            f"  w = z-node  ({vd.dim2})",
            "  · = cell centre",
        ]
    else:
        lines += [
            "",
            "  Staggered grid layout (symbolic 3x3 nodes):",
            "",
            "    ↑ Y",
            "    |",
            "    n --u-- n --u-- n",
            "    |       |       |",
            "    v   ·   v   ·   v",
            "    |       |       |",
            "    n --u-- n --u-- n",
            "    |       |       |",
            "    v   ·   v   ·   v",
            "    |       |       |",
            "    n --u-- n --u-- n --→ X",
            "",
            f"  n = node  ({nd[0]}, {nd[1]})",
            f"  u = x-face  ({fd[0].dim1})",
            f"  v = y-face  ({fd[1].dim1})",
            "  · = cell centre",
        ]
    return "\n".join(lines)


def _grid3d_to_ascii(grid: Grid3DMetadata) -> str:
    vd = grid.volume_dimensions
    nd = grid.node_dimensions
    lines = [
        "Grid3DMetadata",
        f"  X-axis:  face={vd[0].dim1!r}  node={nd[0]!r}  padding={vd[0].padding.value}",
        f"  Y-axis:  face={vd[1].dim1!r}  node={nd[1]!r}  padding={vd[1].padding.value}",
        f"  Z-axis:  face={vd[2].dim1!r}  node={nd[2]!r}  padding={vd[2].padding.value}",
    ]
    if grid.node_coordinates:
        lines.append(f"  Coordinates: {', '.join(grid.node_coordinates)}")
    lines += [
        "",
        "  Staggered grid layout (XY cross-section; Z-faces not shown):",
        "",
        "    ↑ Y",
        "    |",
        "    n --u-- n --u-- n",
        "    |       |       |",
        "    v   ·   v   ·   v",
        "    |       |       |",
        "    n --u-- n --u-- n",
        "    |       |       |",
        "    v   ·   v   ·   v",
        "    |       |       |",
        "    n --u-- n --u-- n --→ X",
        "",
        f"  n = node  ({nd[0]}, {nd[1]}, {nd[2]})",
        f"  u = x-face  ({vd[0].dim1})",
        f"  v = y-face  ({vd[1].dim1})",
        f"  w = z-face  ({vd[2].dim1})  [not shown in cross-section]",
        "  · = cell centre",
    ]
    return "\n".join(lines)


def _face_positions(padding: Padding, n_nodes: int):
    import numpy as np

    if padding in (Padding.LOW, Padding.HIGH):
        return np.arange(n_nodes - 1) + 0.5
    elif padding == Padding.BOTH:
        return np.arange(1, n_nodes - 1, dtype=float)
    else:  # Padding.NONE / outer
        return np.arange(n_nodes, dtype=float)


def _grid2d_plot(grid: Grid2DMetadata):
    import matplotlib.pyplot as plt
    import numpy as np

    fd = grid.face_dimensions
    nd = grid.node_dimensions
    N = 4  # illustrative node count per axis

    node_x = np.arange(N, dtype=float)
    node_y = np.arange(N, dtype=float)
    xface_x = _face_positions(fd[0].padding, N)
    yface_y = _face_positions(fd[1].padding, N)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    for x in node_x:
        ax.axvline(x, color="lightgray", lw=0.7, zorder=0)
    for y in node_y:
        ax.axhline(y, color="lightgray", lw=0.7, zorder=0)

    nx, ny = np.meshgrid(node_x, node_y)
    ax.scatter(nx.ravel(), ny.ravel(), s=70, color="steelblue", zorder=3, label=f"node ({nd[0]}, {nd[1]})", marker="o")

    fxx, fxy = np.meshgrid(xface_x, node_y)
    ax.scatter(fxx.ravel(), fxy.ravel(), s=70, color="firebrick", zorder=3, label=f"x-face ({fd[0].dim1})", marker=">")

    fyx, fyy = np.meshgrid(node_x, yface_y)
    ax.scatter(
        fyx.ravel(), fyy.ravel(), s=70, color="forestgreen", zorder=3, label=f"y-face ({fd[1].dim1})", marker="^"
    )

    ax.set_xlabel(nd[0])
    ax.set_ylabel(nd[1])
    ax.set_title(
        f"Grid2DMetadata — staggered positions\nX padding: {fd[0].padding.value}  |  Y padding: {fd[1].padding.value}"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _grid3d_plot(grid: Grid3DMetadata):
    import matplotlib.pyplot as plt
    import numpy as np

    vd = grid.volume_dimensions
    nd = grid.node_dimensions
    N = 4

    node = [np.arange(N, dtype=float) for _ in range(3)]
    face = [_face_positions(vd[i].padding, N) for i in range(3)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        (
            "XY",
            axes[0],
            node[0],
            node[1],
            face[0],
            face[1],
            nd[0],
            nd[1],
            vd[0].dim1,
            vd[1].dim1,
            vd[0].padding,
            vd[1].padding,
        ),
        (
            "XZ",
            axes[1],
            node[0],
            node[2],
            face[0],
            face[2],
            nd[0],
            nd[2],
            vd[0].dim1,
            vd[2].dim1,
            vd[0].padding,
            vd[2].padding,
        ),
        (
            "YZ",
            axes[2],
            node[1],
            node[2],
            face[1],
            face[2],
            nd[1],
            nd[2],
            vd[1].dim1,
            vd[2].dim1,
            vd[1].padding,
            vd[2].padding,
        ),
    ]

    for label, ax, n_h, n_v, f_h, f_v, dim_h, dim_v, face_h, face_v, pad_h, pad_v in panels:
        for x in n_h:
            ax.axvline(x, color="lightgray", lw=0.7, zorder=0)
        for y in n_v:
            ax.axhline(y, color="lightgray", lw=0.7, zorder=0)

        nh, nv = np.meshgrid(n_h, n_v)
        ax.scatter(nh.ravel(), nv.ravel(), s=50, color="steelblue", zorder=3, marker="o", label="node")

        fhh, fhv = np.meshgrid(f_h, n_v)
        ax.scatter(fhh.ravel(), fhv.ravel(), s=50, color="firebrick", zorder=3, marker=">", label=f"{face_h}-face")

        fvh, fvv = np.meshgrid(n_h, f_v)
        ax.scatter(fvh.ravel(), fvv.ravel(), s=50, color="forestgreen", zorder=3, marker="^", label=f"{face_v}-face")

        ax.set_xlabel(dim_h)
        ax.set_ylabel(dim_v)
        ax.set_title(f"{label} cross-section\npad: {pad_h.value} / {pad_v.value}")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

    fig.suptitle("Grid3DMetadata — staggered positions", fontsize=12)
    fig.tight_layout()
    return fig


def _attach_sgrid_metadata(ds, grid: Grid2DMetadata | Grid3DMetadata):
    """Copies the dataset and attaches the SGRID metadata in 'grid' variable. Modifies 'conventions' attribute."""
    ds = ds.copy()
    ds["grid"] = (
        [],
        0,
        grid.to_attrs(),
    )
    ds.attrs["Conventions"] = "SGRID"
    return ds


@overload
def _metadata_rename(grid: Grid2DMetadata, names_dict: dict[str, str]) -> Grid2DMetadata: ...


@overload
def _metadata_rename(grid: Grid3DMetadata, names_dict: dict[str, str]) -> Grid3DMetadata: ...


def _metadata_rename(grid, names_dict):
    """
    Renames dimensions and coordinates in SGrid metadata.

    Similar in API to xr.Dataset.rename . Renames dimensions according to names_dict mapping
     of old dimension names to new dimension names.
    """
    names_dict = names_dict.copy()
    assert len(names_dict) == len(set(names_dict.values())), "names_dict contains duplicate target dimension names"

    existing_names = get_unique_names(grid)
    for name in names_dict.keys():
        if name not in existing_names:
            raise ValueError(f"Name {name!r} not found in names defined in SGrid metadata {existing_names!r}")

    for name in existing_names:
        if name not in names_dict:
            names_dict[name] = name  # identity mapping for names not being renamed

    kwargs = {}
    for key, value in grid.__dict__.items():
        if isinstance(value, tuple):
            new_value = []
            for item in value:
                if isinstance(item, DimDimPadding):
                    new_item = DimDimPadding(
                        dim1=names_dict[item.dim1],
                        dim2=names_dict[item.dim2],
                        padding=item.padding,
                    )
                    new_value.append(new_item)
                else:
                    assert isinstance(item, str)
                    new_value.append(names_dict[item])
            kwargs[key] = tuple(new_value)
            continue

        if key in ("cf_role", "topology_dimension") or value is None:
            kwargs[key] = value
            continue

        if isinstance(value, str):
            kwargs[key] = names_dict[value]
            continue

        raise ValueError(f"Unexpected attribute {key!r} on {grid!r}")
    return type(grid)(**kwargs)
