"""
Utils and helpers specific for working with SGrid conventions.

https://sgrid.github.io/sgrid/
"""

import enum
import re

import xarray as xr

RE_NODE_FACE_PADDING = r"(\w+):(\w+)\s*\(padding:\s*(\w+)\)"


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


class SGridParsingException(Exception):
    """Exception raised when parsing SGrid attributes fails."""

    pass


def assert_2d_sgrid_metadata_present(attrs: dict[str, str | int]) -> None:
    for key in [
        "cf_role",
        "topology_dimension",
        "node_dimensions",
        "face_dimensions",
    ]:
        assert key in attrs, f"Missing required SGrid attribute: {key}"
    assert attrs["cf_role"] == "grid_topology", "Invalid cf_role for SGrid dataset"
    assert attrs["topology_dimension"] == 2, "Is not a 2D SGrid dataset"


def assert_3d_sgrid_metadata_present(attrs: dict[str, str | int]) -> None:
    for key in [
        "cf_role",
        "topology_dimension",
        "node_dimensions",
        "volume_dimensions",
    ]:
        assert key in attrs, f"Missing required SGrid attribute: {key}"
    assert attrs["cf_role"] == "grid_topology", "Invalid cf_role for SGrid dataset"
    assert attrs["topology_dimension"] == 3, "Is not a 3D SGrid dataset"
    assert "vertical_dimension" not in attrs, "Cannot have vertical_dimension for 3D SGrid dataset"


def parse_edge_node_mapping(s: str) -> list[tuple[str, str, Padding]]:
    """Takes in a string indicating the mappings of nodes faces as well as the padding,
     and returns a list with this data destructured.

    Treats `:` and `: ` equivalently (in line with the convention).
    """
    s = s.replace(": ", ":")
    matches = re.finditer(RE_NODE_FACE_PADDING, s)
    result = []
    for match in matches:
        edge = match.group(1)
        node = match.group(2)
        padding = Padding(match.group(3).lower())
        result.append((edge, node, padding))
    return result


def serialize_edge_node_mapping(mappings: list[tuple[str, str, Padding]]) -> str:
    """Takes in a list of edge-node-padding tuples and serializes them into a string
    according to the SGrid convention.
    """
    parts = []
    for edge, node, padding in mappings:
        parts.append(f"{edge}: {node} (padding: {padding.value})")
    return " ".join(parts)


def parse_grid(attrs: dict[str, str | int]):
    match attrs["topology_dimension"]:
        case 2:
            assert_2d_sgrid_metadata_present(attrs)
        case 3:
            assert_3d_sgrid_metadata_present(attrs)
        case _:
            raise SGridParsingException(f"Unsupported topology_dimension: {attrs['topology_dimension']}")

    assert isinstance(attrs["node_dimensions"], str)
    nodes = attrs["node_dimensions"].split()
    assert len(nodes) == attrs["topology_dimension"], (
        f"Number of nodes must match topology_dimension. len(nodes) != topology_dimension. {len(nodes)} != {attrs['topology_dimension']}"
    )

    faces_or_volumes = parse_edge_node_mapping(
        attrs["face_dimensions"] if attrs["topology_dimension"] == 2 else attrs["volume_dimensions"]
    )
    for _, node, _ in faces_or_volumes:
        assert node in nodes, f"Face/volume node {node!r} not found in nodes {nodes!r}"

    if "vertical_dimension" in attrs:
        faces_or_volumes.extend(parse_edge_node_mapping(attrs["vertical_dimension"]))
        nodes.append(faces_or_volumes[-1][1])  # Add vertical node to nodes list

    return nodes, faces_or_volumes


def get_grid_topology(ds: xr.Dataset) -> xr.DataArray:
    """Extracts grid topology DataArray from an xarray Dataset."""
    for var_name in ds.variables:
        if ds[var_name].attrs.get("cf_role") == "grid_topology":
            return ds[var_name]
    return


def parse_sgrid(ds: xr.Dataset):
    # Function similar to that provided in `xgcm.metadata_parsers.
    # Might at some point be upstreamed to xgcm directly
    try:
        grid_topology = get_grid_topology(ds)
        assert grid_topology is not None, "No grid_topology variable found in dataset"
        nodes, edges = parse_grid(grid_topology.attrs)
    except Exception as e:
        raise SGridParsingException(f"Error parsing {grid_topology=!r}") from e

    xgcm_coords = {}
    for (edge, node, padding), axis in zip(edges, "XYZ", strict=False):
        xgcm_position = SGRID_PADDING_TO_XGCM_POSITION[padding]
        xgcm_coords[axis] = {"center": edge, xgcm_position: node}

    return (ds, {"coords": xgcm_coords})


# @pytest.mark.parametrize(
#     "input_, expected",
#     [
#         (
#             "face1:node1(padding: none)",
#             [("face1", "node1", Padding.NONE)],
#         ),
#         (
#             "face1: node1 (padding: low), face2: node2 (padding: high), face3: node3 (padding: both)",
#             [
#                 ("face1", "node1", Padding.LOW),
#                 ("face2", "node2", Padding.HIGH),
#                 ("face3", "node3", Padding.BOTH),
#             ],
#         ),
#     ],
# )
# def test_parse_face_node_mapping(input_, expected):
#     assert parse_face_node_mapping(input_) == expected


# See how we can now integrate this into all the `from_...` methods in the codebase

# Optional:
# Add Hypothesis tests here?
