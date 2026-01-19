"""Collection of pre-built interpolation kernels for unstructured grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from parcels._core.field import Field, VectorField
    from parcels._core.uxgrid import _UXGRID_AXES


def UxConstantFaceConstantZC(
    particle_positions: dict[str, float | np.ndarray],
    grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
    field: Field,
):
    """Piecewise constant interpolation kernel for face registered data that is vertically centered (on zc points)"""
    return field.data.values[
        grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
    ]


def UxConstantFaceLinearZF(
    particle_positions: dict[str, float | np.ndarray],
    grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
    field: Field,
):
    """
    Piecewise constant interpolation (lateral) with linear vertical interpolation kernel for face registered data
    that is located at vertical interface levels (on zf points)
    """
    ti = grid_positions["T"]["index"]
    zi, fi = grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
    z = particle_positions["z"]

    # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
    # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
    # First, do barycentric interpolation in the lateral direction for each interface level
    fzk = field.data.values[ti, zi, fi]
    fzkp1 = field.data.values[ti, zi + 1, fi]

    # Then, do piecewise linear interpolation in the vertical direction
    zk = field.grid.z.values[zi]
    zkp1 = field.grid.z.values[zi + 1]
    return (fzk * (zkp1 - z) + fzkp1 * (z - zk)) / (zkp1 - zk)  # Linear interpolation in the vertical direction


def UxLinearNodeConstantZC(
    particle_positions: dict[str, float | np.ndarray],
    grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
    field: Field,
):
    """
    Piecewise linear interpolation kernel for node registered data that is vertically centered (zc points).
    Effectively, it applies barycentric interpolation in the lateral direction
    and piecewise constant interpolation in the vertical direction.
    """
    ti = grid_positions["T"]["index"]
    zi, fi = grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
    bcoords = grid_positions["FACE"]["bcoord"]
    node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values
    return np.sum(
        field.data.values[ti[:, None], zi[:, None], node_ids] * bcoords, axis=-1
    )  # Linear interpolation in the vertical direction


def UxLinearNodeLinearZF(
    particle_positions: dict[str, float | np.ndarray],
    grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
    field: Field,
):
    """
    Piecewise linear interpolation kernel for node registered data located at vertical interface levels (zf points).
    Effectively, it applies barycentric interpolation in the lateral direction
    and piecewise linear interpolation in the vertical direction.
    """
    ti = grid_positions["T"]["index"]
    zi, fi = grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
    z = particle_positions["z"]
    bcoords = grid_positions["FACE"]["bcoord"]
    node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values
    # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
    # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
    # First, do barycentric interpolation in the lateral direction for each interface level
    fzk = np.sum(field.data.values[ti[:, None], zi[:, None], node_ids] * bcoords, axis=-1)
    fzkp1 = np.sum(field.data.values[ti[:, None], zi[:, None] + 1, node_ids] * bcoords, axis=-1)

    # Then, do piecewise linear interpolation in the vertical direction
    zk = field.grid.z.values[zi]
    zkp1 = field.grid.z.values[zi + 1]
    return (fzk * (zkp1 - z) + fzkp1 * (z - zk)) / (zkp1 - zk)  # Linear interpolation in the vertical direction


def Ux_Velocity(
    particle_positions: dict[str, float | np.ndarray],
    grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
    vectorfield: VectorField,
):
    """Interpolation kernel for Vectorfields of velocity on a UxGrid."""
    u = vectorfield.U._interp_method(particle_positions, grid_positions, vectorfield.U)
    v = vectorfield.V._interp_method(particle_positions, grid_positions, vectorfield.V)
    if vectorfield.grid._mesh == "spherical":
        u /= 1852 * 60 * np.cos(np.deg2rad(particle_positions["lat"]))
        v /= 1852 * 60

    if "3D" in vectorfield.vector_type:
        w = vectorfield.W._interp_method(particle_positions, grid_positions, vectorfield.W)
    else:
        w = 0.0
    return u, v, w
