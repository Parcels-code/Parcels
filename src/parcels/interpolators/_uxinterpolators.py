"""Collection of pre-built interpolation kernels for unstructured grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from dask import is_dask_collection

from parcels.interpolators._base import ScalarInterpolator, VectorInterpolator, VerticalCoordinateInterpolator

if TYPE_CHECKING:
    from parcels._core.field import Field, VectorField
    from parcels._core.uxgrid import _UXGRID_AXES, UxGrid


def _barycentric_node_values(
    data: xr.DataArray,
    ti: np.ndarray,
    tau: np.ndarray,
    node_ids: np.ndarray,
    bcoords: np.ndarray,
) -> xr.DataArray:
    """Barycentric (lateral) interpolation of node registered data from the nodes of each particle's face, linear in
    time between the snapshots ti and ti + 1, at every vertical level. Returns a ("points", <vertical dim>) array."""
    tdim, _, ndim = data.dims
    face_nodes = xr.DataArray(node_ids, dims=("points", "nodes"))
    node_weights = xr.DataArray(bcoords, dims=("points", "nodes"))

    def _at_snapshot(t_index):
        # Reduce over the "nodes" dimension by name so the result is independent of ``isel`` dim order.
        node_data = data.isel({tdim: xr.DataArray(t_index, dims="points"), ndim: face_nodes}, ignore_grid=True)
        return (node_data * node_weights).sum("nodes")

    if not np.any(tau > 0):
        return _at_snapshot(ti)
    tau_points = xr.DataArray(tau, dims="points")
    return (1 - tau_points) * _at_snapshot(ti) + tau_points * _at_snapshot(np.clip(ti + 1, 0, data.sizes[tdim] - 1))


class UxConstantFaceConstantZC(ScalarInterpolator):
    """Piecewise constant interpolation kernel for face registered data that is vertically centered (on zc points)"""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        # Broadcast the per-axis indices to a common (npart,) shape (``ti`` may be scalar for time-constant fields)
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )

        tdim, zdim, fdim = field.data.dims
        selection_dict = {
            tdim: xr.DataArray(ti, dims="points"),
            zdim: xr.DataArray(zi, dims="points"),
            fdim: xr.DataArray(fi, dims="points"),
        }
        value = field.data.isel(selection_dict, ignore_grid=True).data
        return value.compute() if is_dask_collection(value) else value


class UxConstantFaceLinearZF(ScalarInterpolator):
    """
    Piecewise constant interpolation (lateral) with linear vertical interpolation kernel for face registered data
    that is located at vertical interface levels (on zf points)
    """

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )
        zeta = grid_positions["Z"]["bcoord"]

        tdim, zdim, fdim = field.data.dims

        def _zsample(z_index):
            """Pointwise ``isel`` of the face values at a single vertical interface level."""
            selection_dict = {
                tdim: xr.DataArray(ti, dims="points"),
                zdim: xr.DataArray(z_index, dims="points"),
                fdim: xr.DataArray(fi, dims="points"),
            }
            value = field.data.isel(selection_dict, ignore_grid=True).data
            return value.compute() if is_dask_collection(value) else value

        # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
        # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
        # First, do barycentric interpolation in the lateral direction for each interface level
        fzk = _zsample(zi)
        fzkp1 = _zsample(zi + 1)

        # Then, do piecewise linear interpolation in the vertical direction
        return fzk * (1 - zeta) + fzkp1 * zeta


class UxLinearNodeConstantZC(ScalarInterpolator):
    """Piecewise linear interpolation kernel for node registered data that is vertically centered (zc points)."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """
        Piecewise linear interpolation kernel for node registered data that is vertically centered (zc points).
        Effectively, it applies barycentric interpolation in the lateral direction
        and piecewise constant interpolation in the vertical direction.
        """
        ti, tau, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"],
            grid_positions["T"]["bcoord"],
            grid_positions["Z"]["index"],
            grid_positions["FACE"]["index"],
        )
        node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values
        columns = _barycentric_node_values(field.data, ti, tau, node_ids, grid_positions["FACE"]["bcoord"])

        zdim = field.data.dims[1]
        value = columns.isel({zdim: xr.DataArray(zi, dims="points")}).data
        return value.compute() if is_dask_collection(value) else value


class UxLinearNodeLinearZF(ScalarInterpolator):
    """Piecewise linear interpolation kernel for node registered data located at vertical interface levels (zf points)."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        """
        Piecewise linear interpolation kernel for node registered data located at vertical interface levels (zf points).
        Effectively, it applies barycentric interpolation in the lateral direction
        and piecewise linear interpolation in the vertical direction.
        """
        ti, tau, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"],
            grid_positions["T"]["bcoord"],
            grid_positions["Z"]["index"],
            grid_positions["FACE"]["index"],
        )
        zeta = grid_positions["Z"]["bcoord"]
        node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values
        columns = _barycentric_node_values(field.data, ti, tau, node_ids, grid_positions["FACE"]["bcoord"])

        # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
        # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
        zdim = field.data.dims[1]
        fzk = columns.isel({zdim: xr.DataArray(zi, dims="points")}).data
        fzkp1 = columns.isel({zdim: xr.DataArray(zi + 1, dims="points")}).data

        # Then, do piecewise linear interpolation in the vertical direction
        value = fzk * (1 - zeta) + fzkp1 * zeta
        return value.compute() if is_dask_collection(value) else value


class UxConstantEdgeConstantZC(ScalarInterpolator):
    """
    Piecewise constant interpolation kernel for edge registered data that is vertically centered (on zc points).
    Face values are taken to be the mean of the surrounding edges.
    """

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        # Broadcast the per-axis indices to a common (npart,) shape (``ti`` may be scalar for time-constant fields)
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )
        edge_ids = field.grid.uxgrid.face_edge_connectivity[fi, :].values

        tdim, zdim, edim = field.data.dims
        selection_dict = {
            tdim: xr.DataArray(ti, dims="points"),
            zdim: xr.DataArray(zi, dims="points"),
            edim: xr.DataArray(edge_ids, dims=("points", "edges")),
        }
        edge_data = field.data.isel(selection_dict, ignore_grid=True)
        value = edge_data.mean("edges").data  # Face value is the mean of its surrounding edges
        return value.compute() if is_dask_collection(value) else value


class UxConstantEdgeLinearZF(ScalarInterpolator):
    """
    Piecewise constant interpolation (lateral) with linear vertical interpolation kernel for edge registered data
    that is located at vertical interface levels (on zf points).
    Face values are taken to be the mean of the surrounding edges.
    """

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        # Broadcast the per-axis indices to a common (npart,) shape (``ti`` may be scalar for time-constant fields)
        ti, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"], grid_positions["Z"]["index"], grid_positions["FACE"]["index"]
        )
        zeta = grid_positions["Z"]["bcoord"]
        edge_ids = field.grid.uxgrid.face_edge_connectivity[fi, :].values

        tdim, zdim, edim = field.data.dims

        def _zsample(z_index):
            """Mean of the edge values surrounding each face at a single vertical interface level."""
            selection_dict = {
                tdim: xr.DataArray(ti, dims="points"),
                zdim: xr.DataArray(z_index, dims="points"),
                edim: xr.DataArray(edge_ids, dims=("points", "edges")),
            }
            edge_data = field.data.isel(selection_dict, ignore_grid=True)
            return edge_data.mean("edges").data

        # zi is the vertical layer index, bounded by the interface levels zi and zi + 1
        fzk = _zsample(zi)
        fzkp1 = _zsample(zi + 1)
        value = fzk * (1 - zeta) + fzkp1 * zeta
        return value.compute() if is_dask_collection(value) else value


class Ux_Velocity(VectorInterpolator):  # noqa: N801
    """Interpolation kernel for Vectorfields of velocity on a UxGrid."""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        vectorfield: VectorField,
    ):
        u = vectorfield.U.interp_method.interp(particle_positions, grid_positions, vectorfield.U)
        v = vectorfield.V.interp_method.interp(particle_positions, grid_positions, vectorfield.V)
        if vectorfield.grid.mesh.is_spherical():
            u /= vectorfield.grid.deg2m * np.cos(np.deg2rad(particle_positions["y"]))
            v /= vectorfield.grid.deg2m

        if "3D" in vectorfield.vector_type:
            w = vectorfield.W.interp_method.interp(particle_positions, grid_positions, vectorfield.W)
        else:
            w = np.zeros_like(u)
        return u, v, w


class UxVerticalCoordinateConstantFace(VerticalCoordinateInterpolator):
    """Piecewise constant over the face (face registered z), linear in time."""

    def columns(self, grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]], grid: UxGrid) -> np.ndarray:
        raise NotImplementedError


class UxVerticalCoordinateLinearNode(VerticalCoordinateInterpolator):
    """Barycentric from the face's nodes (node registered z), linear in time."""

    def columns(self, grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]], grid: UxGrid) -> np.ndarray:
        node_ids = grid.uxgrid.face_node_connectivity[grid_positions["FACE"]["index"], :].values
        columns = _barycentric_node_values(
            grid.z, grid_positions["T"]["index"], grid_positions["T"]["bcoord"], node_ids, grid_positions["FACE"]["bcoord"]
        )
        return columns.values
