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


class UxConstantFaceConstantZC(ScalarInterpolator):
    """Piecewise constant interpolation kernel for face registered data that is vertically centered (on zc points)"""

    def interp(
        self,
        particle_positions: dict[str, float | np.ndarray],
        grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]],
        field: Field,
    ):
        # Broadcast the per-axis indices to a common (npart,) shape (``ti`` may be scalar for time-constant fields)
        ti, tau, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"],
            grid_positions["T"]["bcoord"],
            grid_positions["Z"]["index"],
            grid_positions["FACE"]["index"],
        )

        tdim, zdim, fdim = field.data.dims

        if np.any(tau > 0):
            t_levels = np.stack([ti, np.clip(ti + 1, 0, field.data.sizes[tdim] - 1)])
            t_weights = np.stack([1 - tau, tau])
        else:
            t_levels, t_weights = ti[np.newaxis], np.ones((1, ti.size))
        selection_dict = {
            tdim: xr.DataArray(t_levels, dims=("tlevels", "points")),
            zdim: xr.DataArray(zi, dims="points"),
            fdim: xr.DataArray(fi, dims="points"),
        }
        face_data = field.data.isel(selection_dict, ignore_grid=True)

        # Linear interpolation in time between the snapshots ti and ti + 1
        value = (face_data * xr.DataArray(t_weights, dims=("tlevels", "points"))).sum("tlevels").data
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
        ti, tau, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"],
            grid_positions["T"]["bcoord"],
            grid_positions["Z"]["index"],
            grid_positions["FACE"]["index"],
        )
        zeta = grid_positions["Z"]["bcoord"]

        tdim, zdim, fdim = field.data.dims

        if np.any(tau > 0):
            t_levels = np.stack([ti, np.clip(ti + 1, 0, field.data.sizes[tdim] - 1)])
            t_weights = np.stack([1 - tau, tau])
        else:
            t_levels, t_weights = ti[np.newaxis], np.ones((1, ti.size))
        time_weights = xr.DataArray(t_weights, dims=("tlevels", "points"))

        def _z_and_t_sample(z_index):
            """Pointwise ``isel`` of the face values at a single vertical interface level, linear in time between the
            snapshots ti and ti + 1.
            """
            selection_dict = {
                tdim: xr.DataArray(t_levels, dims=("tlevels", "points")),
                zdim: xr.DataArray(z_index, dims="points"),
                fdim: xr.DataArray(fi, dims="points"),
            }
            face_data = field.data.isel(selection_dict, ignore_grid=True)
            value = (face_data * time_weights).sum("tlevels").data
            return value.compute() if is_dask_collection(value) else value

        # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
        # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
        # First, do barycentric interpolation in the lateral direction for each interface level
        fzk = _z_and_t_sample(zi)
        fzkp1 = _z_and_t_sample(zi + 1)

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
        bcoords = xr.DataArray(grid_positions["FACE"]["bcoord"], dims=("points", "nodes"))
        node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values

        tdim, zdim, ndim = field.data.dims

        if np.any(tau > 0):
            t_levels = np.stack([ti, np.clip(ti + 1, 0, field.data.sizes[tdim] - 1)])
            t_weights = np.stack([1 - tau, tau])
        else:
            t_levels, t_weights = ti[np.newaxis], np.ones((1, ti.size))
        selection_dict = {
            tdim: xr.DataArray(t_levels, dims=("tlevels", "points")),
            zdim: xr.DataArray(zi, dims="points"),
            ndim: xr.DataArray(node_ids, dims=("points", "nodes")),
        }

        node_data = field.data.isel(selection_dict, ignore_grid=True)
        # Barycentric interpolation in the lateral direction and linear interpolation in time
        weights = bcoords * xr.DataArray(t_weights, dims=("tlevels", "points"))
        value = (node_data * weights).sum(("nodes", "tlevels")).data
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
        bcoords = xr.DataArray(grid_positions["FACE"]["bcoord"], dims=("points", "nodes"))
        node_ids = field.grid.uxgrid.face_node_connectivity[fi, :].values

        tdim, zdim, ndim = field.data.dims

        if np.any(tau > 0):
            t_levels = np.stack([ti, np.clip(ti + 1, 0, field.data.sizes[tdim] - 1)])
            t_weights = np.stack([1 - tau, tau])
        else:
            t_levels, t_weights = ti[np.newaxis], np.ones((1, ti.size))
        # Barycentric weights in the lateral direction combined with linear weights in time
        weights = bcoords * xr.DataArray(t_weights, dims=("tlevels", "points"))

        def _z_and_t_sample(z_index):
            """Barycentric (lateral) interpolation of the node values at a single vertical interface level, linear in
            time between the snapshots ti and ti + 1.
            """
            selection_dict = {
                tdim: xr.DataArray(t_levels, dims=("tlevels", "points")),
                zdim: xr.DataArray(z_index, dims="points"),
                ndim: xr.DataArray(node_ids, dims=("points", "nodes")),
            }
            # Reduce over the "nodes" and "tlevels" dimensions by name so the result is independent of ``isel`` dim order.
            node_data = field.data.isel(selection_dict, ignore_grid=True)
            return (node_data * weights).sum(("nodes", "tlevels")).data

        # The zi refers to the vertical layer index. The field in this routine are assumed to be defined at the vertical interface levels.
        # For interface zi, the interface indices are [zi, zi+1], so we need to use the values at zi and zi+1.
        # First, do barycentric interpolation in the lateral direction for each interface level
        fzk = _z_and_t_sample(zi)
        fzkp1 = _z_and_t_sample(zi + 1)

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
        ti, tau, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"],
            grid_positions["T"]["bcoord"],
            grid_positions["Z"]["index"],
            grid_positions["FACE"]["index"],
        )
        edge_ids = field.grid.uxgrid.face_edge_connectivity[fi, :].values

        tdim, zdim, edim = field.data.dims

        if np.any(tau > 0):
            t_levels = np.stack([ti, np.clip(ti + 1, 0, field.data.sizes[tdim] - 1)])
            t_weights = np.stack([1 - tau, tau])
        else:
            t_levels, t_weights = ti[np.newaxis], np.ones((1, ti.size))
        selection_dict = {
            tdim: xr.DataArray(t_levels, dims=("tlevels", "points")),
            zdim: xr.DataArray(zi, dims="points"),
            edim: xr.DataArray(edge_ids, dims=("points", "edges")),
        }
        edge_data = field.data.isel(selection_dict, ignore_grid=True)
        face_data = edge_data.mean("edges")  # Face value is the mean of its surrounding edges

        # Linear interpolation in time between the snapshots ti and ti + 1
        value = (face_data * xr.DataArray(t_weights, dims=("tlevels", "points"))).sum("tlevels").data
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
        ti, tau, zi, fi = np.broadcast_arrays(
            grid_positions["T"]["index"],
            grid_positions["T"]["bcoord"],
            grid_positions["Z"]["index"],
            grid_positions["FACE"]["index"],
        )
        zeta = grid_positions["Z"]["bcoord"]
        edge_ids = field.grid.uxgrid.face_edge_connectivity[fi, :].values

        tdim, zdim, edim = field.data.dims

        if np.any(tau > 0):
            t_levels = np.stack([ti, np.clip(ti + 1, 0, field.data.sizes[tdim] - 1)])
            t_weights = np.stack([1 - tau, tau])
        else:
            t_levels, t_weights = ti[np.newaxis], np.ones((1, ti.size))
        time_weights = xr.DataArray(t_weights, dims=("tlevels", "points"))

        def _z_and_t_sample(z_index):
            """Mean of the edge values surrounding each face at a single vertical interface level, linear in time
            between the snapshots ti and ti + 1.
            """
            selection_dict = {
                tdim: xr.DataArray(t_levels, dims=("tlevels", "points")),
                zdim: xr.DataArray(z_index, dims="points"),
                edim: xr.DataArray(edge_ids, dims=("points", "edges")),
            }
            edge_data = field.data.isel(selection_dict, ignore_grid=True)
            return (edge_data.mean("edges") * time_weights).sum("tlevels").data

        # zi is the vertical layer index, bounded by the interface levels zi and zi + 1
        fzk = _z_and_t_sample(zi)
        fzkp1 = _z_and_t_sample(zi + 1)
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

    def columns(
        self, grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]], grid: UxGrid
    ) -> np.ndarray:
        return self._columns_from_scalar(UxConstantFaceConstantZC(), grid_positions, grid)


class UxVerticalCoordinateLinearNode(VerticalCoordinateInterpolator):
    """Barycentric from the face's nodes (node registered z), linear in time."""

    def columns(
        self, grid_positions: dict[_UXGRID_AXES, dict[str, int | float | np.ndarray]], grid: UxGrid
    ) -> np.ndarray:
        return self._columns_from_scalar(UxLinearNodeConstantZC(), grid_positions, grid)
