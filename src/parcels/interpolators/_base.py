from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

import numpy as np

from parcels._repr_utils import interpolator_repr


class ScalarInterpolator(ABC):
    @abstractmethod
    def interp(self, particle_positions, grid_positions, field) -> Any: ...

    def __repr__(self):
        return interpolator_repr(self)


class VectorInterpolator(ABC):
    @abstractmethod
    def interp(self, particle_positions, grid_positions, vectorfield) -> Any: ...

    def __repr__(self):
        return interpolator_repr(self)


class VerticalCoordinateInterpolator(ABC):
    """Base class for the interpolators that build each particle's z column on a grid with a time-varying z.

    Unlike ScalarInterpolator and VectorInterpolator, which interpolate a field's values onto the particle positions,
    these interpolators are part of the particle search. They interpolate the grid's z coordinate horizontally and in
    time to give each particle a column of layer interface depths, which is then searched for the particle's vertical
    index. They are therefore set on the grid (``grid.z_interp_method``) rather than on a field.
    """

    @abstractmethod
    def columns(self, grid_positions, grid) -> np.ndarray:
        """Returns each particle's z column as an (npart, n_interfaces) array."""
        ...

    def _columns_from_scalar(self, interp: ScalarInterpolator, grid_positions, grid) -> np.ndarray:
        """Evaluates a scalar interpolator on the grid's time-varying depth at every vertical level, giving each
        particle's z column as an (npart, nlev) array. This is effectively only a horizontal and temporal interpolation, any
        input vertical position or barycentric coordinate is ignored.
        """
        nlev = grid.depth.shape[1]
        npart = next(np.shape(pos["index"])[0] for axis, pos in grid_positions.items() if axis != "T")

        def repeat(a):
            a = np.broadcast_to(a, (npart, *np.shape(a)[1:]))
            return np.repeat(a, nlev, axis=0)

        level_positions = {
            axis: {"index": repeat(pos["index"]), "bcoord": repeat(pos["bcoord"])}
            for axis, pos in grid_positions.items()
        }
        level_positions["Z"] = {"index": np.tile(np.arange(nlev), npart), "bcoord": np.zeros(npart * nlev)}

        # Scalar interpolators only use ``field.data`` and ``field.grid``, so a lightweight stand-in for the depth works
        depth_field = SimpleNamespace(data=grid.depth, grid=grid)
        return np.asarray(interp.interp({}, level_positions, depth_field)).reshape(npart, nlev)

    def __repr__(self):
        return interpolator_repr(self)
