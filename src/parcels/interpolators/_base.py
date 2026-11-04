from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ScalarInterpolator(ABC):
    @abstractmethod
    def interp(self, particle_positions, grid_positions, field) -> Any:  #! API a WIP
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(...)"


class VectorInterpolator(ABC):
    @abstractmethod
    def interp(self, particle_positions, grid_positions, vectorfield) -> Any:  #! API a WIP
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(...)"


class VerticalCoordinateInterpolator(ABC):
    """Used for models with time varying z."""

    @abstractmethod
    def columns(self, grid_positions, grid) -> np.ndarray:  #! API a WIP, returns an (npart, n_interfaces) array
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(...)"
