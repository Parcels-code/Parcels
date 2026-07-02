from abc import ABC, abstractmethod
from typing import Any


class ScalarInterpolator(ABC):
    @abstractmethod
    def interp(self, particle_positions, grid_positions, field) -> Any:  #! API a WIP
        ...


class VectorInterpolator(ABC):
    @abstractmethod
    def interp(self, particle_positions, grid_positions, vectorfield) -> Any:  #! API a WIP
        ...
