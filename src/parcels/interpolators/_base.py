from abc import ABC, abstractmethod
from typing import Any


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
