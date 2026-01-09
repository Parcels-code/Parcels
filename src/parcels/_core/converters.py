from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

__all__ = [
    "UnitConverter",
    "Unity",
    "_convert_to_flat_array",
    "_unitconverters_map",
]


def _convert_to_flat_array(var: npt.ArrayLike) -> npt.NDArray:
    """Convert lists and single integers/floats to one-dimensional numpy arrays

    Parameters
    ----------
    var : Array
        list or numeric to convert to a one-dimensional numpy array
    """
    return np.array(var).flatten()


class UnitConverter(ABC):
    source_unit: str | None = None
    target_unit: str | None = None

    @abstractmethod
    def to_target(self, value, z, y, x): ...

    @abstractmethod
    def to_source(self, value, z, y, x): ...


class Unity(UnitConverter):
    """Interface class for spatial unit conversion during field sampling that performs no conversion."""

    source_unit: None
    target_unit: None

    def to_target(self, value, z=None, y=None, x=None):
        return value

    def to_source(self, value, z=None, y=None, x=None):
        return value


_unitconverters_map = {}
