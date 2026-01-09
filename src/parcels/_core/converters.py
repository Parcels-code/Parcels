from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "_convert_to_flat_array",
]


def _convert_to_flat_array(var: npt.ArrayLike) -> npt.NDArray:
    """Convert lists and single integers/floats to one-dimensional numpy arrays

    Parameters
    ----------
    var : Array
        list or numeric to convert to a one-dimensional numpy array
    """
    return np.array(var).flatten()
