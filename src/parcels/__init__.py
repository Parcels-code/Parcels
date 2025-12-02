from importlib.metadata import version as _version

try:
    __version__ = _version("parcels")
except Exception:
    # Local copy or not installed with setuptools.
    __version__ = "unknown"

import warnings as _stdlib_warnings

from parcels._core.basegrid import BaseGrid
from parcels._core.converters import (
    Geographic,
    GeographicPolar,
    GeographicPolarSquare,
    GeographicSquare,
    Unity,
)
from parcels._core.field import Field, VectorField
from parcels._core.fieldset import FieldSet
from parcels._core.kernel import Kernel
from parcels._core.particle import (
    KernelParticle,  # ? remove?
    Particle,
    ParticleClass,
    Variable,
)
from parcels._core.particlefile import ParticleFile
from parcels._core.particleset import ParticleSet
from parcels._core.statuscodes import (
    AllParcelsErrorCodes,
    FieldInterpolationError,
    FieldOutOfBoundError,
    FieldSamplingError,
    KernelError,
    OutsideTimeInterval,
    StatusCode,
)
from parcels._core.uxgrid import UxGrid
from parcels._core.warnings import (
    FieldSetWarning,
    FileWarning,
    KernelWarning,
    ParticleSetWarning,
)
from parcels._core.xgrid import XGrid
from parcels._logger import logger
from parcels._tutorial import download_example_dataset, list_example_datasets

__all__ = [  # noqa: RUF022
    # Core classes
    "BaseGrid",
    "Field",
    "VectorField",
    "FieldSet",
    "Kernel",
    "Particle",
    "ParticleClass",
    "ParticleFile",
    "ParticleSet",
    "Variable",
    "XGrid",
    "UxGrid",
    # Converters
    "Geographic",
    "GeographicPolar",
    "GeographicPolarSquare",
    "GeographicSquare",
    "Unity",
    # Status codes and errors
    "AllParcelsErrorCodes",
    "FieldInterpolationError",
    "FieldOutOfBoundError",
    "FieldSamplingError",
    "KernelError",
    "OutsideTimeInterval",
    "StatusCode",
    # Warnings
    "FieldSetWarning",
    "FileWarning",
    "KernelWarning",
    "ParticleSetWarning",
    # Utilities
    "logger",
    "download_example_dataset",
    "list_example_datasets",
    # (marked for potential removal)
    "KernelParticle",
]

_stdlib_warnings.warn(
    "This is an alpha version of Parcels v4. The API is not stable and may change without deprecation warnings.",
    UserWarning,
    stacklevel=2,
)
