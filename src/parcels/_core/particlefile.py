"""Module controlling the writing of ParticleSets to Zarr file."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cftime
import numpy as np
import pyarrow as pa
from parcels._typing import PathLike
import pyarrow.parquet as pq

import parcels
from parcels._core.particle import ParticleClass
from parcels._core.utils.time import timedelta_to_float
from parcels._reprs import particlefile_repr

if TYPE_CHECKING:
    from parcels._core.particle import Variable
    from parcels._core.particleset import ParticleSet
    from parcels._core.utils.time import TimeInterval

__all__ = ["ParticleFile"]


def _get_schema(particle: parcels.ParticleClass, file_metadata: dict[Any, Any]) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                v.name,
                pa.from_numpy_dtype(v.dtype),
                metadata=v.attrs,
            )
            for v in _get_vars_to_write(particle)
        ],
        metadata=file_metadata.copy(),
    )


class ParticleFile:
    """Initialise trajectory output.

    Parameters
    ----------
    name : str
        Basename of the output file. This can also be a Zarr store object.
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a numpy.timedelta64, a datimetime.timedelta object or a positive float (in seconds).

    Returns
    -------
    ParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    def __init__(self, path: PathLike, outputdt):
        if not isinstance(outputdt, (np.timedelta64, timedelta, float)):
            raise ValueError(
                f"Expected outputdt to be a np.timedelta64, datetime.timedelta or float (in seconds), got {type(outputdt)}"
            )

        outputdt = timedelta_to_float(outputdt)
        path = Path(path)

        if path.suffix != ".parquet":
            raise ValueError(f"ParticleFile data is stored in Parquet files - file extension must be '.parquet'. Got {path.suffix=!r}.")

        if outputdt <= 0:
            raise ValueError(f"outputdt must be positive/non-zero. Got {outputdt=!r}")

        self._outputdt = outputdt

        self._path = path # TODO v4: Consider https://arrow.apache.org/docs/python/getstarted.html#working-with-large-data - though a significant question becomes how to partition, perhaps using a particle variable "partition"?
        self._writer: pq.ParquetWriter | None = None
        if path.exists():
            # TODO: Add logic for recovering/appending to existing parquet file
            raise ValueError(f"{path=!r} already exists. Either delete this file or use a path that doesn't exist.")
        if not path.parent.exists():
            raise ValueError(f"Folder location for {path=!r} does not exist. Create the folder location first.")

        self._maxids = 0
        self._pids_written = {}
        self.extra_metadata = {}

        # TODO v4: Add check that if create_new_zarrfile is False, the store already exists

    def __repr__(self) -> str:
        return particlefile_repr(self)

    def set_metadata(self, parcels_grid_mesh: Literal["spherical", "flat"]):
        self.extra_metadata.update(
            {
                "feature_type": "trajectory",
                "Conventions": "CF-1.6/CF-1.7",
                "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
                "parcels_version": parcels.__version__,
                "parcels_grid_mesh": parcels_grid_mesh,
            }
        )

    @property
    def outputdt(self):
        return self._outputdt

    @property
    def path(self):
        return self._path

    def write(self, pset: ParticleSet, time, indices=None):
        """Write all data from one time step to the zarr file,
        before the particle locations are updated.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet (same time object as fieldset)
        """
        pclass = pset._ptype
        time_interval = pset.fieldset.time_interval
        particle_data = pset._data

        if self._writer is None:
            assert not self.path.exists(), "If the file exists, the writer should already be set"
            self._writer = pq.ParquetWriter(self.path, _get_schema(pclass, self.extra_metadata))

        if isinstance(time, (np.timedelta64, np.datetime64)):
            time = timedelta_to_float(time - time_interval.left)
        vars_to_write = _get_vars_to_write(pclass)
        if indices is None:
            indices_to_write = _to_write_particles(particle_data, time)
        else:
            indices_to_write = indices
        
        self._writer.write_table(
            pa.table({v.name: pa.array(particle_data[v.name][indices_to_write]) for v in vars_to_write}),
        )

        # if len(indices_to_write) == 0: # TODO: Remove this?
        #     return

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _get_vars_to_write(particle: ParticleClass) -> list[Variable]:
    return [v for v in particle.variables if v.to_write is not False]


def _to_write_particles(particle_data, time):
    """Return the Particles that need to be written at time: if particle.time is between time-dt/2 and time+dt (/2)"""
    return np.where(
        (
            np.less_equal(
                time - np.abs(particle_data["dt"] / 2),
                particle_data["time"],
                where=np.isfinite(particle_data["time"]),
            )
            & np.greater_equal(
                time + np.abs(particle_data["dt"] / 2),
                particle_data["time"],
                where=np.isfinite(particle_data["time"]),
            )  # check time - dt/2 <= particle_data["time"] <= time + dt/2
            | (
                (np.isnan(particle_data["dt"]))
                & np.equal(time, particle_data["time"], where=np.isfinite(particle_data["time"]))
            )  # or dt is NaN and time matches particle_data["time"]
        )
        & (np.isfinite(particle_data["trajectory"]))
        & (np.isfinite(particle_data["time"]))
    )[0]


def _get_calendar_and_units(time_interval: TimeInterval) -> dict[str, str]:  # TODO: Remove?
    calendar = None
    units = "seconds"
    if time_interval:
        if isinstance(time_interval.left, (np.datetime64, datetime)):
            calendar = "standard"
        elif isinstance(time_interval.left, cftime.datetime):
            calendar = time_interval.left.calendar

    if calendar is not None:
        units += f" since {time_interval.left}"

    attrs = {"units": units}
    if calendar is not None:
        attrs["calendar"] = calendar

    return attrs
