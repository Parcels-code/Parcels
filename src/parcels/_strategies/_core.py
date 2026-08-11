import hypothesis.strategies as st
import numpy as np
import pandas as pd

from parcels._core.particle import ParticleClass

from .particle import particle_class

__all__ = [
    "particlefile_output",
]


def _generate_dummy_data(particle: ParticleClass, nparticles=10, nobs=10) -> pd.DataFrame:
    """Build a pandera DataFrameSchema from a ParticleClass.

    Only variables with ``to_write=True`` are included in the schema.
    Each column is typed with the variable's numpy dtype and carries the
    variable's ``attrs`` as pandera column-level metadata.
    """
    columns = {}
    variables = {var.name: var for var in particle.variables if var.to_write}
    try:
        particle_id = variables["particle_id"]
        t = variables["t"]
    except KeyError as e:
        e.add_note("This function requires 'particle_id' and 't' to be set")

    nobs_total = nparticles * nobs
    columns = {}
    columns["particle_id"] = np.repeat(
        np.arange(0, nparticles, dtype=particle_id.dtype).reshape((-1, 1)),
        nobs,
        axis=1,
    ).flatten()
    columns["t"] = np.repeat(
        np.linspace(0, nparticles * 3, num=nparticles, dtype=t.dtype).reshape((-1, 1)),
        nobs,
        axis=1,
    ).flatten()

    data_vars = set(variables.keys()) - {"particle_id", "t"}

    for name in data_vars:
        var = variables[name]
        columns[name] = np.linspace(0, 10000, num=nobs_total, dtype=var.dtype)

    return pd.DataFrame(columns)


@st.composite
def particlefile_output(draw, nobs=None, nparticles=None) -> pd.DataFrame:
    # at the moment this doesn't include the metadata (due to poor support in 
    # polars/pandas)
    #
    # we could also explore whether this can include the metadata, and whether the
    # return type can be closer to Parquet output (e.g., a temporary file, or
    # a BytesIO object)
    particle = draw(particle_class())
    if nobs is None:
        nobs = draw(st.integers(min_value=5, max_value=100))
    if nparticles is None:
        nparticles = draw(st.integers(min_value=5, max_value=100))
    return _generate_dummy_data(particle, nparticles, nobs)
