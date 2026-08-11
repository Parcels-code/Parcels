"""Provides Hypothesis strategies for generating Variable, ParticleClass, and related particle data."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st

from parcels._core.particle import Particle, Variable, get_default_particle

# Valid numpy dtypes for Variable
_VARIABLE_DTYPES = [np.float32, np.float64, np.int32, np.int64, np.bool_]

variable_dtype = st.sampled_from(_VARIABLE_DTYPES).map(np.dtype)

# Names used by the default Particle — generated variables must not collide with these
_DEFAULT_PARTICLE_NAMES = {var.name for var in Particle.variables}

# Python identifiers that are not keywords (required by _assert_str_and_python_varname)
variable_name = (
    st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)
    .filter(lambda s: s.isidentifier())
    .filter(lambda s: not __import__("keyword").iskeyword(s))
    .filter(lambda s: s not in _DEFAULT_PARTICLE_NAMES)
)


@st.composite
def variable(draw, name=None, dtype=None, to_write=None):
    """Strategy for generating Variable instances.

    Parameters
    ----------
    name : str, optional
        Fixed variable name. If None, generates a valid Python identifier.
    dtype : numpy.dtype, optional
        Fixed dtype. If None, draws from common numpy dtypes.
    to_write : bool, optional
        Fixed to_write value. If None, draws True or False.
    """
    if name is None:
        name = draw(variable_name)
    if dtype is None:
        dtype = draw(variable_dtype)
    if to_write is None:
        to_write = draw(st.booleans())

    if to_write:
        attrs = draw(
            st.just({})
            | st.dictionaries(
                keys=st.text(min_size=1, max_size=10, alphabet="abcdefghijklmnopqrstuvwxyz_"),
                values=st.text(min_size=1, max_size=20),
                max_size=3,
            )
        )
    else:
        attrs = {}

    return Variable(name=name, dtype=dtype, initial=0, to_write=to_write, attrs=attrs)


@st.composite
def particle_class(draw, min_vars=0, max_vars=5, spatial_dtype=None):
    """Strategy that extends the default Particle with additional variables.

    This mirrors the predominant use case in Parcels: starting from
    ``get_default_particle`` and adding custom variables via ``add_variable``.

    Parameters
    ----------
    min_vars : int
        Minimum number of extra variables to add.
    max_vars : int
        Maximum number of extra variables to add.
    spatial_dtype : type, optional
        np.float32 or np.float64 for the base particle. If None, draws one.
    """
    if spatial_dtype is None:
        spatial_dtype = draw(st.sampled_from([np.float32, np.float64]))

    base = get_default_particle(spatial_dtype)

    n = draw(st.integers(min_value=min_vars, max_value=max_vars))
    if n == 0:
        return base

    names = draw(st.lists(variable_name, min_size=n, max_size=n, unique=True))
    extra_vars = [draw(variable(name=name)) for name in names]
    return base.add_variable(extra_vars)
