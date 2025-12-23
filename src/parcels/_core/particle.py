from __future__ import annotations

import operator
from typing import Literal

import numpy as np

from parcels._compat import _attrgetter_helper
from parcels._core.statuscodes import StatusCode
from parcels._core.utils.string import _assert_str_and_python_varname
from parcels._core.utils.time import TimeInterval
from parcels._reprs import _format_list_items_multiline

__all__ = ["Particle", "ParticleClass", "ParticleSetView", "Variable"]
_TO_WRITE_OPTIONS = [True, False, "once"]


class Variable:
    """Descriptor class that delegates data access to particle data.

    Parameters
    ----------
    name : str
        Variable name as used within kernels
    dtype :
        Data type (numpy.dtype) of the variable
    initial :
        Initial value of the variable. Note that this can also be a Field object,
        which will then be sampled at the location of the particle
    to_write : bool, 'once', optional
        Boolean or 'once'. Controls whether Variable is written to NetCDF file.
        If to_write = 'once', the variable will be written as a time-independent 1D array
    attrs : dict, optional
        Attributes to be stored with the variable when written to file. This can include metadata such as units, long_name, etc.
    """

    def __init__(
        self,
        name,
        dtype: np.dtype = np.float32,
        initial=0,
        to_write: bool | Literal["once"] = True,
        attrs: dict | None = None,
    ):
        _assert_str_and_python_varname(name)

        try:
            dtype = np.dtype(dtype)
        except (TypeError, ValueError) as e:
            raise TypeError(f"Variable dtype must be a valid numpy dtype. Got {dtype=!r}") from e

        if to_write not in _TO_WRITE_OPTIONS:
            raise ValueError(f"to_write must be one of {_TO_WRITE_OPTIONS!r}. Got {to_write=!r}")

        if attrs is None:
            attrs = {}

        if not to_write:
            if attrs != {}:
                raise ValueError(f"Attributes cannot be set if {to_write=!r}.")

        self._name = name
        self.dtype = dtype
        self.initial = initial
        self.to_write = to_write
        self.attrs = attrs

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"Variable(name={self._name!r}, dtype={self.dtype!r}, initial={self.initial!r}, to_write={self.to_write!r}, attrs={self.attrs!r})"


class ParticleClass:
    """Define a class of particles. This is used to generate the particle data which is then used in the simulation.

    Parameters
    ----------
    variables : list[Variable]
        List of Variable objects that define the particle's attributes.

    """

    def __init__(self, variables: list[Variable]):
        if not isinstance(variables, list):
            raise TypeError(f"Expected list of Variable objects, got {type(variables)}")
        if not all(isinstance(var, Variable) for var in variables):
            raise ValueError(f"All items in variables must be instances of Variable. Got {variables=!r}")

        self.variables = variables

    def __repr__(self):
        vars = [repr(v) for v in self.variables]
        return f"ParticleClass(variables={_format_list_items_multiline(vars)})"

    def add_variable(self, variable: Variable | list[Variable]):
        """Add a new variable to the Particle class. This returns a new Particle class with the added variable(s).

        Parameters
        ----------
        variable : Variable or list[Variable]
            Variable or list of Variables to be added to the Particle class.
            If a list is provided, all variables will be added to the class.
        """
        if isinstance(variable, Variable):
            variable = [variable]

        for var in variable:
            if not isinstance(var, Variable):
                raise TypeError(f"Expected Variable, got {type(var)}")

        _assert_no_duplicate_variable_names(existing_vars=self.variables, new_vars=variable)

        return ParticleClass(variables=self.variables + variable)


class ParticleSetView:
    """Class to be used in a kernel that links a particle (on the kernel level) to a particle dataset."""

    def __init__(self, data, index):
        self._data = data
        self._index = index

    def __getattr__(self, name):
        # Return a proxy that behaves like the underlying numpy array but
        # writes back into the parent arrays when sliced/modified. This
        # enables constructs like `particles.dlon[mask] += vals` to update
        # the parent arrays rather than temporary copies.
        if name in self._data:
            # If this ParticleSetView represents a single particle (integer
            # index), return the underlying scalar directly to preserve
            # user-facing semantics (e.g., `pset[0].time` should be a number).
            if isinstance(self._index, (int, np.integer)):
                return self._data[name][self._index]
            # For 0-d numpy integer scalars
            if isinstance(self._index, np.ndarray) and self._index.ndim == 0:
                return self._data[name][int(self._index)]
            return ParticleSetViewArray(self._data, self._index, name)
        return self._data[name][self._index]

    def __setattr__(self, name, value):
        if name in ["_data", "_index"]:
            object.__setattr__(self, name, value)
        else:
            self._data[name][self._index] = value

    def __getitem__(self, index):
        # normalize single-element tuple indexing (e.g., (inds,))
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]

        base = self._index
        new_index = np.zeros_like(base, dtype=bool)

        # Boolean mask (could be local-length or global-length)
        if isinstance(index, (np.ndarray, list)) and np.asarray(index).dtype == bool:
            arr = np.asarray(index)
            if arr.size == base.size:
                # global mask
                new_index = arr
            elif arr.size == int(np.sum(base)):
                new_index[base] = arr
            else:
                raise ValueError(
                    f"Boolean index has incompatible length {arr.size} for selection of size {int(np.sum(base))}"
                )
            return ParticleSetView(self._data, new_index)

        # Integer array / list of indices relative to local view
        if isinstance(index, (np.ndarray, list)):
            idx_arr = np.asarray(index)
            if idx_arr.dtype == bool:
                # handled above, but keep for safety
                if idx_arr.size == base.size:
                    new_index = idx_arr
                else:
                    new_index[base] = idx_arr
            else:
                if base.dtype == bool:
                    particle_idxs = np.flatnonzero(base)
                    sel = particle_idxs[idx_arr]
                    new_index[sel] = True
                else:
                    base_arr = np.asarray(base)
                    sel = base_arr[idx_arr]
                    new_index[sel] = True
            return ParticleSetView(self._data, new_index)

        # Slice or single integer index relative to local view
        if isinstance(index, slice) or isinstance(index, int):
            if base.dtype == bool:
                particle_idxs = np.flatnonzero(base)
                sel = particle_idxs[index]
                new_index[sel] = True
            else:
                base_arr = np.asarray(base)
                sel = base_arr[index]
                new_index[sel] = True
            return ParticleSetView(self._data, new_index)

        # Fallback: try to assign directly (preserves previous behaviour for other index types)
        try:
            new_index[base] = index
            return ParticleSetView(self._data, new_index)
        except Exception as e:
            raise TypeError(f"Unsupported index type for ParticleSetView.__getitem__: {type(index)!r}") from e

    # def __setitem__(self, index, value):
    #     """Assign to a subset of particles represented by `index` relative to
    #     this ParticleSetView's current selection.

    #     The incoming `index` is interpreted in the same way as for
    #     `__getitem__`: it indexes into the subset defined by `self._index`.

    #     `value` may be another ParticleSetView (in which case common variables
    #     are copied), or a dict mapping variable names to arrays/scalars which
    #     will be written into the parent arrays at the computed positions.
    #     """
    #     # Map the provided index (which indexes into the current subset)
    #     # back to the full parent-array index.
    #     new_index = np.zeros_like(self._index, dtype=bool)
    #     new_index[self._index] = index

    #     # Helper to perform assignment for a given variable name
    #     def _assign(varname, src):
    #         # write into parent array at positions new_index
    #         self._data[varname][new_index] = src

    #     # Case: assign from another ParticleSetView-like object
    #     if isinstance(value, ParticleSetView):
    #         # copy across common fields
    #         for k in set(self._data.keys()).intersection(value._data.keys()):
    #             _assign(k, value._data[k][value._index])
    #         return

    #     # Case: assign from a dict-like mapping variable names -> values
    #     if isinstance(value, dict):
    #         for k, v in value.items():
    #             if k not in self._data:
    #                 raise KeyError(f"Unknown particle variable: {k}")
    #             _assign(k, v)
    #         return

    #     # Otherwise, if a scalar/array is provided, assign it to all variables
    #     # is ambiguous: raise TypeError to avoid surprising behaviour.
    #     raise TypeError("Unsupported value for ParticleSetView.__setitem__; provide a ParticleSetView or dict of variable values")

    def __len__(self):
        return len(self._index)


class ParticleSetViewArray:
    """Array-like proxy for a particle variable that writes through to the
    parent arrays when mutated.

    Parameters
    ----------
    data : dict-like
        Parent particle storage (mapping varname -> ndarray)
    index : array-like
        Index representing the subset in the parent arrays (boolean mask or integer indices)
    name : str
        Variable name in `data` to proxy
    """

    def __init__(self, data, index, name):
        self._data = data
        self._index = index
        self._name = name

    def __array__(self, dtype=None):
        arr = self._data[self._name][self._index]
        return arr.astype(dtype) if dtype is not None else arr

    def __repr__(self):
        return repr(self.__array__())

    def __len__(self):
        return len(self.__array__())

    def _to_global_index(self, subindex=None):
        """Return a global index (boolean mask or integer indices) that
        addresses the parent arrays. If `subindex` is provided it selects
        within the current local view and maps back to the global index.
        """
        base = self._index
        if subindex is None:
            return base

        # If subindex is a boolean array, support both local-length masks
        # (length == base.sum()) and global-length masks (length == base.size).
        if isinstance(subindex, (np.ndarray, list)) and np.asarray(subindex).dtype == bool:
            arr = np.asarray(subindex)
            if arr.size == base.size:
                # already a global mask
                return arr
            if arr.size == int(np.sum(base)):
                global_mask = np.zeros_like(base, dtype=bool)
                global_mask[base] = arr
                return global_mask
            raise ValueError(
                f"Boolean index has incompatible length {arr.size} for selection of size {int(np.sum(base))}"
            )

        # Handle tuple indexing where the first axis indexes particles
        # and later axes index into the per-particle array shape (e.g. ei[:, igrid])
        if isinstance(subindex, tuple):
            first, *rest = subindex
            # map the first index (local selection) to global particle indices
            if base.dtype == bool:
                particle_idxs = np.flatnonzero(base)
                if isinstance(first, slice):
                    sel = particle_idxs[first]
                elif isinstance(first, (np.ndarray, list)):
                    first_arr = np.asarray(first)
                    if first_arr.dtype == bool:
                        sel = particle_idxs[first_arr]
                    else:
                        sel = particle_idxs[first_arr]
                elif isinstance(first, int):
                    sel = particle_idxs[first]
                else:
                    sel = particle_idxs[first]
            else:
                base_arr = np.asarray(base)
                if isinstance(first, slice):
                    sel = base_arr[first]
                else:
                    sel = base_arr[first]

            # if rest contains a single int (e.g., column), return tuple index
            if len(rest) == 1:
                return (sel, rest[0])
            # return full tuple (sel, ...) for higher-dim cases
            return tuple([sel] + rest)

        # If base is a boolean mask over the parent array and subindex is
        # an integer or slice relative to the local view, map it to integer
        # indices in the parent array.
        if base.dtype == bool:
            if isinstance(subindex, (slice, int)):
                rel = np.flatnonzero(base)[subindex]
                return rel
            # otherwise assume subindex is an integer/array selection relative
            # to the local view and map to global indices
            global_mask = np.zeros_like(base, dtype=bool)
            global_mask[base] = subindex
            return global_mask

        # If base is an array of integer indices
        base_arr = np.asarray(base)
        try:
            return base_arr[subindex]
        except Exception:
            return base_arr[np.asarray(subindex, dtype=bool)]

    def __getitem__(self, subindex):
        # Handle tuple indexing (e.g. [:, igrid]) by applying the tuple
        # to the local selection first. This covers the common case
        # `particles.ei[:, igrid]` where `ei` is a 2D parent array and the
        # second index selects the grid index.
        if isinstance(subindex, tuple):
            local = self._data[self._name][self._index]
            return local[subindex]

        new_index = self._to_global_index(subindex)
        return ParticleSetViewArray(self._data, new_index, self._name)

    def __setitem__(self, subindex, value):
        tgt = self._to_global_index(subindex)
        self._data[self._name][tgt] = value

    # in-place ops must write back into the parent array
    def __iadd__(self, other):
        vals = self._data[self._name][self._index] + (
            other.__array__() if isinstance(other, ParticleSetViewArray) else other
        )
        self._data[self._name][self._index] = vals
        return self

    def __isub__(self, other):
        vals = self._data[self._name][self._index] - (
            other.__array__() if isinstance(other, ParticleSetViewArray) else other
        )
        self._data[self._name][self._index] = vals
        return self

    def __imul__(self, other):
        vals = self._data[self._name][self._index] * (
            other.__array__() if isinstance(other, ParticleSetViewArray) else other
        )
        self._data[self._name][self._index] = vals
        return self

    # Provide simple numpy-like evaluation for binary ops by delegating to ndarray
    def __add__(self, other):
        return self.__array__() + (other.__array__() if isinstance(other, ParticleSetViewArray) else other)

    def __sub__(self, other):
        return self.__array__() - (other.__array__() if isinstance(other, ParticleSetViewArray) else other)

    def __mul__(self, other):
        return self.__array__() * (other.__array__() if isinstance(other, ParticleSetViewArray) else other)

    def __truediv__(self, other):
        return self.__array__() / (other.__array__() if isinstance(other, ParticleSetViewArray) else other)

    def __floordiv__(self, other):
        return self.__array__() // (other.__array__() if isinstance(other, ParticleSetViewArray) else other)

    def __pow__(self, other):
        return self.__array__() ** (other.__array__() if isinstance(other, ParticleSetViewArray) else other)

    def __neg__(self):
        return -self.__array__()

    def __pos__(self):
        return +self.__array__()

    def __abs__(self):
        return abs(self.__array__())

    # Right-hand operations to handle cases like `scalar - ParticleSetViewArray`
    def __radd__(self, other):
        return (other.__array__() if isinstance(other, ParticleSetViewArray) else other) + self.__array__()

    def __rsub__(self, other):
        return (other.__array__() if isinstance(other, ParticleSetViewArray) else other) - self.__array__()

    def __rmul__(self, other):
        return (other.__array__() if isinstance(other, ParticleSetViewArray) else other) * self.__array__()

    def __rtruediv__(self, other):
        return (other.__array__() if isinstance(other, ParticleSetViewArray) else other) / self.__array__()

    def __rfloordiv__(self, other):
        return (other.__array__() if isinstance(other, ParticleSetViewArray) else other) // self.__array__()

    def __rpow__(self, other):
        return (other.__array__() if isinstance(other, ParticleSetViewArray) else other) ** self.__array__()

    # Comparison operators should return plain numpy boolean arrays so that
    # expressions like `mask = particles.gridID == gid` produce an ndarray
    # usable for indexing (rather than another ParticleSetViewArray).
    def __eq__(self, other):
        left = np.asarray(self.__array__())
        if isinstance(other, ParticleSetViewArray):
            right = np.asarray(other.__array__())
        else:
            right = other
        return left == right

    def __ne__(self, other):
        left = np.asarray(self.__array__())
        if isinstance(other, ParticleSetViewArray):
            right = np.asarray(other.__array__())
        else:
            right = other
        return left != right

    def __lt__(self, other):
        left = np.asarray(self.__array__())
        if isinstance(other, ParticleSetViewArray):
            right = np.asarray(other.__array__())
        else:
            right = other
        return left < right

    def __le__(self, other):
        left = np.asarray(self.__array__())
        if isinstance(other, ParticleSetViewArray):
            right = np.asarray(other.__array__())
        else:
            right = other
        return left <= right

    def __gt__(self, other):
        left = np.asarray(self.__array__())
        if isinstance(other, ParticleSetViewArray):
            right = np.asarray(other.__array__())
        else:
            right = other
        return left > right

    def __ge__(self, other):
        left = np.asarray(self.__array__())
        if isinstance(other, ParticleSetViewArray):
            right = np.asarray(other.__array__())
        else:
            right = other
        return left >= right

    # Allow attribute access like .dtype etc. by forwarding to the ndarray
    def __getattr__(self, item):
        arr = self.__array__()
        return getattr(arr, item)


def _assert_no_duplicate_variable_names(*, existing_vars: list[Variable], new_vars: list[Variable]):
    existing_names = {var.name for var in existing_vars}
    for var in new_vars:
        if var.name in existing_names:
            raise ValueError(f"Variable name already exists: {var.name}")


def get_default_particle(spatial_dtype: np.float32 | np.float64) -> ParticleClass:
    if spatial_dtype not in [np.float32, np.float64]:
        raise ValueError(f"spatial_dtype must be np.float32 or np.float64. Got {spatial_dtype=!r}")

    return ParticleClass(
        variables=[
            Variable(
                "lon",
                dtype=spatial_dtype,
                attrs={"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
            ),
            Variable(
                "lat",
                dtype=spatial_dtype,
                attrs={"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
            ),
            Variable(
                "z",
                dtype=spatial_dtype,
                attrs={"standard_name": "vertical coordinate", "units": "m", "positive": "down"},
            ),
            Variable("dlon", dtype=spatial_dtype, to_write=False),
            Variable("dlat", dtype=spatial_dtype, to_write=False),
            Variable("dz", dtype=spatial_dtype, to_write=False),
            Variable(
                "time",
                dtype=np.float64,
                attrs={"standard_name": "time", "units": "seconds", "axis": "T"},
            ),
            Variable(
                "trajectory",
                dtype=np.int64,
                to_write="once",
                attrs={
                    "long_name": "Unique identifier for each particle",
                    "cf_role": "trajectory_id",
                },
            ),
            Variable("obs_written", dtype=np.int32, initial=0, to_write=False),
            Variable("dt", dtype=np.float64, initial=1.0, to_write=False),
            Variable("state", dtype=np.int32, initial=StatusCode.Evaluate, to_write=False),
        ]
    )


Particle = get_default_particle(np.float32)
"""The default Particle used in Parcels simulations."""


def create_particle_data(
    *,
    pclass: ParticleClass,
    nparticles: int,
    ngrids: int,
    time_interval: TimeInterval,
    initial: dict[str, np.array] | None = None,
):
    if initial is None:
        initial = {}

    variables = {var.name: var for var in pclass.variables}

    assert "ei" not in initial, "'ei' is for internal use, and is unique since is only non 1D array"

    dtypes = {var.name: var.dtype for var in variables.values()}

    for var_name in initial:
        if var_name not in variables:
            raise ValueError(f"Variable {var_name} is not defined in the ParticleClass.")

        values = initial[var_name]
        if values.shape != (nparticles,):
            raise ValueError(f"Initial value for {var_name} must have shape ({nparticles},). Got {values.shape=}")

        initial[var_name] = values.astype(dtypes[var_name])

    data = {"ei": np.zeros((nparticles, ngrids), dtype=np.int32), **initial}

    vars_to_create = {k: v for k, v in variables.items() if k not in data}

    for var in vars_to_create.values():
        if isinstance(var.initial, operator.attrgetter):
            name_to_copy = var.initial(_attrgetter_helper)
            data[var.name] = data[name_to_copy].copy()
        else:
            data[var.name] = _create_array_for_variable(var, nparticles, time_interval)
    return data


def _create_array_for_variable(variable: Variable, nparticles: int, time_interval: TimeInterval):
    assert not isinstance(variable.initial, operator.attrgetter), (
        "This function cannot handle attrgetter initial values."
    )
    return np.full(
        shape=(nparticles,),
        fill_value=variable.initial,
        dtype=variable.dtype,
    )
