"""Parcels reprs"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from zarr.storage import DirectoryStore

if TYPE_CHECKING:
    from parcels import Field, FieldSet, ParticleSet


def fieldset_repr(fieldset: FieldSet) -> str:
    """Return a pretty repr for FieldSet"""
    fields = [f for f in fieldset.fields.values() if getattr(f.__class__, "__name__", "") == "Field"]
    vfields = [f for f in fieldset.fields.values() if getattr(f.__class__, "__name__", "") == "VectorField"]

    fields_repr = "\n".join([repr(f) for f in fields])
    vfields_repr = "\n".join([vectorfield_repr(vf, from_fieldset_repr=True) for vf in vfields])

    out = f"""<{type(fieldset).__name__}>
    fields:
{textwrap.indent(fields_repr, 8 * " ")}
    vectorfields:
{textwrap.indent(vfields_repr, 8 * " ")}
"""
    return textwrap.dedent(out).strip()


# TODO add land_value here after HG #2451 is merged
def field_repr(field: Field, level: int = 0) -> str:
    """Return a pretty repr for Field"""
    with xr.set_options(display_expand_data=False):
        out = f"""<{type(field).__name__} {field.name!r}>
    Parcels attributes:
        name            : {field.name!r}
        interp_method   : {field.interp_method!r}
        time_interval   : {field.time_interval!r}
        units           : {field.units!r}
        igrid           : {field.igrid!r}
    DataArray:
{textwrap.indent(repr(field.data), 8 * " ")}
{textwrap.indent(repr(field.grid), 4 * " ")}
"""
    return textwrap.indent(out, " " * level * 4).strip()


def vectorfield_repr(fieldset: FieldSet, from_fieldset_repr=False) -> str:
    """Return a pretty repr for VectorField"""
    out = f"""<{type(fieldset).__name__} {fieldset.name!r}>
    Parcels attributes:
        name                  : {fieldset.name!r}
        vector_interp_method  : {fieldset.vector_interp_method!r}
        vector_type           : {fieldset.vector_type!r}
    {field_repr(fieldset.U, level=1) if not from_fieldset_repr else ""}
    {field_repr(fieldset.V, level=1) if not from_fieldset_repr else ""}
    {field_repr(fieldset.W, level=1) if not from_fieldset_repr and fieldset.W else ""}"""
    return out


def xgrid_repr(grid: Any) -> str:
    """Return a pretty repr for Grid"""
    out = f"""<{type(grid).__name__}>
    Parcels attributes:
        mesh                  : {grid._mesh}
        spatialhash           : {grid._spatialhash}
    xgcm Grid:
{textwrap.indent(repr(grid.xgcm_grid), 8 * " ")}
"""
    return textwrap.dedent(out).strip()


def particleset_repr(pset: ParticleSet) -> str:
    """Return a pretty repr for ParticleSet"""
    if len(pset) < 10:
        particles = [repr(p) for p in pset]
    else:
        particles = [repr(pset[i]) for i in range(7)] + ["..."] + [repr(pset[-1])]

    out = f"""<{type(pset).__name__}>
    Number of particles: {len(pset)}
    Particles:
{_format_list_items_multiline(particles, level=2, with_brackets=False)}
    Pclass:
{textwrap.indent(repr(pset._ptype), 8 * " ")}
"""
    return textwrap.dedent(out).strip()


def particlesetview_repr(pview: Any) -> str:
    """Return a pretty repr for ParticleSetView"""
    time_string = "not_yet_set" if pview.time is None or np.isnan(pview.time) else f"{pview.time:f}"
    out = f"P[{pview.trajectory}]: time={time_string}, z={pview.z:f}, lat={pview.lat:f}, lon={pview.lon:f}"
    vars = [v.name for v in pview._ptype.variables if v.to_write is True and v.name not in ["lon", "lat", "z", "time"]]
    for var in vars:
        out += f", {var}={getattr(pview, var):f}"

    return textwrap.dedent(out).strip()


def particleclass_repr(pclass: Any) -> str:
    """Return a pretty repr for ParticleClass"""
    vars = [repr(v) for v in pclass.variables]
    out = f"""
{_format_list_items_multiline(vars, level=1, with_brackets=False)}
"""
    return textwrap.dedent(out).strip()


def variable_repr(var: Any) -> str:
    """Return a pretty repr for Variable"""
    return f"Variable(name={var._name!r}, dtype={var.dtype!r}, initial={var.initial!r}, to_write={var.to_write!r}, attrs={var.attrs!r})"


def timeinterval_repr(ti: Any) -> str:
    """Return a pretty repr for TimeInterval"""
    return f"TimeInterval(left={ti.left!r}, right={ti.right!r})"


def particlefile_repr(pfile: Any) -> str:
    """Return a pretty repr for ParticleFile"""
    out = f"""<{type(pfile).__name__}>
    store               : {_format_zarr_output_location(pfile.store)}
    outputdt            : {pfile.outputdt!r}
    chunks              : {pfile.chunks!r}
    create_new_zarrfile : {pfile.create_new_zarrfile!r}
    metadata            :
{_format_list_items_multiline(pfile.metadata, level=2, with_brackets=False)}
"""
    return textwrap.dedent(out).strip()


def default_repr(obj: Any):
    if is_builtin_object(obj):
        return repr(obj)
    return object.__repr__(obj)


def _format_list_items_multiline(items: list[str] | dict, level: int = 1, with_brackets: bool = True) -> str:
    """Given a list of strings or a dict, formats them across multiple lines.

    Uses indentation levels of 4 spaces provided by ``level``.

    Example
    -------
    >>> output = _format_list_items_multiline(["item1", "item2", "item3"], 4)
    >>> f"my_items: {output}"
    my_items: [
        item1,
        item2,
        item3,
    ]
    """
    if len(items) == 0:
        return "[]"

    assert level >= 1, "Indentation level >=1 supported"
    indentation_str = level * 4 * " "
    indentation_str_end = (level - 1) * 4 * " "

    if isinstance(items, dict):
        entries = [f"{k!r}: {v!r}" for k, v in items.items()]
    else:
        entries = [i if isinstance(i, str) else repr(i) for i in items]

    if with_brackets:
        items_str = ",\n".join([textwrap.indent(e, indentation_str) for e in entries])
        return f"[\n{items_str}\n{indentation_str_end}]"
    else:
        return "\n".join([textwrap.indent(e, indentation_str) for e in entries])


def _format_zarr_output_location(zarr_obj):
    if isinstance(zarr_obj, DirectoryStore):
        return zarr_obj.path
    return repr(zarr_obj)


def is_builtin_object(obj):
    return obj.__class__.__module__ == "builtins"
