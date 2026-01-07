"""Parcels reprs"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import xarray as xr

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


def field_repr(field: Field, offset: int = 0) -> str:
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
    return textwrap.indent(out, " " * offset).strip()


def vectorfield_repr(fieldset: FieldSet, from_fieldset_repr=False) -> str:
    """Return a pretty repr for VectorField"""
    out = f"""<{type(fieldset).__name__} {fieldset.name!r}>
    Parcels attributes:
        name                  : {fieldset.name!r}
        vector_interp_method  : {fieldset.vector_interp_method!r}
        vector_type           : {fieldset.vector_type!r}
    {field_repr(fieldset.U, offset=4) if not from_fieldset_repr else ""}
    {field_repr(fieldset.V, offset=4) if not from_fieldset_repr else ""}
    {field_repr(fieldset.W, offset=4) if not from_fieldset_repr and fieldset.W else ""}"""
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


def _format_list_items_multiline(items: list[str], level: int = 1) -> str:
    """Given a list of strings, formats them across multiple lines.

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

    items_str = ",\n".join([textwrap.indent(i, indentation_str) for i in items])
    return f"[\n{items_str}\n{indentation_str_end}]"


def particleset_repr(pset: ParticleSet) -> str:
    """Return a pretty repr for ParticleSet"""
    if len(pset) < 10:
        particles = [repr(p) for p in pset]
    else:
        particles = [repr(pset[i]) for i in range(7)] + ["..."]

    out = f"""<{type(pset).__name__}>
    fieldset   :
{textwrap.indent(repr(pset.fieldset), " " * 8)}
    ptype      : {pset._ptype}
    # particles: {len(pset)}
    particles  : {_format_list_items_multiline(particles, level=2)}
"""
    return textwrap.dedent(out).strip()


def default_repr(obj: Any):
    if is_builtin_object(obj):
        return repr(obj)
    return object.__repr__(obj)


def is_builtin_object(obj):
    return obj.__class__.__module__ == "builtins"
