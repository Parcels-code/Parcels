#!/usr/bin/env python
"""A script that can be quickly run that explores the public API of Parcels
and validates docstrings along the way according to the numpydoc conventions.

This script is a best attempt, and it meant as a first line of defence (compared
to the sphinx numpydoc integration which is the ground truth - as those are the
docstrings that end up in the documentation).
"""

import functools
import importlib
import sys
import tomllib
import types
from pathlib import Path

from numpydoc.validate import validate

PROJECT_ROOT = (Path(__file__).parent / "..").resolve()
PUBLIC_MODULES = ["parcels", "parcels.interpolators"]
ROOT_PACKAGE = "parcels"

with open(PROJECT_ROOT / "tools/tool-data.toml", "rb") as f:
    skip_errors = tomllib.load(f)["numpydoc_skip_errors"]


def is_built_in(type_or_instance: type | object):
    if isinstance(type_or_instance, type):
        return type_or_instance.__module__ == "builtins"
    else:
        return type_or_instance.__class__.__module__ == "builtins"


def walk_module(module_str: str, public_api: list[str] | None = None) -> list[str]:
    if public_api is None:
        public_api = []

    module = importlib.import_module(module_str)
    try:
        all_ = module.__all__
    except AttributeError:
        print(f"No __all__ variable found in public module {module_str!r}")
        return public_api

    if module_str not in public_api:
        public_api.append(module_str)
    for item_str in all_:
        item = getattr(module, item_str)
        if isinstance(item, types.ModuleType):
            walk_module(f"{module_str}.{item_str}", public_api)
        if isinstance(item, (types.FunctionType,)):
            public_api.append(f"{module_str}.{item_str}")
        elif is_built_in(item):
            print(f"Found builtin at '{module_str}.{item_str}' of type {type(item)}")
            continue
        elif isinstance(item, type):
            public_api.append(f"{module_str}.{item_str}")
            walk_class(module_str, item, public_api)
        else:
            print(
                f"Encountered unexpected public object at '{module_str}.{item_str}' of {item!r} in public API. Don't know how to handle with numpydoc - ignoring."
            )

    return public_api


def get_public_class_attrs(class_: type) -> set[str]:
    return {a for a in dir(class_) if not a.startswith("_")}


def walk_class(module_str: str, class_: type, public_api: list[str]) -> list[str]:
    class_str = class_.__name__

    # attributes that were introduced by this class specifically - not from inheritance
    attrs = get_public_class_attrs(class_) - functools.reduce(
        set.add, (get_public_class_attrs(base) for base in class_.__bases__)
    )

    public_api.extend([f"{module_str}.{class_str}.{attr_str}" for attr_str in attrs])
    return public_api


def main():
    public_api = []
    for module in PUBLIC_MODULES:
        public_api += walk_module(module)

    public_api = filter(lambda x: x != ROOT_PACKAGE, public_api)  # For some reason doesn't work on root package
    errors = 0
    for item in public_api:
        try:
            res = validate(item)
        except AttributeError:
            continue
        if res["type"] in ("module", "float", "int", "dict"):
            continue
        for err in res["errors"]:
            if err[0] not in skip_errors:
                print(f"{item}: {err}")
                errors += 1
    sys.exit(errors)


if __name__ == "__main__":
    main()
