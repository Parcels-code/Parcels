from __future__ import annotations

import ast
import inspect
import textwrap
import types
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from parcels._core.warnings import KernelWarning

_MIGRATION_GUIDE: str = "https://docs.oceanparcels.org/en/latest/user_guide/v4-migration.html"

_DEPRECATED_COORD_ATTRS: dict[str, str] = {
    "lon": "x",
    "lat": "y",
    "depth": "z",
    "xi": "ei",
    "yi": "ei",
    "zi": "ei",
}

_DEPRECATED_DELTA_NAMES: dict[str, str] = {
    "particle_dlon": "particles.dx",
    "particle_dlat": "particles.dy",
    "particle_ddepth": "particles.dz",
}

_PARTICLE_NAMES: tuple[str, ...] = ("particle", "particles")

_F = TypeVar("_F", bound=types.FunctionType)


class KernelValidationError(Exception):
    """Raised when a kernel function has errors that make it incompatible with Parcels v4."""


@dataclass
class _Violation:
    message: str
    lineno: int | None = None


def _format_violations(func_name: str, violations: list[_Violation], label: str) -> str:
    """Format a list of violations into a readable string."""
    header: str = f"Kernel `{func_name}` has {len(violations)} {label}(s):\n"
    lines: list[str] = []
    for v in violations:
        prefix: str = f"Line {v.lineno}: " if v.lineno is not None else ""
        lines.append(f"  - {prefix}{v.message}")
    return header + "\n".join(lines)


@dataclass
class _ValidationResult:
    errors: list[_Violation] = field(default_factory=list)
    warnings: list[_Violation] = field(default_factory=list)

    def get_report(self, func: types.FunctionType):
        return "".join(
            [
                _format_violations(func.__name__, self.errors, "validation error"),
                "\n\n",
                _format_violations(func.__name__, self.warnings, "warning"),
            ]
        ).strip()


class _KernelValidator(ast.NodeVisitor):
    """AST NodeVisitor that collects kernel validation errors and warnings."""

    def __init__(self, func_name: str) -> None:
        self.func_name: str = func_name
        self.result: _ValidationResult = _ValidationResult()

    def check_signature(self, node: ast.FunctionDef) -> None:
        """Check that the kernel signature is (particles, fieldset). Error if not."""
        param_names: list[str] = [a.arg for a in node.args.args]
        if param_names != ["particles", "fieldset"]:
            self.result.errors.append(
                _Violation(
                    message=(
                        f"Kernel signature must be `def {self.func_name}(particles, fieldset)`, "
                        f"but got `def {self.func_name}({', '.join(param_names)})`. "
                        f"See {_MIGRATION_GUIDE} for more info."
                    ),
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        """Check for particle.delete() / particles.delete() calls. Error."""
        callee = node.func
        if (
            isinstance(callee, ast.Attribute)
            and callee.attr == "delete"
            and isinstance(callee.value, ast.Name)
            and callee.value.id in _PARTICLE_NAMES
        ):
            self.result.errors.append(
                _Violation(
                    message=(
                        f"`{callee.value.id}.delete()` is no longer valid syntax in Parcels kernels. "
                        f"Use `particle.state = StatusCode.Delete` instead. "
                        f"See {_MIGRATION_GUIDE} for more info."
                    ),
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Check for deprecated and problematic attribute accesses. Warnings."""
        if isinstance(node.value, ast.Name) and node.value.id in _PARTICLE_NAMES:
            name: str = node.value.id
            attr: str = node.attr

            # Deprecated coordinate attributes (warning)
            if attr in _DEPRECATED_COORD_ATTRS:
                replacement: str = _DEPRECATED_COORD_ATTRS[attr]
                self.result.warnings.append(
                    _Violation(
                        message=(
                            f"`{name}.{attr}` is deprecated. "
                            f"Use `particles.{replacement}` instead. "
                            f"See {_MIGRATION_GUIDE} for more info."
                        ),
                        lineno=node.lineno,
                    )
                )

            # Index attribute sampling (warning)
            if attr == "ei":
                self.result.warnings.append(
                    _Violation(
                        message=(
                            f"Be careful when sampling `{name}.ei`, as this is updated "
                            f"in the kernel loop. Best to place the sampling statement "
                            f"before advection."
                        ),
                        lineno=node.lineno,
                    )
                )

            # Direct assignment to location attributes (error)
            # Handled in visit_Assign / visit_AugAssign instead

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Check for deprecated v3 delta variable names like particle_dlon. Warning."""
        if node.id in _DEPRECATED_DELTA_NAMES:
            replacement: str = _DEPRECATED_DELTA_NAMES[node.id]
            self.result.warnings.append(
                _Violation(
                    message=(
                        f"`{node.id}` is no longer valid in Parcels v4. "
                        f"Use `{replacement}` instead. "
                        f"See {_MIGRATION_GUIDE} for more info."
                    ),
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def _check_location_assignment(self, targets: Sequence[ast.expr], lineno: int) -> None:
        """Check if any target is a direct assignment to a particle location attribute. Error."""
        location_attrs: set[str] = {"lon", "lat", "depth", "x", "y", "z"}
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in location_attrs
                and isinstance(target.value, ast.Name)
                and target.value.id in _PARTICLE_NAMES
            ):
                self.result.errors.append(
                    _Violation(
                        message=(
                            f"Don't change the location of a particle directly in a Kernel. "
                            f"Use `particles.dx`, `particles.dy`, `particles.dz` instead of "
                            f"assigning to `{target.value.id}.{target.attr}`. "
                            f"See {_MIGRATION_GUIDE} for more info."
                        ),
                        lineno=lineno,
                    )
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_location_assignment(node.targets, node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_location_assignment([node.target], node.lineno)
        self.generic_visit(node)


def _validate_kernel_ast(func: Callable[..., object]) -> _ValidationResult:
    """Parse the AST of a kernel function and return a _ValidationResult."""
    source: str = textwrap.dedent(inspect.getsource(func))
    tree: ast.Module = ast.parse(source)
    func_def: ast.stmt = tree.body[0]

    validator = _KernelValidator(func.__name__)
    validator.check_signature(func_def)  # type: ignore[arg-type]
    validator.visit(func_def)

    return validator.result


def validate_kernel(func: _F) -> _F:
    """Decorator that validates a Parcels v4 kernel function.

    Errors (raises ``KernelValidationError``):
    - Kernel signature is not ``def ...(particles, fieldset)``
    - ``particle.delete()`` or ``particles.delete()`` calls
    - Direct assignment to particle location attributes (x/y/z/lon/lat/depth)

    Warnings
    --------
    - References to deprecated ``particle.lon``, ``.lat``, ``.depth`` (use ``.x``, ``.y``, ``.z``)
    - References to deprecated ``particle_dlon``, ``particle_dlat``, ``particle_ddepth``
    - Sampling ``particle.ei`` (ordering concern)
    """
    result: _ValidationResult = _validate_kernel_ast(func)
    report: str = result.get_report(func)

    if result.warnings:
        warnings.warn(report, KernelWarning, stacklevel=2)

    if result.errors:
        raise KernelValidationError(report)

    return func
