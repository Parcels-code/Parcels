"""Check that `parcels.convert` behaves the same as a frozen copy of it (`tests/_legacy_convert.py`).

These tests are a safety net for refactoring `parcels.convert`. Each case builds fresh inputs, then
runs them through the legacy and current converter, and requires the same output dataset, the same
error (if any), the same log messages and the same warnings.

Inputs are built from datasets that already exist in the test suite, plus small variations of them
designed to exercise branches of `convert.py` that the other tests don't reach. Use

    pytest tests/test_convert_legacy.py --cov=parcels.convert --cov-branch --cov-report=term-missing

to see which branches are still not exercised.

Delete this file and `tests/_legacy_convert.py` once the refactor has landed.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import uxarray as ux
import xarray as xr

import parcels.convert as convert
import parcels.tutorial
from parcels._datasets.remote import open_remote_dataset
from parcels._datasets.structured.circulation_models import datasets as datasets_circulation_models
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from tests import _legacy_convert

Inputs = tuple[tuple[Any, ...], dict[str, Any]]


@dataclass(frozen=True)
class Case:
    id: str
    converter: str
    make_inputs: Callable[[], Inputs]
    """Must build fresh inputs on every call, as converters may modify their inputs in place."""
    raises: type[Exception] | None = None
    """Exception that the legacy converter is expected to raise, to check the case tests what it claims to."""
    marks: list[pytest.MarkDecorator] = field(default_factory=list)


@dataclass
class Outcome:
    result: Any
    error: tuple[type[Exception], str] | None
    logs: list[tuple[str, str]]
    warnings: list[tuple[type[Warning], str]]


def _run(module: ModuleType, case: Case, caplog: pytest.LogCaptureFixture) -> Outcome:
    args, kwargs = case.make_inputs()
    caplog.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result, error = getattr(module, case.converter)(*args, **kwargs), None
        except Exception as e:
            result, error = None, (type(e), str(e))
    return Outcome(
        result=result,
        error=error,
        logs=[(r.levelname, r.getMessage()) for r in caplog.records],
        warnings=[(w.category, str(w.message)) for w in caught],
    )


# ---------------------------------------------------------------------------------------------
# Input builders. Synthetic datasets are module-level objects shared across the test suite, so
# they are always deep-copied.
# ---------------------------------------------------------------------------------------------


def _synthetic(name: str) -> xr.Dataset:
    return datasets_circulation_models[name].copy(deep=True)


def _synthetic_ux(name: str) -> ux.UxDataset:
    return datasets_unstructured[name].copy(deep=True)


def _kw(**kwargs) -> Inputs:
    return (), kwargs


# NEMO


def _nemo_zonal() -> dict[str, Any]:
    return dict(
        U=parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/U"),
        V=parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/V"),
        coords=parcels.tutorial.open_dataset("NemoCurvilinear_data_zonal/mesh_mask"),
    )


def _nemo_zonal_datasets() -> Inputs:
    d = _nemo_zonal()
    return _kw(fields=dict(U=d["U"], V=d["V"]), coords=d["coords"])


def _nemo_zonal_dataarrays() -> Inputs:
    d = _nemo_zonal()
    return _kw(fields=dict(U=d["U"]["U"], V=d["V"]["V"]), coords=d["coords"])


def _nemo_zonal_coords_with_time(ntime: int) -> Callable[[], Inputs]:
    def make() -> Inputs:
        d = _nemo_zonal()
        coords = d["coords"].expand_dims(time=np.arange(ntime))
        return _kw(fields=dict(U=d["U"], V=d["V"]), coords=coords)

    return make


def _nemo_zonal_missing_required_coord() -> Inputs:
    d = _nemo_zonal()
    return _kw(fields=dict(U=d["U"], V=d["V"]), coords=d["coords"].drop_vars("gphif"))


def _nemo_northsea_with_w() -> Inputs:
    U, V, W, mesh = (
        parcels.tutorial.open_dataset(f"NemoNorthSeaORCA025-N006_data/{name}") for name in ("U", "V", "W", "mesh_mask")
    )
    return _kw(fields=dict(U=U["uo"], V=V["vo"], W=W["wo"]), coords=mesh)


def _nemo_northsea_no_time() -> Inputs:
    U, V, mesh = (
        parcels.tutorial.open_dataset(f"NemoNorthSeaORCA025-N006_data/{name}") for name in ("U", "V", "mesh_mask")
    )
    fields = dict(U=U["uo"].isel(time_counter=0, drop=True), V=V["vo"].isel(time_counter=0, drop=True))
    return _kw(fields=fields, coords=mesh[["glamf", "gphif"]])


def _nemo_northsea_with_tracer() -> Inputs:
    U, V, mesh = (
        parcels.tutorial.open_dataset(f"NemoNorthSeaORCA025-N006_data/{name}") for name in ("U", "V", "mesh_mask")
    )
    return _kw(fields=dict(U=U["uo"], V=V["vo"], T=xr.zeros_like(U["uo"])), coords=mesh[["glamf", "gphif"]])


def _nemo_moi_metadata_only() -> Inputs:
    # Global 1/12 degree data - subset so that comparing values stays cheap
    subset = dict(x=slice(0, 20), y=slice(0, 15))
    U = open_remote_dataset("Benchmarks_MOi_data_metadata-only/U")[["vozocrtx"]].rename_vars({"vozocrtx": "U"})
    V = open_remote_dataset("Benchmarks_MOi_data_metadata-only/V")[["vomecrty"]].rename_vars({"vomecrty": "V"})
    depth = open_remote_dataset("Benchmarks_MOi_data_metadata-only/W")["depthw"]
    coords = open_remote_dataset("Benchmarks_MOi_data_metadata-only/mesh")[["glamf", "gphif"]].isel(t=0)
    coords["depthw"] = depth
    return _kw(fields=dict(U=U.isel(subset), V=V.isel(subset)), coords=coords.isel(subset))


# MITgcm


def _mitgcm_tutorial() -> xr.Dataset:
    return parcels.tutorial.open_dataset("MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant")


def _mitgcm_dataarrays() -> Inputs:
    ds = _mitgcm_tutorial()
    return _kw(fields={"U": ds.UVEL, "V": ds.VVEL}, coords=ds)


def _mitgcm_datasets() -> Inputs:
    ds = _mitgcm_tutorial().rename_vars(UVEL="U", VVEL="V")
    return _kw(fields={"U": ds, "V": ds}, coords=ds[["XG", "YG", "Zl", "time"]])


def _mitgcm_depth_increasing() -> Inputs:
    ds = _mitgcm_tutorial()
    ds = ds.assign_coords(Zl=-ds.Zl)
    return _kw(fields={"U": ds.UVEL, "V": ds.VVEL}, coords=ds)


def _mitgcm_single_depth() -> Inputs:
    ds = _mitgcm_tutorial().isel(Zl=[0])
    return _kw(fields={"U": ds.UVEL, "V": ds.VVEL}, coords=ds)


def _mitgcm_missing_required_coord() -> Inputs:
    ds = _mitgcm_tutorial()
    return _kw(fields={"U": ds.UVEL, "V": ds.VVEL}, coords=ds.drop_vars("Zl"))


# CROCO


def _croco_synthetic() -> Inputs:
    ds = _synthetic("ds_CROCO_idealized")
    return _kw(fields={"U": ds["u"], "V": ds["v"]}, coords=ds[["x_rho", "y_rho", "s_w", "time"]])


def _croco_synthetic_time_units(units: str) -> Callable[[], Inputs]:
    def make() -> Inputs:
        ds = _synthetic("ds_CROCO_idealized")
        ds["time"].attrs["units"] = units
        return _kw(fields={"U": ds["u"], "V": ds["v"]}, coords=ds[["x_rho", "y_rho", "s_w", "time"]])

    return make


def _croco_synthetic_datetime_time() -> Inputs:
    ds = _synthetic("ds_CROCO_idealized")
    ds = ds.assign_coords(time=np.datetime64("2000-01-01") + ds["time"].values.astype("timedelta64[s]"))
    return _kw(fields={"U": ds["u"], "V": ds["v"]}, coords=ds[["x_rho", "y_rho", "s_w", "time"]])


def _croco_synthetic_datasets() -> Inputs:
    ds = _synthetic("ds_CROCO_idealized").rename_vars(u="U", v="V")
    return _kw(fields={"U": ds, "V": ds}, coords=ds)


def _croco_tutorial_3d() -> Inputs:
    ds = parcels.tutorial.open_dataset("CROCOidealized_data/data")
    fields = {
        "U": ds["u"],
        "V": ds["v"],
        "W": ds["w"],
        "h": ds["h"],
        "zeta": ds["zeta"],
        "Cs_w": ds["Cs_w"],
    }
    return _kw(fields=fields, coords=ds)


# Copernicus Marine


def _copernicus(**fields_from_vars: str) -> Callable[[], Inputs]:
    def make() -> Inputs:
        ds = _synthetic("ds_copernicusmarine")
        return _kw(fields={name: ds[var] for name, var in fields_from_vars.items()})

    return make


def _copernicus_waves() -> Inputs:
    ds = _synthetic("ds_copernicusmarine_waves")
    return _kw(fields={"U": ds["VSDX"], "V": ds["VSDY"]})


def _copernicus_with_w() -> Inputs:
    ds = _synthetic("ds_copernicusmarine")
    return _kw(fields={"U": ds["uo"], "V": ds["vo"], "W": 0.5 * ds["uo"]})


def _copernicus_no_depth() -> Inputs:
    ds = _synthetic("ds_copernicusmarine").isel(depth=0).drop_vars("depth")
    return _kw(fields={"U": ds["uo"], "V": ds["vo"]})


def _copernicus_datasets() -> Inputs:
    ds = _synthetic("ds_copernicusmarine").rename_vars(uo="U", vo="V")
    return _kw(fields={"U": ds, "V": ds})


def _copernicus_with_coords() -> Inputs:
    ds = _synthetic("ds_copernicusmarine")
    return _kw(fields={"U": ds["uo"], "V": ds["vo"]}, coords=ds[["time", "depth", "latitude", "longitude"]])


def _copernicus_two_x_coords() -> Inputs:
    ds = _synthetic("ds_copernicusmarine")
    ds = ds.assign_coords(longitude2=ds["longitude"])
    return _kw(fields={"U": ds["uo"], "V": ds["vo"]})


# Delft3D


def _delft3d_dataarrays() -> Inputs:
    ds = open_remote_dataset("Delft3D_data/Rotterdam_tiny")
    return _kw(fields={"U": ds["VELU"], "V": ds["VELV"]}, coords=ds[["XZETA", "YZETA", "SIGMA_C"]])


def _delft3d_datasets() -> Inputs:
    ds = open_remote_dataset("Delft3D_data/Rotterdam_tiny").rename_vars(VELU="U", VELV="V")
    return _kw(fields={"U": ds, "V": ds}, coords=ds[["XZETA", "YZETA", "SIGMA_C"]])


# FESOM2 and ICON (UGRID)


def _fesom_remote() -> Inputs:
    grid = ux.open_grid(open_remote_dataset("Benchmarks_FESOM2-baroclinic-gyre/grid"))
    data = open_remote_dataset("Benchmarks_FESOM2-baroclinic-gyre/data")
    return (ux.UxDataset(data, uxgrid=grid),), {}


def _ux_synthetic(name: str, transform: Callable[[ux.UxDataset], ux.UxDataset] = lambda ds: ds) -> Callable[[], Inputs]:
    def make() -> Inputs:
        return (transform(_synthetic_ux(name)),), {}

    return make


_FESOM_SYNTHETIC = "fesom2_square_delaunay_uniform_z_coordinate"
_ICON_SYNTHETIC = "icon_square_delaunay_uniform_z_coordinate"


def _rename_vertical(interface: str, center: str, attrs: dict[str, str] | None = None):
    """Rename FESOM vertical dims so that the known (nz, nz1) mapping no longer applies."""

    def transform(ds: ux.UxDataset) -> ux.UxDataset:
        ds = ds.rename({"nz": interface, "nz1": center})
        if attrs is not None:
            ds = ds.assign_coords({dim: ds[dim].assign_attrs(attrs) for dim in (interface, center)})
        return ds

    return transform


def _drop_vertical(ds: ux.UxDataset) -> ux.UxDataset:
    return ds.drop_dims(["nz", "nz1"])


def _already_zf_zc(ds: ux.UxDataset) -> ux.UxDataset:
    return ds.rename({"depth_2": "zf", "depth": "zc"})


CASES = [
    # NEMO
    Case("nemo-zonal-datasets", "nemo_to_sgrid", _nemo_zonal_datasets),
    Case("nemo-zonal-dataarrays", "nemo_to_sgrid", _nemo_zonal_dataarrays),
    Case("nemo-zonal-coords-time-1", "nemo_to_sgrid", _nemo_zonal_coords_with_time(1)),
    Case("nemo-zonal-coords-time-2", "nemo_to_sgrid", _nemo_zonal_coords_with_time(2), raises=ValueError),
    Case("nemo-zonal-missing-coord", "nemo_to_sgrid", _nemo_zonal_missing_required_coord, raises=ValueError),
    Case("nemo-northsea-with-w", "nemo_to_sgrid", _nemo_northsea_with_w),
    Case("nemo-northsea-no-time", "nemo_to_sgrid", _nemo_northsea_no_time, raises=ValueError),
    Case("nemo-northsea-with-tracer", "nemo_to_sgrid", _nemo_northsea_with_tracer),
    Case("nemo-moi-metadata-only", "nemo_to_sgrid", _nemo_moi_metadata_only),
    # MITgcm
    Case("mitgcm-dataarrays", "mitgcm_to_sgrid", _mitgcm_dataarrays),
    Case("mitgcm-datasets", "mitgcm_to_sgrid", _mitgcm_datasets),
    Case("mitgcm-depth-increasing", "mitgcm_to_sgrid", _mitgcm_depth_increasing),
    Case("mitgcm-single-depth", "mitgcm_to_sgrid", _mitgcm_single_depth),
    Case("mitgcm-missing-coord", "mitgcm_to_sgrid", _mitgcm_missing_required_coord, raises=ValueError),
    # CROCO
    Case("croco-synthetic", "croco_to_sgrid", _croco_synthetic),
    Case("croco-datetime-time", "croco_to_sgrid", _croco_synthetic_datetime_time),
    Case("croco-synthetic-datasets", "croco_to_sgrid", _croco_synthetic_datasets),
    Case("croco-time-hours", "croco_to_sgrid", _croco_synthetic_time_units("hours since 2000-01-01")),
    Case("croco-time-days", "croco_to_sgrid", _croco_synthetic_time_units("days since 2000-01-01")),
    Case("croco-time-minutes", "croco_to_sgrid", _croco_synthetic_time_units("minutes since 2000-01-01")),
    Case("croco-tutorial-3d", "croco_to_sgrid", _croco_tutorial_3d),
    # Copernicus Marine
    Case("copernicus-uv", "copernicusmarine_to_sgrid", _copernicus(U="uo", V="vo")),
    Case("copernicus-no-currents", "copernicusmarine_to_sgrid", _copernicus(do="uo")),
    Case("copernicus-waves", "copernicusmarine_to_sgrid", _copernicus_waves),
    Case("copernicus-with-w", "copernicusmarine_to_sgrid", _copernicus_with_w),
    Case("copernicus-no-depth", "copernicusmarine_to_sgrid", _copernicus_no_depth),
    Case("copernicus-datasets", "copernicusmarine_to_sgrid", _copernicus_datasets),
    Case("copernicus-with-coords", "copernicusmarine_to_sgrid", _copernicus_with_coords),
    Case("copernicus-two-x-coords", "copernicusmarine_to_sgrid", _copernicus_two_x_coords, raises=ValueError),
    # Delft3D
    Case("delft3d-dataarrays", "delft3d_to_sgrid", _delft3d_dataarrays),
    Case("delft3d-datasets", "delft3d_to_sgrid", _delft3d_datasets),
    # FESOM2
    Case("fesom-remote", "fesom_to_ugrid", _fesom_remote),
    Case("fesom-synthetic", "fesom_to_ugrid", _ux_synthetic(_FESOM_SYNTHETIC)),
    Case("fesom-synthetic-antimeridian", "fesom_to_ugrid", _ux_synthetic("fesom2_square_delaunay_antimeridian")),
    Case(
        "fesom-vertical-from-cf-axis",
        "fesom_to_ugrid",
        _ux_synthetic(_FESOM_SYNTHETIC, _rename_vertical("lev_i", "lev_c", {"axis": "Z"})),
    ),
    Case(
        "fesom-vertical-from-cf-positive",
        "fesom_to_ugrid",
        _ux_synthetic(_FESOM_SYNTHETIC, _rename_vertical("lev_i", "lev_c", {"positive": "down"})),
    ),
    Case(
        "fesom-vertical-from-cf-standard-name",
        "fesom_to_ugrid",
        _ux_synthetic(_FESOM_SYNTHETIC, _rename_vertical("lev_i", "lev_c", {"standard_name": "depth"})),
    ),
    Case(
        "fesom-vertical-from-sizes",
        "fesom_to_ugrid",
        _ux_synthetic(_FESOM_SYNTHETIC, _rename_vertical("lev_i", "lev_c")),
    ),
    Case(
        "fesom-vertical-missing",
        "fesom_to_ugrid",
        _ux_synthetic(_FESOM_SYNTHETIC, _drop_vertical),
        raises=ValueError,
    ),
    # ICON
    Case("icon-synthetic", "icon_to_ugrid", _ux_synthetic(_ICON_SYNTHETIC)),
    Case("icon-already-zf-zc", "icon_to_ugrid", _ux_synthetic(_ICON_SYNTHETIC, _already_zf_zc)),
]


def _apply_intentional_changes(case: Case, legacy: Outcome) -> Outcome:
    """Apply the behaviour changes made on purpose since the legacy copy was taken to the legacy outcome."""
    if legacy.result is not None and case.converter.endswith("_to_sgrid"):
        # All SGRID converters now clear the global attributes and only set "Conventions"
        legacy.result.attrs = {"Conventions": "SGRID"}

    # Adding a missing depth dimension now logs the same message for all converters, and puts depth after time
    old_depth_messages = {
        "No depth dimension found in your dataset. Assuming no depth (i.e., surface data).",
        "No depth dimension found in dataset. Added a singleton depth dimension.",
    }
    if any(message in old_depth_messages for _, message in legacy.logs):
        legacy.logs = [
            (level, _NEW_DEPTH_MESSAGE if message in old_depth_messages else message) for level, message in legacy.logs
        ]
        if legacy.result is not None:
            legacy.result = legacy.result.transpose("time", "depth", ..., missing_dims="ignore")
    return legacy


_NEW_DEPTH_MESSAGE = (
    "No depth dimension found in dataset. Assuming surface data and adding a singleton depth dimension."
)


@pytest.mark.parametrize("case", [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
def test_convert_matches_legacy(case: Case, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    expected = _apply_intentional_changes(case, _run(_legacy_convert, case, caplog))
    actual = _run(convert, case, caplog)

    if case.raises is None:
        assert expected.error is None, f"Legacy converter unexpectedly raised {expected.error}"
    else:
        assert expected.error is not None and expected.error[0] is case.raises, (
            f"Legacy converter did not raise {case.raises.__name__} (got {expected.error})"
        )

    assert actual.error == expected.error
    assert actual.logs == expected.logs
    assert actual.warnings == expected.warnings
    if expected.result is not None:
        assert type(actual.result) is type(expected.result)
        xr.testing.assert_identical(actual.result, expected.result)
