import pytest
import xarray as xr

import parcels
import parcels.convert as convert
from parcels import FieldSet
from parcels._core.utils import sgrid
from parcels._datasets.structured.circulation_models import datasets as datasets_circulation_models


def test_nemo_to_sgrid():
    data_folder = parcels.download_example_dataset("NemoCurvilinear_data")
    U = xr.open_mfdataset(data_folder.glob("*U.nc4"))
    V = xr.open_mfdataset(data_folder.glob("*V.nc4"))
    coords = xr.open_dataset(data_folder / "mesh_mask.nc4")

    ds = convert.nemo_to_sgrid(fields=dict(U=U, V=V), coords=coords)

    assert ds["grid"].attrs == {
        "cf_role": "grid_topology",
        "topology_dimension": 2,
        "node_dimensions": "x y",
        "face_dimensions": "x_center:x (padding:low) y_center:y (padding:low)",
        "node_coordinates": "lon lat",
        "vertical_dimensions": "z_center:depth (padding:high)",
    }

    meta = sgrid.parse_grid_attrs(ds["grid"].attrs)

    # Assuming that node_dimension1 and node_dimension2 correspond to X and Y respectively
    # check that U and V are properly defined on the staggered grid
    assert {
        meta.get_value_by_id("node_dimension1"),  # X edge
        meta.get_value_by_id("face_dimension2"),  # Y center
    }.issubset(set(ds["U"].dims))
    assert {
        meta.get_value_by_id("face_dimension1"),  # X center
        meta.get_value_by_id("node_dimension2"),  # Y edge
    }.issubset(set(ds["V"].dims))


_COPERNICUS_DATASETS = [
    datasets_circulation_models["ds_copernicusmarine"],
    datasets_circulation_models["ds_copernicusmarine_waves"],
]


@pytest.mark.parametrize("ds", _COPERNICUS_DATASETS)
def test_fieldset_from_copernicusmarine(ds, caplog):
    if "uo" in ds:
        fields = {"U": ds["uo"], "V": ds["vo"]}
    elif "VSDX" in ds:
        fields = {"U": ds["VSDX"], "V": ds["VSDY"]}
    else:
        raise ValueError("Test dataset does not contain recognized current variables.")
    ds_fset = convert.copernicusmarine_to_sgrid(fields=fields)
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields


def test_fieldset_from_copernicusmarine_no_currents(caplog):
    ds = datasets_circulation_models["ds_copernicusmarine"]
    ds_fset = convert.copernicusmarine_to_sgrid(fields={"do": ds["uo"]})
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    assert "U" not in fieldset.fields
    assert "V" not in fieldset.fields
    assert "UV" not in fieldset.fields
    assert caplog.text == ""


@pytest.mark.parametrize("ds", _COPERNICUS_DATASETS)
def test_fieldset_from_copernicusmarine_no_logs(ds, caplog):
    ds = ds.copy()
    zeros = xr.zeros_like(list(ds.data_vars.values())[0])
    ds["U"] = zeros
    ds["V"] = zeros

    ds_fset = convert.copernicusmarine_to_sgrid(fields={"U": ds["U"], "V": ds["V"]})
    fieldset = FieldSet.from_sgrid_conventions(ds_fset)
    assert "U" in fieldset.fields
    assert "V" in fieldset.fields
    assert "UV" in fieldset.fields
    assert caplog.text == ""
