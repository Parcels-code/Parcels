import xarray as xr

import parcels
import parcels.convert as convert
from parcels._core.utils import sgrid


def test_nemo_to_sgrid():
    data_folder = parcels.download_example_dataset("NemoCurvilinear_data")
    U = xr.open_mfdataset(data_folder.glob("*U.nc4"))
    V = xr.open_mfdataset(data_folder.glob("*V.nc4"))
    coords = xr.open_dataset(data_folder / "mesh_mask.nc4")

    ds = convert.nemo_to_sgrid(U=U, V=V, coords=coords)

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
