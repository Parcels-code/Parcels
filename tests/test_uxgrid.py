import numpy as np
import pytest
import uxarray as ux

from parcels import UxGrid
from parcels._core.mesh import FlatMesh, SphericalMesh
from parcels._core.model import _infer_mesh_from_ugrid_data
from parcels._datasets.unstructured.generic import datasets as uxdatasets

GENERIC_Z_COORDS = ["nz", "zf", "depth_2"]


@pytest.mark.parametrize("uxds", [pytest.param(uxds, id=key) for key, uxds in uxdatasets.items()])
def test_uxgrid_init_on_generic_datasets(uxds):
    vertical_coord = next((z_coord for z_coord in uxds.coords if z_coord in GENERIC_Z_COORDS), None)
    UxGrid(uxds.uxgrid, z=uxds.coords[vertical_coord], mesh=_infer_mesh_from_ugrid_data(uxds))


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_axes(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh=_infer_mesh_from_ugrid_data(uxds))
    assert grid.axes == ["Z", "FACE"]


@pytest.mark.parametrize("time_varying", [False, True])
def test_uxgrid_depth_positive_direction(time_varying):
    uxds = uxdatasets["stommel_gyre_delaunay"]
    if time_varying:
        values = np.broadcast_to(np.array([0.0, 1.0])[None, :, None], (2, 2, uxds.uxgrid.n_face))
        z = ux.UxDataArray(values, dims=("time", "zf", "n_face"), uxgrid=uxds.uxgrid)
    else:
        z = uxds.coords["zf"].copy(deep=True)
        z.attrs.pop("positive", None)
    grid = UxGrid(uxds.uxgrid, z=z, mesh=FlatMesh())

    assert grid.depth_positive_direction == "down"
    grid.z.attrs["positive"] = "up"
    assert grid.depth_positive_direction == "up"


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
@pytest.mark.parametrize("mesh", [FlatMesh(), SphericalMesh()])
def test_uxgrid_mesh(uxds, mesh):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh=mesh)
    assert grid.mesh == mesh


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_mesh_inferred_from_units(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh=_infer_mesh_from_ugrid_data(uxds))
    assert isinstance(grid.mesh, SphericalMesh)  # uxarray sets degrees_east/north units


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_uxgrid_mesh_string_not_allowed(uxds):
    with pytest.raises(TypeError):
        UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh="flat")


@pytest.mark.parametrize("uxds", [uxdatasets["stommel_gyre_delaunay"]])
def test_xgrid_get_axis_dim(uxds):
    grid = UxGrid(uxds.uxgrid, z=uxds.coords["zf"], mesh=_infer_mesh_from_ugrid_data(uxds))

    assert grid.get_axis_dim("FACE") == 721
    assert grid.get_axis_dim("Z") == 2
