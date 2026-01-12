import numpy as np
import pytest
import uxarray as ux

from parcels import (
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    UxGrid,
    VectorField,
    download_example_dataset,
)
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.interpolators import (
    UxPiecewiseConstantFace,
    UxPiecewiseLinearNode,
)


@pytest.fixture
def ds_fesom_channel() -> ux.UxDataset:
    fesom_path = download_example_dataset("FESOM_periodic_channel")
    grid_path = f"{fesom_path}/fesom_channel.nc"
    data_path = [
        f"{fesom_path}/u.fesom_channel.nc",
        f"{fesom_path}/v.fesom_channel.nc",
        f"{fesom_path}/w.fesom_channel.nc",
    ]
    ds = ux.open_mfdataset(grid_path, data_path).rename_vars({"u": "U", "v": "V", "w": "W"})
    return ds


@pytest.fixture
def uv_fesom_channel(ds_fesom_channel) -> VectorField:
    UV = VectorField(
        name="UV",
        U=Field(
            name="U",
            data=ds_fesom_channel.U,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"], mesh="flat"),
            interp_method=UxPiecewiseConstantFace,
        ),
        V=Field(
            name="V",
            data=ds_fesom_channel.V,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"], mesh="flat"),
            interp_method=UxPiecewiseConstantFace,
        ),
    )
    return UV


@pytest.fixture
def uvw_fesom_channel(ds_fesom_channel) -> VectorField:
    UVW = VectorField(
        name="UVW",
        U=Field(
            name="U",
            data=ds_fesom_channel.U,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"], mesh="flat"),
            interp_method=UxPiecewiseConstantFace,
        ),
        V=Field(
            name="V",
            data=ds_fesom_channel.V,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"], mesh="flat"),
            interp_method=UxPiecewiseConstantFace,
        ),
        W=Field(
            name="W",
            data=ds_fesom_channel.W,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["nz"], mesh="flat"),
            interp_method=UxPiecewiseLinearNode,
        ),
    )
    return UVW


def test_fesom_fieldset(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel, uv_fesom_channel.U, uv_fesom_channel.V])
    # Check that the fieldset has the expected properties
    assert (fieldset.U.data == ds_fesom_channel.U).all()
    assert (fieldset.V.data == ds_fesom_channel.V).all()


def test_fesom_in_particleset(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel, uv_fesom_channel.U, uv_fesom_channel.V])

    # Check that the fieldset has the expected properties
    assert (fieldset.U.data == ds_fesom_channel.U).all()
    assert (fieldset.V.data == ds_fesom_channel.V).all()
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.fieldset == fieldset


def test_set_interp_methods(ds_fesom_channel, uv_fesom_channel):
    fieldset = FieldSet([uv_fesom_channel, uv_fesom_channel.U, uv_fesom_channel.V])
    # Check that the fieldset has the expected properties
    assert (fieldset.U.data == ds_fesom_channel.U).all()
    assert (fieldset.V.data == ds_fesom_channel.V).all()

    # Set the interpolation method for each field
    fieldset.U.interp_method = UxPiecewiseConstantFace
    fieldset.V.interp_method = UxPiecewiseConstantFace


def test_fesom2_square_delaunay_uniform_z_coordinate_eval():
    """
    Test the evaluation of a fieldset with a FESOM2 square Delaunay grid and uniform z-coordinate.
    Ensures that the fieldset can be created and evaluated correctly.
    Since the underlying data is constant, we can check that the values are as expected.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"]
    grid = UxGrid(ds.uxgrid, z=ds.coords["nz"], mesh="flat")
    UVW = VectorField(
        name="UVW",
        U=Field(name="U", data=ds.U, grid=grid, interp_method=UxPiecewiseConstantFace),
        V=Field(name="V", data=ds.V, grid=grid, interp_method=UxPiecewiseConstantFace),
        W=Field(name="W", data=ds.W, grid=grid, interp_method=UxPiecewiseLinearNode),
    )
    P = Field(name="p", data=ds.p, grid=grid, interp_method=UxPiecewiseLinearNode)
    fieldset = FieldSet([UVW, P, UVW.U, UVW.V, UVW.W])

    (u, v, w) = fieldset.UVW.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0], apply_conversion=False)
    assert np.allclose([u.item(), v.item(), w.item()], [1.0, 1.0, 0.0], rtol=1e-3, atol=1e-6)

    assert np.isclose(
        fieldset.U.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0]),
        1.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.V.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0]),
        1.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.W.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0]),
        0.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.p.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0]),
        1.0,
        rtol=1e-3,
        atol=1e-6,
    )


def test_fesom2_square_delaunay_antimeridian_eval():
    """
    Test the evaluation of a fieldset with a FESOM2 square Delaunay grid that crosses the antimeridian.
    Ensures that the fieldset can be created and evaluated correctly.
    Since the underlying data is constant, we can check that the values are as expected.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_antimeridian"]
    P = Field(
        name="p",
        data=ds.p,
        grid=UxGrid(ds.uxgrid, z=ds.coords["nz"], mesh="spherical"),
        interp_method=UxPiecewiseLinearNode,
    )
    fieldset = FieldSet([P])

    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[-170.0]), 1.0)
    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[-180.0]), 1.0)
    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[180.0]), 1.0)
    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[170.0]), 1.0)
