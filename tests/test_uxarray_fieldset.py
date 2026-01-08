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
    UxConstantFaceConstantZC,
    UxLinearNodeLinearZF,
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
    ds = ds.rename(
        {
            "nz": "zf",  # Vertical Interface
            "nz1": "zc",  # Vertical Center
        }
    ).set_index(zf="zf", zc="zc")
    return ds


@pytest.fixture
def uv_fesom_channel(ds_fesom_channel) -> VectorField:
    UV = VectorField(
        name="UV",
        U=Field(
            name="U",
            data=ds_fesom_channel.U,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["zf"], mesh="flat"),
            interp_method=UxConstantFaceConstantZC,
        ),
        V=Field(
            name="V",
            data=ds_fesom_channel.V,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["zf"], mesh="flat"),
            interp_method=UxConstantFaceConstantZC,
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
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["zf"], mesh="flat"),
            interp_method=UxConstantFaceConstantZC,
        ),
        V=Field(
            name="V",
            data=ds_fesom_channel.V,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["zf"], mesh="flat"),
            interp_method=UxConstantFaceConstantZC,
        ),
        W=Field(
            name="W",
            data=ds_fesom_channel.W,
            grid=UxGrid(ds_fesom_channel.uxgrid, z=ds_fesom_channel.coords["zf"], mesh="flat"),
            interp_method=UxLinearNodeLinearZF,
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
    fieldset.U.interp_method = UxConstantFaceConstantZC
    fieldset.V.interp_method = UxConstantFaceConstantZC


def test_fesom2_square_delaunay_uniform_z_coordinate_eval():
    """
    Test the evaluation of a fieldset with a FESOM2 square Delaunay grid and uniform z-coordinate.
    Ensures that the fieldset can be created and evaluated correctly.
    Since the underlying data is constant, we can check that the values are as expected.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"]
    ds = ds.rename(
        {
            "nz": "zf",  # Vertical Interface
            "nz1": "zc",  # Vertical Center
        }
    ).set_index(zf="zf", zc="zc")
    grid = UxGrid(ds.uxgrid, z=ds.coords["zf"], mesh="flat")
    UVW = VectorField(
        name="UVW",
        U=Field(name="U", data=ds.U, grid=grid, interp_method=UxConstantFaceConstantZC),
        V=Field(name="V", data=ds.V, grid=grid, interp_method=UxConstantFaceConstantZC),
        W=Field(name="W", data=ds.W, grid=grid, interp_method=UxLinearNodeLinearZF),
    )
    P = Field(name="p", data=ds.p, grid=grid, interp_method=UxLinearNodeLinearZF)
    fieldset = FieldSet([UVW, P, UVW.U, UVW.V, UVW.W])

    assert np.isclose(
        fieldset.U.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0], applyConversion=False),
        1.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.V.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0], applyConversion=False),
        1.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.W.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0], applyConversion=False),
        0.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.p.eval(time=[0.0], z=[1.0], y=[30.0], x=[30.0], applyConversion=False),
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
    ds = datasets_unstructured["fesom2_square_delaunay_antimeridian"].copy(deep=True)
    ds = ds.rename(
        {
            "nz": "zf",  # Vertical Interface
            "nz1": "zc",  # Vertical Center
        }
    ).set_index(zf="zf", zc="zc")
    P = Field(
        name="p",
        data=ds.p,
        grid=UxGrid(ds.uxgrid, z=ds.coords["zf"], mesh="spherical"),
        interp_method=UxLinearNodeLinearZF,
    )
    fieldset = FieldSet([P])

    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[-170.0], applyConversion=False), 1.0)
    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[-180.0], applyConversion=False), 1.0)
    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[180.0], applyConversion=False), 1.0)
    assert np.isclose(fieldset.p.eval(time=[0], z=[1.0], y=[30.0], x=[170.0], applyConversion=False), 1.0)


def test_icon_evals():
    ds = datasets_unstructured["icon_square_delaunay_uniform_z_coordinate"].copy(deep=True)
    fieldset = FieldSet.from_icon(ds)

    # Query points, are chosen to be just a fraction off from the center of a cell for testing
    # This generic dataset has an effective lateral grid-spacing of 3 degrees and vertical grid
    # spacing of 100m - shifting by 1/10 of a degree laterally and 10m vertically should keep us
    # within the cell and make for easy exactness checking of constant and linear interpolation
    xc = ds.uxgrid.face_lon.values
    yc = ds.uxgrid.face_lat.values
    zc = 0.0 * xc + ds.depth.values[1]  # Make zc the same length as xc

    tq = 0.0 * xc
    xq = xc + 0.1
    yq = yc + 0.1
    zq = zc + 10.0

    # The exact function for U is U=z*x . The U variable is center registered both laterally and
    # vertically. In this case, piecewise constant interpolation is expected in both directions.
    # The expected value for interpolation is then just computed using the cell center locations
    assert np.allclose(fieldset.U.eval(time=tq, z=zq, y=yq, x=xq, applyConversion=False), zc * xc)

    # The exact function for V is V=z*y . The V variable is center registered both laterally and
    # vertically. In this case, piecewise constant interpolation is expected in both directions
    # The expected value for interpolation is then just computed using the cell center locations
    assert np.allclose(fieldset.V.eval(time=tq, z=zq, y=yq, x=xq, applyConversion=False), zc * yc)

    # The exact function for W is W=z*x*y . The W variable is center registered laterally and
    # interface registered vertically. In this case, piecewise constant interpolation is expected
    # laterally, while piecewise linear is expected vertically.
    # The expected value for interpolation is then just computed using the cell center locations
    # for the latitude and longitude, and the query point for the vertical interpolation
    assert np.allclose(fieldset.W.eval(time=tq, z=zq, y=yq, x=xq, applyConversion=False), zq * yc * xc)

    # The exact function for P is P=0.0001*z*x*y . The P variable is node registered laterally and
    # center registered vertically. In this case, piecewise linear interpolation is expected
    # laterally and piecewise constant is expected vertically
    # The expected value for interpolation is then just computed using query point locations
    # for the latitude and longitude, and the layer centers vertically.
    assert np.allclose(
        fieldset.p.eval(time=tq, z=zq, y=yq, x=xq, applyConversion=False), 0.0001 * zc * xq * yq, rtol=1e-2
    )
