from pathlib import Path

import numpy as np
import pytest
import uxarray as ux

import parcels._datasets.remote as _parcels_remote
import parcels.tutorial
from parcels import (
    FieldSet,
    convert,
)
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.interpolators import (
    UxConstantFaceConstantZC,
    UxLinearNodeLinearZF,
)
from tests.utils import create_uxgrid_from_triangulation


@pytest.fixture
def ds_fesom_channel() -> ux.UxDataset:
    # Download FESOM files via the new tutorial API
    parcels.tutorial.open_dataset("FESOM_periodic_channel/fesom_channel")
    # uxarray requires file paths; access the downloaded files from the pooch cache
    _fesom_dir = Path(_parcels_remote._DATA_HOME) / "data" / "FESOM_periodic_channel"
    grid_path = str(_fesom_dir / "fesom_channel.nc")
    data_path = [
        str(_fesom_dir / "u.fesom_channel.nc"),
        str(_fesom_dir / "v.fesom_channel.nc"),
        str(_fesom_dir / "w.fesom_channel.nc"),
    ]
    ds = ux.open_mfdataset(grid_path, data_path).rename_vars({"u": "U", "v": "V", "w": "W"})
    ds = convert.fesom_to_ugrid(ds)
    return ds


@pytest.fixture
def fieldset_fesom_channel(ds_fesom_channel):
    return FieldSet.from_ugrid_conventions(ds_fesom_channel)


def test_fesom_fieldset(ds_fesom_channel, fieldset_fesom_channel):
    # Check that the fieldset has the expected properties
    assert (fieldset_fesom_channel.U.data == ds_fesom_channel.U).all()
    assert (fieldset_fesom_channel.V.data == ds_fesom_channel.V).all()


@pytest.mark.xfail(reason="#2674 - 'p' interpolator is not being selected properly")
def test_fesom2_square_delaunay_uniform_z_coordinate_eval():
    """
    Test the evaluation of a fieldset with a FESOM2 square Delaunay grid and uniform z-coordinate.
    Ensures that the fieldset can be created and evaluated correctly.
    Since the underlying data is constant, we can check that the values are as expected.
    """
    ds = datasets_unstructured["fesom2_square_delaunay_uniform_z_coordinate"]
    ds = convert.fesom_to_ugrid(ds)
    fieldset = FieldSet.from_ugrid_conventions(ds)

    assert isinstance(fieldset.U.interp_method, UxConstantFaceConstantZC)
    assert isinstance(fieldset.V.interp_method, UxConstantFaceConstantZC)
    assert isinstance(fieldset.W.interp_method, UxLinearNodeLinearZF)
    assert isinstance(fieldset.p.interp_method, UxLinearNodeLinearZF)

    (u, v, w) = fieldset.UVW.eval(t=[0.0], z=[1.0], y=[30.0], x=[30.0])
    assert np.allclose([u.item(), v.item(), w.item()], [1.0, 1.0, 0.0], rtol=1e-3, atol=1e-6)

    assert np.isclose(
        fieldset.U.eval(t=[0.0], z=[1.0], y=[30.0], x=[30.0]),
        1.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.V.eval(t=[0.0], z=[1.0], y=[30.0], x=[30.0]),
        1.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.W.eval(t=[0.0], z=[1.0], y=[30.0], x=[30.0]),
        0.0,
        rtol=1e-3,
        atol=1e-6,
    )
    assert np.isclose(
        fieldset.p.eval(t=[0.0], z=[1.0], y=[30.0], x=[30.0]),
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
    ds = convert.fesom_to_ugrid(ds)
    fieldset = FieldSet.from_ugrid_conventions(ds)
    fieldset.p.interp_method = UxLinearNodeLinearZF()

    assert np.isclose(fieldset.p.eval(t=[0], z=[1.0], y=[30.0], x=[-170.0]), 1.0)
    assert np.isclose(fieldset.p.eval(t=[0], z=[1.0], y=[30.0], x=[-180.0]), 1.0)
    assert np.isclose(fieldset.p.eval(t=[0], z=[1.0], y=[30.0], x=[180.0]), 1.0)
    assert np.isclose(fieldset.p.eval(t=[0], z=[1.0], y=[30.0], x=[170.0]), 1.0)


def test_icon_evals():
    ds = datasets_unstructured["icon_square_delaunay_uniform_z_coordinate"].copy(deep=True)
    ds = convert.icon_to_ugrid(ds)
    fieldset = FieldSet.from_ugrid_conventions(ds, mesh="flat")

    # Query points, are chosen to be just a fraction off from the center of a cell for testing
    # This generic dataset has an effective lateral grid-spacing of 3 degrees and vertical grid
    # spacing of 100m - shifting by 1/10 of a degree laterally and 10m vertically should keep us
    # within the cell and make for easy exactness checking of constant and linear interpolation
    xc = ds.uxgrid.face_lon.values
    yc = ds.uxgrid.face_lat.values
    zc = 0.0 * xc + ds.zc.values[1]  # Make zc the same length as xc

    tq = 0.0 * xc
    xq = xc + 0.1
    yq = yc + 0.1
    zq = zc + 10.0

    # The exact function for U is U=z*x . The U variable is center registered both laterally and
    # vertically. In this case, piecewise constant interpolation is expected in both directions.
    # The expected value for interpolation is then just computed using the cell center locations
    assert np.allclose(fieldset.U.eval(t=tq, z=zq, y=yq, x=xq), zc * xc)

    # The exact function for V is V=z*y . The V variable is center registered both laterally and
    # vertically. In this case, piecewise constant interpolation is expected in both directions
    # The expected value for interpolation is then just computed using the cell center locations
    assert np.allclose(fieldset.V.eval(t=tq, z=zq, y=yq, x=xq), zc * yc)

    # The exact function for W is W=z*x*y . The W variable is center registered laterally and
    # interface registered vertically. In this case, piecewise constant interpolation is expected
    # laterally, while piecewise linear is expected vertically.
    # The expected value for interpolation is then just computed using the cell center locations
    # for the latitude and longitude, and the query point for the vertical interpolation
    assert np.allclose(fieldset.W.eval(t=tq, z=zq, y=yq, x=xq), zq * yc * xc)

    # The exact function for P is P=z*(x+y) . The P variable is node registered laterally and
    # center registered vertically. In this case, barycentric interpolation is expected
    # laterally and piecewise constant is expected vertically
    # Since barycentric interpolation is exact for functions f=a*x+b*y laterally, the expected
    # value for interpolation is then just computed using query point locations
    # for the latitude and longitude, and the layer centers vertically.
    assert np.allclose(fieldset.p.eval(t=tq, z=zq, y=yq, x=xq), zc * (xq + yq))


# Constrained (PSLG) Delaunay triangulation of the three nested polygons in
# docs/user_guide/examples/tutorial_nestedgrids.ipynb, as produced by
# triangle.triangulate({"vertices": ..., "segments": ...}, "p").
#
# Baked in as literals rather than recomputed: py-triangle is only in the "notebooks"
# pixi feature, not "test", and the notebook's geometry is what #2878 is about, so
# pinning it also stops the reproducer drifting if the triangulator changes.
_NESTEDGRIDS_NODES = np.array(
    [
        [10.0, 15.0],
        [25.0, 10.0],
        [25.0, 25.0],
        [17.0, 36.0],
        [10.0, 32.0],
        [0.0, -5.0],
        [35.0, 0.0],
        [35.0, 25.0],
        [0.0, 20.0],
        [-10.0, -20.0],
        [60.0, -20.0],
        [60.0, 40.0],
        [-10.0, 40.0],
        [25.0, 165.0 / 7.0],  # Steiner point added by the triangulator
        [10.0, 150.0 / 7.0],  # Steiner point added by the triangulator
    ]
)
_NESTEDGRIDS_FACES = np.array(
    [
        [8, 9, 5], [5, 9, 6], [0, 5, 1], [8, 5, 0], [12, 8, 4], [8, 12, 9],
        [14, 8, 0], [12, 4, 3], [13, 14, 0], [14, 2, 4], [7, 1, 6], [6, 1, 5],
        [10, 6, 9], [11, 6, 10], [3, 2, 7], [3, 4, 2], [7, 11, 3], [11, 7, 6],
        [13, 1, 7], [3, 11, 12], [4, 8, 14], [13, 0, 1], [2, 14, 13], [7, 2, 13],
    ]
)  # fmt: skip


@pytest.mark.xfail(reason="#2878 - see tests/test_spatialhash.py and tests/test_index_search.py for the two causes")
def test_nestedgrids_notebook_triangulation_spherical_search():
    """Every particle in tutorial_nestedgrids.ipynb must be located on a spherical mesh.

    The notebook reduced to the grid-search step: same triangulation, same 500
    positions, no fields or kernel. Covers the reported symptom; the two underlying
    defects are covered separately by the tests named in the xfail reason.
    """
    grid = create_uxgrid_from_triangulation(
        _NESTEDGRIDS_NODES[:, 0], _NESTEDGRIDS_NODES[:, 1], _NESTEDGRIDS_FACES, mesh="spherical"
    )

    x, y = np.meshgrid(np.linspace(-8, 58, 25), np.linspace(-18, 38, 20))
    x = x.ravel()
    y = y.ravel()
    z = np.zeros_like(x)

    face = grid.search(z, y, x)["FACE"]["index"]
    n_lost = int(np.count_nonzero(face < 0))
    assert n_lost == 0, (
        f"grid search failed for {n_lost} of {len(x)} particles; e.g. (lon, lat)="
        f"{np.column_stack((x, y))[face < 0][:3].tolist()}"
    )
