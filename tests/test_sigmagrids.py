import numpy as np
import xarray as xr

import parcels
from parcels import Particle, ParticleSet, Variable
from parcels.kernels import AdvectionRK4_3D_CROCO, SampleOmegaCroco, convert_z_to_sigma_croco


def test_conversion_3DCROCO():
    """Test of the conversion from depth to sigma in CROCO

    Values below are retrieved using xroms and hardcoded in the method (to avoid dependency on xroms):
    ```py
    x, y = 10, 20
    s_xroms = ds.s_w.values
    z_xroms = ds.z_w.isel(time=0).isel(eta_rho=y).isel(xi_rho=x).values
    lat, lon = ds.y_rho.values[y, x], ds.x_rho.values[y, x]
    ```
    """
    data_folder = parcels.download_example_dataset("CROCOidealized_data")
    ds_fields = xr.open_dataset(data_folder / "CROCO_idealized.nc")
    fields = {
        "U": ds_fields["u"],
        "V": ds_fields["v"],
        "W": ds_fields["w"],
        "h": ds_fields["h"],
        "zeta": ds_fields["zeta"],
        "Cs_w": ds_fields["Cs_w"],
    }

    ds_fset = parcels.convert.croco_to_sgrid(fields=fields, coords=ds_fields)

    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
    fieldset.add_constant("hc", ds_fields.hc.item())

    s_xroms = np.array([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0], dtype=np.float32)
    z_xroms = np.array([-1.26000000e02, -1.10585846e02, -9.60985413e01, -8.24131317e01, -6.94126511e01, -5.69870148e01, -4.50318756e01, -3.34476166e01, -2.21383114e01, -1.10107975e01, 2.62768921e-02,], dtype=np.float32,)  # fmt: skip

    time = np.zeros_like(z_xroms)
    lon = np.full_like(z_xroms, 38000.0)
    lat = np.full_like(z_xroms, 78000.0)

    sigma = convert_z_to_sigma_croco(fieldset, time, z_xroms, lat, lon, None)

    np.testing.assert_allclose(sigma, s_xroms, atol=1e-3)


def test_advection_3DCROCO():
    data_folder = parcels.download_example_dataset("CROCOidealized_data")
    ds_fields = xr.open_dataset(data_folder / "CROCO_idealized.nc")
    ds_fields.load()

    fields = {
        "U": ds_fields["u"],
        "V": ds_fields["v"],
        "W": ds_fields["w"],
        "h": ds_fields["h"],
        "zeta": ds_fields["zeta"],
        "Cs_w": ds_fields["Cs_w"],
        "omega": ds_fields["omega"],
    }

    ds_fset = parcels.convert.croco_to_sgrid(fields=fields, coords=ds_fields)

    fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fset)
    fieldset.add_constant("hc", ds_fields.hc.item())

    runtime = 10_000
    X, Z = np.meshgrid([40e3, 80e3, 120e3], [-10, -130])
    Y = np.ones(X.size) * 100e3

    pclass = Particle.add_variable(Variable("omega"))
    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=X, lat=Y, z=Z)

    pset.execute(
        [AdvectionRK4_3D_CROCO, SampleOmegaCroco], runtime=np.timedelta64(runtime, "s"), dt=np.timedelta64(100, "s")
    )
    np.testing.assert_allclose(pset.z, Z.flatten(), atol=5)  # TODO lower this atol
    np.testing.assert_allclose(pset.lon, [x + runtime for x in X.flatten()], atol=1e-3)
