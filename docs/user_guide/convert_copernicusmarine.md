# 🖥️ Copernicus Marine Service tutorial

Parcels is closely aligned with the Copernicus Marine Service for oceanographic data. This tutorial will show you how to use the [`copernicusmarine` toolbox](https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction) and {py:func}`parcels.convert.copernicusmarine_to_sgrid` to stream or download (global) ocean data and use it to advect particles.

```{note}
Because the `copernicusmarine` toolbox requires a log-in, this tutorial is not interactive.
You will need to have access to the Copernicus Marine Service and be logged in to reproduce this tutorial yourself.

If you think anything in this notebook isn't working as expected, please [create an Issue or PR](https://github.com/Parcels-code/Parcels/issues).
```

First, create a dataset object using the `copernicusmarine` toolbox, which you can install with conda, pip or pixi. Here, we use the global ocean analysis and forecast dataset.

```
import copernicusmarine
import numpy as np
import xarray as xr
import parcels

copernicus_dict = {
    "start_datetime": "2024-01-01",
    "end_datetime": "2024-12-31",
    "minimum_depth": 0,
    "maximum_depth": 200,
    "minimum_longitude": -20,
    "maximum_longitude": 20,
    "minimum_latitude": -20,
    "maximum_latitude": 20,

    # Chunking that seems to work best for Parcels
    "service": "arco-geo-series",
    "chunk_size_limit": 1,
}

ds_uv = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
    variables=["uo", "vo"],
    **copernicus_dict
)
ds_w = copernicusmarine.open_dataset(
    dataset_id="cmems_mod_glo_phy-wcur_anfc_0.083deg_P1D-m",
    variables=["wo"],
    **copernicus_dict
)
ds = xr.merge([ds_uv, ds_w])
```

Of course, feel free to adjust the minimum and maximum longitude, latitude and depths according to your area of interest - as well as the start and end times.

```{note}
If you have a global domain, it is useful to wrap the longitudes, so Parcels can seamlessly advect particles across the dateline. Do that with
```

```
# Only use this code if you have a global domain
ds = xr.concat(
    [
        ds,
        ds.isel(longitude=slice(0, 1)).assign_coords(
            longitude=ds.longitude.isel(longitude=slice(0, 1)) + 360
        ),
    ],
    dim="longitude",
)
```

Once you have this dataset, you can create a Parcels FieldSet using the {py:func}`parcels.convert.copernicusmarine_to_sgrid()` function. This will convert the Copernicus Marine Service dataset into a format that Parcels can use for particle advection.

```
ds_fields = parcels.convert.copernicusmarine_to_sgrid(
    fields={"U": ds["uo"], "V": ds["vo"], "W": ds["wo"]}
)
fieldset = parcels.FieldSet.from_sgrid_conventions(ds_fields)
```

Now, you will have to set the interpolation method for the UV fields to `XFreeslip`, which is more suitable for A-Grid data like the one provided by Copernicus Marine Service. This ensures that particles don't get stuck near the boundaries. See the [preventing stuck particles tutorial](./interpolation_unstuck_Agrid.ipynb) for more information.

```
fieldset.UV.interp_method = parcels.interpolators.XFreeslip()
```

For [better performance](./simulation_performance.md), we should also use the `ChunkCachedArray` backend

```
fieldset.to_chunk_cached_arrays()
```

```{note}
While Parcels works with "streamed" data from the Copernicus Marine Data Store, streaming the data can be rather slow. It is often more efficient to first download the data and then use it locally. You can do that by (before you create the fieldset) doing a `ds.to_zarr("local_store.zarr")` and then opening the local store with `ds = xr.open_zarr("local_store.zarr")`.
```

We can then run particles as usual in Parcels, by defining a `ParticleSet` and then executing it with the desired advection kernel and runtime.

```
lon, lat = np.meshgrid(np.arange(-10, 10, 2), np.arange(-10, 10, 2))
z = np.full_like(lon, 5)  # Start all particles at 5m depth
pset = parcels.ParticleSet(fieldset, x=lon, y=lat, z=z)
pset.execute(
  parcels.kernels.AdvectionRK2_3D,
  runtime=np.timedelta64(2, "D"),
  dt=np.timedelta64(15, "m")
)
```
