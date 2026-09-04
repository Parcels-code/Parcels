import operator
from functools import reduce
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from zarr.storage import LocalStore

X = 1000
Y = 1000
Z = 50
T = 30

X_SMALL = 200
Y_SMALL = 200

TIME = xr.date_range("2000", "2001", T)


def prod(seq):
    return reduce(operator.mul, seq, 1)


def _rotated_curvilinear_grid():
    XG = np.arange(X)
    YG = np.arange(Y)
    LON, LAT = np.meshgrid(XG, YG)

    angle = -np.pi / 24
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # rotate the LON and LAT grids
    LON, LAT = np.einsum("ji, mni -> jmn", rotation, np.dstack([LON, LAT]))

    return xr.Dataset(
        {
            "data_g": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "data_c": (
                ["time", "ZC", "YC", "XC"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "U_A_grid": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "V_A_grid": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "U_C_grid": (
                ["time", "ZG", "YC", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "V_C_grid": (
                ["time", "ZG", "YG", "XC"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
        },
        coords={
            "XG": (["XG"], XG, {"axis": "X", "c_grid_axis_shift": -0.5}),
            "YG": (["YG"], YG, {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], XG + 0.5, {"axis": "X"}),
            "YC": (["YC"], YG + 0.5, {"axis": "Y"}),
            "ZG": (
                ["ZG"],
                np.arange(Z),
                {"axis": "Z", "c_grid_axis_shift": -0.5},
            ),
            "ZC": (
                ["ZC"],
                np.arange(Z) + 0.5,
                {"axis": "Z"},
            ),
            "depth": (["ZG"], np.arange(Z), {"axis": "Z"}),
            "time": (["time"], TIME, {"axis": "T"}),
            "lon": (
                ["YG", "XG"],
                LON,
                {"axis": "X", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
            "lat": (
                ["YG", "XG"],
                LAT,
                {"axis": "Y", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
        },
    )


def random_dask_array(shape, scaling=1):
    return da.random.uniform(scaling, size=prod(shape)).reshape(shape)


def _cartesion_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def _polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _unrolled_cone_curvilinear_grid():
    # Not a great unrolled cone, but this is good enough for testing
    # you can use matplotlib pcolormesh to plot
    XG = np.arange(X)
    YG = np.arange(Y) * 0.25

    pivot = -10, 0
    LON, LAT = np.meshgrid(XG, YG)

    new_lon_lat = []

    min_lon = np.min(XG)
    for lon, lat in zip(LON.flatten(), LAT.flatten(), strict=True):
        r, _ = _cartesion_to_polar(lon - pivot[0], lat - pivot[1])
        _, theta = _cartesion_to_polar(min_lon - pivot[0], lat - pivot[1])
        theta *= 1.2
        r *= 1.2
        lon, lat = _polar_to_cartesian(r, theta)
        new_lon_lat.append((lon + pivot[0], lat + pivot[1]))

    new_lon, new_lat = zip(*new_lon_lat, strict=True)
    LON, LAT = (
        np.array(new_lon).reshape(LON.shape),
        np.array(new_lat).reshape(LAT.shape),
    )

    return xr.Dataset(
        {
            "data_g": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "data_c": (
                ["time", "ZC", "YC", "XC"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "U_A_grid": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "V_A_grid": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "U_C_grid": (
                ["time", "ZG", "YC", "XG"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
            "V_C_grid": (
                ["time", "ZG", "YG", "XC"],
                random_dask_array(scaling=5, shape=(T, Z, Y, X)),
            ),
        },
        coords={
            "XG": (["XG"], XG, {"axis": "X", "c_grid_axis_shift": -0.5}),
            "YG": (["YG"], YG, {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], XG + 0.5, {"axis": "X"}),
            "YC": (["YC"], YG + 0.5, {"axis": "Y"}),
            "ZG": (
                ["ZG"],
                np.arange(Z),
                {"axis": "Z", "c_grid_axis_shift": -0.5},
            ),
            "ZC": (
                ["ZC"],
                np.arange(Z) + 0.5,
                {"axis": "Z"},
            ),
            "depth": (["ZG"], np.arange(Z), {"axis": "Z"}),
            "time": (["time"], TIME, {"axis": "T"}),
            "lon": (
                ["YG", "XG"],
                LON,
                {"axis": "X", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
            "lat": (
                ["YG", "XG"],
                LAT,
                {"axis": "Y", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
        },
    )


def _ds_2d_left(x, y, z, t, time):
    """MITgcm indexing style dataset."""
    return xr.Dataset(
        {
            "data_g": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(t, z, y, x)),
            ),
            "data_c": (
                ["time", "ZC", "YC", "XC"],
                random_dask_array(scaling=5, shape=(t, z, y, x)),
            ),
            "U_A_grid": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(t, z, y, x)),
            ),
            "V_A_grid": (
                ["time", "ZG", "YG", "XG"],
                random_dask_array(scaling=5, shape=(t, z, y, x)),
            ),
            "U_C_grid": (
                ["time", "ZG", "YC", "XG"],
                random_dask_array(scaling=5, shape=(t, z, y, x)),
            ),
            "V_C_grid": (
                ["time", "ZG", "YG", "XC"],
                random_dask_array(scaling=5, shape=(t, z, y, x)),
            ),
        },
        coords={
            "XG": (
                ["XG"],
                2 * np.pi / x * np.arange(0, x),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "XC": (["XC"], 2 * np.pi / x * (np.arange(0, x) + 0.5), {"axis": "X"}),
            "YG": (
                ["YG"],
                2 * np.pi / y * np.arange(0, y),
                {"axis": "Y", "c_grid_axis_shift": -0.5},
            ),
            "YC": (
                ["YC"],
                2 * np.pi / y * (np.arange(0, y) + 0.5),
                {"axis": "Y"},
            ),
            "ZG": (
                ["ZG"],
                np.arange(z),
                {"axis": "Z", "c_grid_axis_shift": -0.5},
            ),
            "ZC": (
                ["ZC"],
                np.arange(z) + 0.5,
                {"axis": "Z"},
            ),
            "lon": (["XG"], 2 * np.pi / x * np.arange(0, x)),
            "lat": (["YG"], 2 * np.pi / y * np.arange(0, y)),
            "depth": (["ZG"], np.arange(z)),
            "time": (["time"], time, {"axis": "T"}),
        },
    )


datasets = {
    "2d_left_rotated": _rotated_curvilinear_grid(),
    "ds_2d_left": _ds_2d_left(X, Y, Z, T, TIME),
    "2d_left_unrolled_cone": _unrolled_cone_curvilinear_grid(),
}


def save(ds: xr.Dataset, path: str, chunks: dict) -> None:
    """Save dataset to zarr with specified chunking."""
    store = LocalStore(path)
    ds.chunk(chunks).to_zarr(store, mode="w", encoding=None, consolidated=False)
    size_mb = sum(ds[v].nbytes for v in ds.data_vars) / 1e6
    print(f"  {path}  dims={dict(ds.sizes)}  ~{size_mb:.0f} MB uncompressed")


if __name__ == "__main__":
    dataset_path = "datasets/ds_2d_left_agrid.zarr"
    print("Generating ds_2d_left...")
    if Path(dataset_path).exists():
        print(f"Dataset {dataset_path} already exists")
    else:
        save(
            datasets["ds_2d_left"][["U_A_grid", "V_A_grid"]],
            dataset_path,
            {"time": 15, "XG": 40, "YG": 40, "ZG": 8},
        )

    dataset_path_small = "datasets/ds_2d_left_agrid_small.zarr"
    print("Generating ds_2d_left_small...")
    if Path(dataset_path_small).exists():
        print(f"Dataset {dataset_path_small} already exists")
    else:
        save(
            _ds_2d_left(X_SMALL, Y_SMALL, Z, T, TIME)[["U_A_grid", "V_A_grid"]],
            dataset_path_small,
            {"time": 15, "XG": 40, "YG": 40, "ZG": 8},
        )
