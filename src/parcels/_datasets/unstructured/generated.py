import math

import numpy as np
import uxarray as ux
import xarray as xr

T = 2
vmax = 1.0
delta = 0.1
TIME = xr.date_range("2000", "2001", T)


def simple_small_delaunay(nx=10, ny=10):
    """
    Data on a small Delaunay grid. The naming convention of the dataset and grid is consistent with what is
    provided by UXArray when reading in FESOM2 datasets.
    """
    lon, lat = np.meshgrid(np.linspace(0, 1.0, nx, dtype=np.float32), np.linspace(0, 1.0, ny, dtype=np.float32))
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 2, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers

    # mask any point on one of the boundaries
    mask = np.isclose(lon_flat, 0.0) | np.isclose(lon_flat, 1.0) | np.isclose(lat_flat, 0.0) | np.isclose(lat_flat, 1.0)

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional), W (vertical), and P (sea surface height)
    U = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    V = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    W = np.zeros((1, zf.size, uxgrid.n_node), dtype=np.float64)
    P = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)
    # Define Tface, a ficticious tracer field on the face centroids
    Tface = np.zeros((1, zc.size, uxgrid.n_face), dtype=np.float64)

    for i, (x, y) in enumerate(zip(uxgrid.face_lon, uxgrid.face_lat, strict=False)):
        P[0, :, i] = -vmax * delta * (1 - x) * (math.exp(-x / delta) - 1) * np.sin(math.pi * y)
        U[0, :, i] = -vmax * (1 - math.exp(-x / delta) - x) * np.cos(math.pi * y)
        V[0, :, i] = vmax * ((2.0 - x) * math.exp(-x / delta) - 1) * np.sin(math.pi * y)
        Tface[0, :, i] = np.sin(math.pi * y) * np.cos(math.pi * x)

    # Define Tnode, the same ficticious tracer field as above but on the face corner vertices
    Tnode = np.zeros((1, zc.size, uxgrid.n_node), dtype=np.float64)
    for i, (x, y) in enumerate(zip(uxgrid.node_lon, uxgrid.node_lat, strict=False)):
        Tnode[0, :, i] = np.sin(math.pi * y) * np.cos(math.pi * x)

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="W",
        uxgrid=uxgrid,
        dims=["time", "zf", "n_node"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zf=(["zf"], zf),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="face", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    tface = ux.UxDataArray(
        data=Tface,
        name="T_face",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="Tracer field sampled on face centers",
            units="None",
            location="face",
            mesh="delaunay",
            Conventions="UGRID-1.0",
        ),
    )
    tnode = ux.UxDataArray(
        data=Tnode,
        name="T_node",
        uxgrid=uxgrid,
        dims=["time", "zc", "n_node"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            zc=(["zc"], zc),
        ),
        attrs=dict(
            description="Tracer field sampled on face vertices",
            units="None",
            location="node",
            mesh="delaunay",
            Conventions="UGRID-1.0",
        ),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p, "T_face": tface, "T_node": tnode}, uxgrid=uxgrid)
