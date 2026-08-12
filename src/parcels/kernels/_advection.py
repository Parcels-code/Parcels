"""Collection of pre-built advection kernels."""

import numpy as np

from parcels._core.statuscodes import StatusCode
from parcels.interpolators._xinterpolators import _get_cgrid_velocities

__all__ = [
    "AdvectionAnalytical",
    "AdvectionEE",
    "AdvectionRK2",
    "AdvectionRK2_3D",
    "AdvectionRK4",
    "AdvectionRK4_3D",
    "AdvectionRK45",
]


def AdvectionRK2(particles, fieldset):  # pragma: no cover
    """Advection of particles using second-order Runge-Kutta integration."""
    (u1, v1) = fieldset.UV[particles]
    x1 = particles.x + u1 * 0.5 * particles.dt
    y1 = particles.y + v1 * 0.5 * particles.dt
    (u2, v2) = fieldset.UV[particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles]
    particles.dx += u2 * particles.dt
    particles.dy += v2 * particles.dt


def AdvectionRK2_3D(particles, fieldset):  # pragma: no cover
    """Advection of particles using second-order Runge-Kutta integration including vertical velocity."""
    (u1, v1, w1) = fieldset.UVW[particles]
    x1 = particles.x + u1 * 0.5 * particles.dt
    y1 = particles.y + v1 * 0.5 * particles.dt
    z1 = particles.z + w1 * 0.5 * particles.dt
    (u2, v2, w2) = fieldset.UVW[particles.t + 0.5 * particles.dt, z1, y1, x1, particles]
    particles.dx += u2 * particles.dt
    particles.dy += v2 * particles.dt
    particles.dz += w2 * particles.dt


def AdvectionRK4(particles, fieldset):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration."""
    (u1, v1) = fieldset.UV[particles]
    x1 = particles.x + u1 * 0.5 * particles.dt
    y1 = particles.y + v1 * 0.5 * particles.dt
    (u2, v2) = fieldset.UV[particles.t + 0.5 * particles.dt, particles.z, y1, x1, particles]
    x2 = particles.x + u2 * 0.5 * particles.dt
    y2 = particles.y + v2 * 0.5 * particles.dt
    (u3, v3) = fieldset.UV[particles.t + 0.5 * particles.dt, particles.z, y2, x2, particles]
    x3 = particles.x + u3 * particles.dt
    y3 = particles.y + v3 * particles.dt
    (u4, v4) = fieldset.UV[particles.t + particles.dt, particles.z, y3, x3, particles]
    particles.dx += (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * particles.dt
    particles.dy += (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * particles.dt


def AdvectionRK4_3D(particles, fieldset):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity."""
    (u1, v1, w1) = fieldset.UVW[particles]
    x1 = particles.x + u1 * 0.5 * particles.dt
    y1 = particles.y + v1 * 0.5 * particles.dt
    z1 = particles.z + w1 * 0.5 * particles.dt
    (u2, v2, w2) = fieldset.UVW[particles.t + 0.5 * particles.dt, z1, y1, x1, particles]
    x2 = particles.x + u2 * 0.5 * particles.dt
    y2 = particles.y + v2 * 0.5 * particles.dt
    z2 = particles.z + w2 * 0.5 * particles.dt
    (u3, v3, w3) = fieldset.UVW[particles.t + 0.5 * particles.dt, z2, y2, x2, particles]
    x3 = particles.x + u3 * particles.dt
    y3 = particles.y + v3 * particles.dt
    z3 = particles.z + w3 * particles.dt
    (u4, v4, w4) = fieldset.UVW[particles.t + particles.dt, z3, y3, x3, particles]
    particles.dx += (u1 + 2 * u2 + 2 * u3 + u4) / 6 * particles.dt
    particles.dy += (v1 + 2 * v2 + 2 * v3 + v4) / 6 * particles.dt
    particles.dz += (w1 + 2 * w2 + 2 * w3 + w4) / 6 * particles.dt


def AdvectionEE(particles, fieldset):  # pragma: no cover
    """Advection of particles using Explicit Euler (aka Euler Forward) integration."""
    (u1, v1) = fieldset.UV[particles]
    particles.dx += u1 * particles.dt
    particles.dy += v1 * particles.dt


def AdvectionRK45(particles, fieldset):  # pragma: no cover
    """Advection of particles using adaptive Runge-Kutta 4/5 integration.

    Note that this kernel requires a FieldSet with constants 'RK45_tol' (in meters),
    'RK45_min_dt' (in seconds) and 'RK45_max_dt' (in seconds).

    Time-step dt is halved if error is larger than fieldset.RK45_tol,
    and doubled if error is smaller than 1/10th of tolerance.
    """
    sign_dt = np.sign(particles.dt)

    c = [1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0]
    A = [
        [1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0],
        [1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0],
        [439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0],
        [-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0],
    ]
    b4 = [25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0]
    b5 = [16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0]

    (u1, v1) = fieldset.UV[particles]
    x1 = particles.x + u1 * A[0][0] * particles.dt
    y1 = particles.y + v1 * A[0][0] * particles.dt
    (u2, v2) = fieldset.UV[particles.t + c[0] * particles.dt, particles.z, y1, x1, particles]
    x2 = particles.x + (u1 * A[1][0] + u2 * A[1][1]) * particles.dt
    y2 = particles.y + (v1 * A[1][0] + v2 * A[1][1]) * particles.dt
    (u3, v3) = fieldset.UV[particles.t + c[1] * particles.dt, particles.z, y2, x2, particles]
    x3 = particles.x + (u1 * A[2][0] + u2 * A[2][1] + u3 * A[2][2]) * particles.dt
    y3 = particles.y + (v1 * A[2][0] + v2 * A[2][1] + v3 * A[2][2]) * particles.dt
    (u4, v4) = fieldset.UV[particles.t + c[2] * particles.dt, particles.z, y3, x3, particles]
    x4 = particles.x + (u1 * A[3][0] + u2 * A[3][1] + u3 * A[3][2] + u4 * A[3][3]) * particles.dt
    y4 = particles.y + (v1 * A[3][0] + v2 * A[3][1] + v3 * A[3][2] + v4 * A[3][3]) * particles.dt
    (u5, v5) = fieldset.UV[particles.t + c[3] * particles.dt, particles.z, y4, x4, particles]
    x5 = particles.x + (u1 * A[4][0] + u2 * A[4][1] + u3 * A[4][2] + u4 * A[4][3] + u5 * A[4][4]) * particles.dt
    y5 = particles.y + (v1 * A[4][0] + v2 * A[4][1] + v3 * A[4][2] + v4 * A[4][3] + v5 * A[4][4]) * particles.dt
    (u6, v6) = fieldset.UV[particles.t + c[4] * particles.dt, particles.z, y5, x5, particles]

    x_4th = (u1 * b4[0] + u2 * b4[1] + u3 * b4[2] + u4 * b4[3] + u5 * b4[4]) * particles.dt
    y_4th = (v1 * b4[0] + v2 * b4[1] + v3 * b4[2] + v4 * b4[3] + v5 * b4[4]) * particles.dt
    x_5th = (u1 * b5[0] + u2 * b5[1] + u3 * b5[2] + u4 * b5[3] + u5 * b5[4] + u6 * b5[5]) * particles.dt
    y_5th = (v1 * b5[0] + v2 * b5[1] + v3 * b5[2] + v4 * b5[3] + v5 * b5[4] + v6 * b5[5]) * particles.dt

    kappa = np.sqrt(np.pow(x_5th - x_4th, 2) + np.pow(y_5th - y_4th, 2))

    good_particles = (kappa <= fieldset.RK45_tol) | (np.fabs(particles.dt) <= np.fabs(fieldset.RK45_min_dt))
    particles.dx += np.where(good_particles, x_5th, 0)
    particles.dy += np.where(good_particles, y_5th, 0)

    increase_dt_particles = (
        good_particles
        & (kappa <= fieldset.RK45_tol / 10)
        & (np.fabs(particles.dt * 2) <= np.fabs(fieldset.RK45_max_dt))
    )
    particles.next_dt = np.where(increase_dt_particles, particles.dt * 2, particles.dt)
    particles.next_dt = np.where(
        np.abs(particles.next_dt) > np.abs(fieldset.RK45_max_dt),
        fieldset.RK45_max_dt * sign_dt,
        particles.next_dt,
    )
    particles.state = np.where(good_particles, StatusCode.Evaluate, particles.state)

    repeat_particles = np.invert(good_particles)
    particles.dt = np.where(repeat_particles, particles.dt / 2, particles.dt)
    particles.dt = np.where(
        np.abs(particles.dt) < np.abs(fieldset.RK45_min_dt),
        fieldset.RK45_min_dt * sign_dt,
        particles.dt,
    )
    particles.state = np.where(repeat_particles, StatusCode.Repeat, particles.state)


def AdvectionAnalytical(particles, fieldset):  # pragma: no cover
    """Advection of particles using 'analytical advection' integration.

    Based on Ariane/TRACMASS algorithm, as detailed in e.g. Doos et al (https://doi.org/10.5194/gmd-10-1733-2017).
    Note that the time-dependent scheme is currently implemented with 'intermediate timesteps'
    (default 10 per model timestep) and not yet with the full analytical time integration.
    """
    import parcels._core.utils.interpolation as i_u
    from parcels._core.field import _get_positions

    tol = 1e-10
    # I_s = 10  # number of intermediate time steps
    dt = particles.dt
    direction = 1.0 if dt > 0 else -1.0
    withW = True if "W" in [f.name for f in fieldset.fields.values()] else False

    vectorfield = fieldset.UVW if withW else fieldset.UV
    # withTime = True if len(vectorfield.grid.time) > 1 else False
    igrid = vectorfield.igrid
    grid = vectorfield.grid

    _, grid_positions = _get_positions(
        fieldset.U, particles.t, particles.z, particles.y, particles.x, particles, particles.ei[:, igrid]
    )
    _, xsi = grid_positions["X"]["index"], grid_positions["X"]["bcoord"]
    _, eta = grid_positions["Y"]["index"], grid_positions["Y"]["bcoord"]
    zi, zeta = grid_positions["Z"]["index"], grid_positions["Z"]["bcoord"]
    U0, U1, V0, V1, W0, W1, px, py = _get_cgrid_velocities(vectorfield, grid_positions)

    if withW:
        pz = np.array([grid.depth[zi], grid.depth[zi + 1]])
        dz = pz[1] - pz[0]
    else:
        dz = 1.0

    rad = np.pi / 180.0
    deg2m = 1852 * 60.0
    meshJac = (deg2m * deg2m * np.cos(rad * particles.y)) if grid._mesh.is_spherical() else 1
    dxdy = i_u._compute_jacobian_determinant(py, px, eta, xsi) * meshJac

    U0 *= direction * dz
    U1 *= direction * dz
    V0 *= direction * dz
    V1 *= direction * dz
    if withW:
        W0 *= direction * dxdy
        W1 *= direction * dxdy

    def compute_ds(F0, F1, r, direction, tol):  # noqa: N803
        with np.errstate(divide="ignore", invalid="ignore"):
            print("NOW IN COMPUTE_DS")
            print(F0, F1, r, direction)
            up = F0 * (1 - r) + F1 * r
            r_target = np.where(direction * up >= 0.0, 1.0, 0.0)
            B = F0 - F1
            delta = -F0
            B = np.where(np.abs(B) < tol, np.zeros_like(B), B)

            F_r1 = np.where(np.abs(B) > tol, r_target + delta / B, np.nan)
            F_r0 = np.where(np.abs(B) > tol, r + delta / B, np.nan)

            d_s = -1.0 / B * np.log(F_r1 / F_r0)
            d_s = np.where(F_r1 * F_r0 < tol, np.inf, d_s)
            d_s = np.where(B == 0, -delta * direction / up, d_s)
            d_s = np.where((np.abs(B) < tol) & (np.abs(delta) < tol), np.inf, d_s)

            d_s = np.where(d_s < tol, np.inf, d_s)
        return d_s, B, delta

    ds_x, B_x, delta_x = compute_ds(U0, U1, xsi, direction, tol)
    ds_y, B_y, delta_y = compute_ds(V0, V1, eta, direction, tol)
    if withW:
        ds_z, B_z, delta_z = compute_ds(W0, W1, zeta, direction, tol)
    else:
        ds_z = np.inf

    # take the minimum travel time
    s_min = min(abs(ds_x), abs(ds_y), abs(ds_z), abs(dt / (dxdy * dz)))

    # calculate end position in time s_min
    def compute_rs(r, B, delta, s_min):  # noqa: N803
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(abs(B) < tol, -delta * s_min + r, (r + delta / B) * np.exp(-B * s_min) - delta / B)

    rs_x = compute_rs(xsi, B_x, delta_x, s_min)
    rs_y = compute_rs(eta, B_y, delta_y, s_min)

    particles.dx += (
        (1.0 - rs_x) * (1.0 - rs_y) * px[0]
        + rs_x * (1.0 - rs_y) * px[1]
        + rs_x * rs_y * px[2]
        + (1.0 - rs_x) * rs_y * px[3]
        - particles.x
    )
    particles.dy += (
        (1.0 - rs_x) * (1.0 - rs_y) * py[0]
        + rs_x * (1.0 - rs_y) * py[1]
        + rs_x * rs_y * py[2]
        + (1.0 - rs_x) * rs_y * py[3]
        - particles.y
    )

    if withW:
        rs_z = compute_rs(zeta, B_z, delta_z, s_min)
        particles.dz += (1.0 - rs_z) * pz[0] + rs_z * pz[1] - particles.z

    if particles.dt > 0:
        particles.dt = max(direction * s_min * (dxdy * dz), 1e-7)
    else:
        particles.dt = min(direction * s_min * (dxdy * dz), -1e-7)
