"""Collection of pre-built advection kernels."""

import math

import numpy as np

from parcels._core.statuscodes import StatusCode

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


def MRAdvectionRK4_3D(particles, fieldset):  # pragma: no cover
    # Maxey-Riley advection of particles using fourth-order Runge-Kutta integration including vertical velocity, inspired from Meike and Jimena
    """
    Advection of particles using Maxey-Riley equation in 2D without Basset
    history term and Faxen corrections without sinking or floating force.
    The equation is numerically integrated using the 4th order runge kutta
    scheme for a 2nd order ODE equation. We appromate the time derivative at t
    (1rst step rk4) with a forward finite difference and the time derivative
    at t+delta_t (4th step rk4) with a backward finite difference.

    dependencies:
    - up, vp, particle velocity (particle variables)
    - tau, stokes relaxation time particle (particle variable)
    - B, buoyancy particle (particle variable)
    - Omega_earth, angular velocity earth (fieldset constant)
    - delta_x, delta_y, delta_t step for finite difference method gradients
      (fieldset constants)
    """
    tau_inv = 1.0 / particles.tau
    Bterm = 3.0 / (1.0 + 2.0 * particles.B)
    Bterm2 = 2 * (1 - particles.B) / (1 + 2 * particles.B)
    w0 = Bterm2 * 9.81 * particles.tau  ##fieldset.g
    norm_deltax = 1.0 / (2.0 * 0.001)  # fieldset.dx
    norm_deltay = 1.0 / (2.0 * 0.0125)  # fieldset.dy
    norm_deltaz = 1.0 / (2.0 * 0.05)  ##fieldset.dz

    dt_seconds = particles.dt  # already in seconds (0.001 for your 1 ms step)
    norm_deltat = 1.0 / dt_seconds

    # RK4 STEP 1
    ## read in velocity field at location of particle
    (uf1, vf1, wf1) = fieldset.UVW[particles]

    # velocity particle at current step
    u, v, w = fieldset.UVW[particles]
    up1 = u  # particles.up
    vp1 = v  # particles.vp
    wp1 = w  # particle.wp

    # calculate time derivative of fluid field
    # (uf_tp1, vf_tp1) = fieldset.UV[time+particle.dt,
    (uf_tp1, vf_tp1, wf_tp1) = fieldset.UVW[particles.t + particles.dt, particles.z, particles.y, particles.x]
    (uf_tm1, vf_tm1, wf_tm1) = fieldset.UVW[particles.t, particles.z, particles.y, particles.x]
    dudt1 = (uf_tp1 - uf_tm1) * norm_deltat
    dvdt1 = (vf_tp1 - vf_tm1) * norm_deltat
    dwdt1 = (wf_tp1 - wf_tm1) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm1, v_dxm1, w_dxm1) = fieldset.UVW[particles.t, particles.z, particles.y, particles.x - 0.001]  # fieldset.dx
    (u_dxp1, v_dxp1, w_dxp1) = fieldset.UVW[particles.t, particles.z, particles.y, particles.x + 0.001]  # + fieldset.dx
    (u_dym1, v_dym1, w_dym1) = fieldset.UVW[particles.t, particles.z, particles.y - 0.0125, particles.x]  # fieldset.dy
    (u_dyp1, v_dyp1, w_dyp1) = fieldset.UVW[particles.t, particles.z, particles.y + 0.0125, particles.x]  # fieldset.dy
    (u_dzm1, v_dzm1, w_dzm1) = fieldset.UVW[particles.t, particles.z - 0.05, particles.y, particles.x]  ##fieldset.dz
    (u_dzp1, v_dzp1, w_dzp1) = fieldset.UVW[particles.t, particles.z + 0.05, particles.y, particles.x]  ##fieldset.dz
    dudx1 = (u_dxp1 - u_dxm1) * norm_deltax
    dudy1 = (u_dyp1 - u_dym1) * norm_deltay
    dudz1 = (u_dzp1 - u_dzm1) * norm_deltaz
    dvdx1 = (v_dxp1 - v_dxm1) * norm_deltax
    dvdy1 = (v_dyp1 - v_dym1) * norm_deltay
    dvdz1 = (v_dzp1 - v_dzm1) * norm_deltaz
    dwdx1 = (w_dxp1 - w_dxm1) * norm_deltax
    dwdy1 = (w_dyp1 - w_dym1) * norm_deltay
    dwdz1 = (w_dzp1 - w_dzm1) * norm_deltaz

    # caluclate material derivative fluid
    DuDt1 = dudt1 + uf1 * dudx1 + vf1 * dudy1 + wf1 * dudz1
    DvDt1 = dvdt1 + uf1 * dvdx1 + vf1 * dvdy1 + wf1 * dvdz1
    DwDt1 = dwdt1 + uf1 * dwdx1 + vf1 * dwdy1 + wf1 * dwdz1

    # drag force
    udrag1 = tau_inv * (uf1 - up1)
    vdrag1 = tau_inv * (vf1 - vp1)
    wdrag1 = tau_inv * (w0 + wf1 - wp1)

    # acceleration
    a_lon1 = Bterm * (DuDt1) + udrag1
    a_lat1 = Bterm * (DvDt1) + vdrag1
    a_depth1 = Bterm * DwDt1 + wdrag1

    # lon, lat for next step
    lon1 = particles.x + 0.5 * up1 * dt_seconds
    lat1 = particles.y + 0.5 * vp1 * dt_seconds
    depth1 = particles.z + 0.5 * wp1 * particles.dt
    time1 = particles.t + 0.5 * particles.dt

    # RK4 STEP 2
    # velocity particle at current step
    up2 = particles.up + 0.5 * a_lon1 * dt_seconds
    vp2 = particles.vp + 0.5 * a_lat1 * dt_seconds
    wp2 = particles.wp + 0.5 * a_depth1 * particles.dt

    # read in velocity at location of particle
    (uf2, vf2, wf2) = fieldset.UVW[time1, depth1, lat1, lon1]

    # calculate time derivative of fluid field
    (uf_tp2, vf_tp2, wf_tp2) = fieldset.UVW[particles.t + particles.dt, depth1, lat1, lon1]
    (uf_tm2, vf_tm2, wf_tm2) = fieldset.UVW[particles.t, depth1, lat1, lon1]
    dudt2 = (uf_tp2 - uf_tm2) * norm_deltat
    dvdt2 = (vf_tp2 - vf_tm2) * norm_deltat
    dwdt2 = (wf_tp2 - wf_tm2) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm2, v_dxm2, w_dxm2) = fieldset.UVW[time1, depth1, lat1, lon1 - 0.001]  # fieldset.dx
    (u_dxp2, v_dxp2, w_dxp2) = fieldset.UVW[time1, depth1, lat1, lon1 + 0.001]  ##fieldset.dx
    (u_dym2, v_dym2, w_dym2) = fieldset.UVW[time1, depth1, lat1 - 0.0125, lon1]  # fieldset.dy
    (u_dyp2, v_dyp2, w_dyp2) = fieldset.UVW[time1, depth1, lat1 + 0.0125, lon1]  # fieldset.dy
    (u_dzm2, v_dzm2, w_dzm2) = fieldset.UVW[time1, depth1 - 0.05, lat1, lon1]  ##fieldset.dz
    (u_dzp2, v_dzp2, w_dzp2) = fieldset.UVW[time1, depth1 + 0.05, lat1, lon1]  ##fieldset.dz
    dudx2 = (u_dxp2 - u_dxm2) * norm_deltax
    dudy2 = (u_dyp2 - u_dym2) * norm_deltay
    dudz2 = (u_dzp2 - u_dzm2) * norm_deltaz
    dvdx2 = (v_dxp2 - v_dxm2) * norm_deltax
    dvdy2 = (v_dyp2 - v_dym2) * norm_deltay
    dvdz2 = (v_dzp2 - v_dzm2) * norm_deltaz
    dwdx2 = (w_dxp2 - w_dxm2) * norm_deltax
    dwdy2 = (w_dyp2 - w_dym2) * norm_deltay
    dwdz2 = (w_dzp2 - w_dzm2) * norm_deltaz

    # caluclate material derivative fluid
    DuDt2 = dudt2 + uf2 * dudx2 + vf2 * dudy2 + wf2 * dudz2
    DvDt2 = dvdt2 + uf2 * dvdx2 + vf2 * dvdy2 + wf2 * dvdz2
    DwDt2 = dwdt2 + uf2 * dwdx2 + vf2 * dwdy2 + wf2 * dwdz2

    # drag force
    udrag2 = tau_inv * (uf2 - up2)
    vdrag2 = tau_inv * (vf2 - vp2)
    wdrag2 = tau_inv * (w0 + wf2 - wp2)

    # acceleration
    a_lon2 = Bterm * (DuDt2) + udrag2
    a_lat2 = Bterm * (DvDt2) + vdrag2
    a_depth2 = Bterm * DwDt2 + wdrag2

    # lon, lat for next step
    lon2 = particles.x + 0.5 * up2 * dt_seconds
    lat2 = particles.y + 0.5 * vp2 * dt_seconds
    depth2 = particles.z + 0.5 * wp2 * particles.dt
    time2 = particles.t + 0.5 * particles.dt

    # RK4 STEP 3
    # velocity particle at current step
    up3 = particles.up + 0.5 * a_lon2 * dt_seconds
    vp3 = particles.vp + 0.5 * a_lat2 * dt_seconds
    wp3 = particles.wp + 0.5 * a_depth2 * particles.dt

    # read in velocity at location of particle
    (uf3, vf3, wf3) = fieldset.UVW[time2, depth2, lat2, lon2]

    # calculate time derivative of fluid field
    (uf_tp3, vf_tp3, wf_tp3) = fieldset.UVW[particles.t + particles.dt, depth2, lat2, lon2]
    (uf_tm3, vf_tm3, wf_tm3) = fieldset.UVW[particles.t, depth2, lat2, lon2]
    dudt3 = (uf_tp3 - uf_tm3) * norm_deltat
    dvdt3 = (vf_tp3 - vf_tm3) * norm_deltat
    dwdt3 = (wf_tp3 - wf_tm3) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm3, v_dxm3, w_dxm3) = fieldset.UVW[time2, depth2, lat2, lon2 - 0.001]  ##fieldset.dx
    (u_dxp3, v_dxp3, w_dxp3) = fieldset.UVW[time2, depth2, lat2, lon2 + 0.001]  # fieldset.dx
    (u_dym3, v_dym3, w_dym3) = fieldset.UVW[time2, depth2, lat2 - 0.0125, lon2]  # fieldset.dy
    (u_dyp3, v_dyp3, w_dyp3) = fieldset.UVW[time2, depth2, lat2 + 0.0125, lon2]  # fieldset.dy
    (u_dzm3, v_dzm3, w_dzm3) = fieldset.UVW[time2, depth2 - 0.05, lat2, lon2]  ## fieldset.dz
    (u_dzp3, v_dzp3, w_dzp3) = fieldset.UVW[time2, depth2 + 0.05, lat2, lon2]  ##fieldset.dz
    dudx3 = (u_dxp3 - u_dxm3) * norm_deltax
    dudy3 = (u_dyp3 - u_dym3) * norm_deltay
    dudz3 = (u_dzp3 - u_dzm3) * norm_deltaz
    dvdz3 = (v_dzp3 - v_dzm3) * norm_deltaz
    dvdx3 = (v_dxp3 - v_dxm3) * norm_deltax
    dvdy3 = (v_dyp3 - v_dym3) * norm_deltay
    dwdx3 = (w_dxp3 - w_dxm3) * norm_deltax
    dwdy3 = (w_dyp3 - w_dym3) * norm_deltay
    dwdz3 = (w_dzp3 - w_dzm3) * norm_deltaz

    # caluclate material derivative fluid
    DuDt3 = dudt3 + uf3 * dudx3 + vf3 * dudy3 + wf3 * dudz3
    DvDt3 = dvdt3 + uf3 * dvdx3 + vf3 * dvdy3 + wf3 * dvdz3
    DwDt3 = dwdt3 + uf3 * dwdx3 + vf3 * dwdy3 + wf3 * dwdz3

    # drag force
    udrag3 = tau_inv * (uf3 - up3)
    vdrag3 = tau_inv * (vf3 - vp3)
    wdrag3 = tau_inv * (w0 + wf3 - wp3)

    # acceleration
    a_lon3 = Bterm * (DuDt3) + udrag3
    a_lat3 = Bterm * (DvDt3) + vdrag3
    a_depth3 = Bterm * DwDt3 + wdrag3

    # lon, lat for next step
    lon3 = particles.x + up3 * dt_seconds
    lat3 = particles.y + vp3 * dt_seconds
    depth3 = particles.z + wp3 * particles.dt
    time3 = particles.t + particles.dt

    # RK4 STEP 4
    # velocity particle at current step
    up4 = particles.up + a_lon3 * dt_seconds
    vp4 = particles.vp + a_lat3 * dt_seconds
    wp4 = particles.wp + a_depth3 * particles.dt

    # read in velocity at location of particle
    (uf4, vf4, wf4) = fieldset.UVW[time3, depth3, lat3, lon3]

    # calculate time derivative of fluid field
    (uf_tp4, vf_tp4, wf_tp4) = fieldset.UVW[particles.t + particles.dt, depth3, lat3, lon3]
    (uf_tm4, vf_tm4, wf_tm4) = fieldset.UVW[particles.t, depth3, lat3, lon3]
    dudt4 = (uf_tp4 - uf_tm4) * norm_deltat
    dvdt4 = (vf_tp4 - vf_tm4) * norm_deltat
    dwdt4 = (wf_tp4 - wf_tm4) * norm_deltat

    # calculate spatial gradients fluid field
    (u_dxm4, v_dxm4, w_dxm4) = fieldset.UVW[time3, depth3, lat3, lon3 - 0.001]  ##fieldset.dx
    (u_dxp4, v_dxp4, w_dxp4) = fieldset.UVW[time3, depth3, lat3, lon3 + 0.001]  ## fieldset.dx
    (u_dym4, v_dym4, w_dym4) = fieldset.UVW[time3, depth3, lat3 - 0.0125, lon3]  ##fieldset.dy
    (u_dyp4, v_dyp4, w_dyp4) = fieldset.UVW[time3, depth3, lat3 + 0.0125, lon3]  ##fieldset.dy
    (u_dzm4, v_dzm4, w_dzm4) = fieldset.UVW[time3, depth3 - 0.05, lat3, lon3]  ## fieldset.dz
    (u_dzp4, v_dzp4, w_dzp4) = fieldset.UVW[time3, depth3 + 0.05, lat3, lon3]  ## fieldset.dz
    dudx4 = (u_dxp4 - u_dxm4) * norm_deltax
    dudy4 = (u_dyp4 - u_dym4) * norm_deltay
    dudz4 = (u_dzp4 - u_dzm4) * norm_deltaz
    dvdx4 = (v_dxp4 - v_dxm4) * norm_deltax
    dvdy4 = (v_dyp4 - v_dym4) * norm_deltay
    dvdz4 = (v_dzp4 - v_dzm4) * norm_deltaz
    dwdx4 = (w_dxp4 - w_dxm4) * norm_deltax
    dwdy4 = (w_dyp4 - w_dym4) * norm_deltay
    dwdz4 = (w_dzp4 - w_dzm4) * norm_deltaz

    # caluclate material derivative fluid
    DuDt4 = dudt4 + uf4 * dudx4 + vf4 * dudy4 + wf4 * dudz4
    DvDt4 = dvdt4 + uf4 * dvdx4 + vf4 * dvdy4 + wf4 * dvdz4
    DwDt4 = dwdt4 + uf4 * dwdx4 + vf4 * dwdy4 + wf4 * dwdz4

    # drag force
    udrag4 = tau_inv * (uf4 - up4)
    vdrag4 = tau_inv * (vf4 - vp4)
    wdrag4 = tau_inv * (w0 + wf4 - wp4)

    # acceleration
    a_lon4 = Bterm * (DuDt4) + udrag4
    a_lat4 = Bterm * (DvDt4) + vdrag4
    a_depth4 = Bterm * DwDt4 + wdrag4

    # RK4 INTEGRATION STEP
    particles.up += (a_lon1 + 2 * a_lon2 + 2 * a_lon3 + a_lon4) * dt_seconds / 6.0
    particles.vp += (a_lat1 + 2 * a_lat2 + 2 * a_lat3 + a_lat4) * dt_seconds / 6.0
    particles.wp += (a_depth1 + 2 * a_depth2 + 2 * a_depth3 + a_depth4) * dt_seconds / 6.0
    particles.dx += (up1 + 2 * up2 + 2 * up3 + up4) * dt_seconds / 6.0
    particles.dy += (vp1 + 2 * vp2 + 2 * vp3 + vp4) * dt_seconds / 6.0
    particles.dz += (wp1 + 2 * wp2 + 2 * wp3 + wp4) * dt_seconds / 6.0


def AdvectionAnalytical(particles, fieldset):  # pragma: no cover
    """Advection of particles using 'analytical advection' integration.

    Based on Ariane/TRACMASS algorithm, as detailed in e.g. Doos et al (https://doi.org/10.5194/gmd-10-1733-2017).
    Note that the time-dependent scheme is currently implemented with 'intermediate timesteps'
    (default 10 per model timestep) and not yet with the full analytical time integration.
    """
    import numpy as np

    import parcels._core.utils.interpolation as i_u

    tol = 1e-10
    I_s = 10  # number of intermediate time steps
    dt = particles.dt
    direction = 1.0 if dt > 0 else -1.0
    withW = True if "W" in [f.name for f in fieldset.fields.values()] else False
    withTime = True if len(fieldset.U.grid.time) > 1 else False
    tau, zeta, eta, xsi, ti, zi, yi, xi = fieldset.U._search_indices(
        particles.z, particles.y, particles.x, particles=particles
    )
    ds_t = dt
    if withTime:
        time_i = np.linspace(0, fieldset.U.grid.time[ti + 1] - fieldset.U.grid.time[ti], I_s)
        ds_t = min(ds_t, time_i[np.where(particles.time - fieldset.U.grid.time[ti] < time_i)[0][0]])

    if withW:
        if abs(xsi - 1) < tol:
            if fieldset.U.data[0, zi + 1, yi + 1, xi + 1] > 0:
                xi += 1
                xsi = 0
        if abs(eta - 1) < tol:
            if fieldset.V.data[0, zi + 1, yi + 1, xi + 1] > 0:
                yi += 1
                eta = 0
        if abs(zeta - 1) < tol:
            if fieldset.W.data[0, zi + 1, yi + 1, xi + 1] > 0:
                zi += 1
                zeta = 0
    else:
        if abs(xsi - 1) < tol:
            if fieldset.U.data[0, yi + 1, xi + 1] > 0:
                xi += 1
                xsi = 0
        if abs(eta - 1) < tol:
            if fieldset.V.data[0, yi + 1, xi + 1] > 0:
                yi += 1
                eta = 0

    particles.ei[:] = fieldset.U.ravel_index(zi, yi, xi)

    grid = fieldset.U.grid
    if grid._gtype < 2:
        px = np.array([grid.x[xi], grid.x[xi + 1], grid.x[xi + 1], grid.x[xi]])
        py = np.array([grid.y[yi], grid.y[yi], grid.y[yi + 1], grid.y[yi + 1]])
    else:
        px = np.array([grid.x[yi, xi], grid.x[yi, xi + 1], grid.x[yi + 1, xi + 1], grid.x[yi + 1, xi]])
        py = np.array([grid.y[yi, xi], grid.y[yi, xi + 1], grid.y[yi + 1, xi + 1], grid.y[yi + 1, xi]])
    if grid.mesh == "spherical":
        px[0] = px[0] + 360 if px[0] < particles.x - 225 else px[0]
        px[0] = px[0] - 360 if px[0] > particles.y + 225 else px[0]
        px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
        px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
    if withW:
        pz = np.array([grid.depth[zi], grid.depth[zi + 1]])
        dz = pz[1] - pz[0]
    else:
        dz = 1.0

    c1 = i_u._geodetic_distance(py[0], py[1], px[0], px[1], grid.mesh, np.dot(i_u.phi2D_lin(0.0, xsi), py), grid.deg2m)
    c2 = i_u._geodetic_distance(py[1], py[2], px[1], px[2], grid.mesh, np.dot(i_u.phi2D_lin(eta, 1.0), py), grid.deg2m)
    c3 = i_u._geodetic_distance(py[2], py[3], px[2], px[3], grid.mesh, np.dot(i_u.phi2D_lin(1.0, xsi), py), grid.deg2m)
    c4 = i_u._geodetic_distance(py[3], py[0], px[3], px[0], grid.mesh, np.dot(i_u.phi2D_lin(eta, 0.0), py), grid.deg2m)
    rad = np.pi / 180.0
    deg2m = grid.deg2m
    meshJac = (deg2m * deg2m * math.cos(rad * particles.y)) if grid.mesh == "spherical" else 1
    dxdy = i_u._compute_jacobian_determinant(py, px, eta, xsi) * meshJac

    if withW:
        U0 = direction * fieldset.U.data[ti, zi + 1, yi + 1, xi] * c4 * dz
        U1 = direction * fieldset.U.data[ti, zi + 1, yi + 1, xi + 1] * c2 * dz
        V0 = direction * fieldset.V.data[ti, zi + 1, yi, xi + 1] * c1 * dz
        V1 = direction * fieldset.V.data[ti, zi + 1, yi + 1, xi + 1] * c3 * dz
        if withTime:
            U0 = U0 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, zi + 1, yi + 1, xi] * c4 * dz
            U1 = U1 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, zi + 1, yi + 1, xi + 1] * c2 * dz
            V0 = V0 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, zi + 1, yi, xi + 1] * c1 * dz
            V1 = V1 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, zi + 1, yi + 1, xi + 1] * c3 * dz
    else:
        U0 = direction * fieldset.U.data[ti, yi + 1, xi] * c4 * dz
        U1 = direction * fieldset.U.data[ti, yi + 1, xi + 1] * c2 * dz
        V0 = direction * fieldset.V.data[ti, yi, xi + 1] * c1 * dz
        V1 = direction * fieldset.V.data[ti, yi + 1, xi + 1] * c3 * dz
        if withTime:
            U0 = U0 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, yi + 1, xi] * c4 * dz
            U1 = U1 * (1 - tau) + tau * direction * fieldset.U.data[ti + 1, yi + 1, xi + 1] * c2 * dz
            V0 = V0 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, yi, xi + 1] * c1 * dz
            V1 = V1 * (1 - tau) + tau * direction * fieldset.V.data[ti + 1, yi + 1, xi + 1] * c3 * dz

    def compute_ds(F0, F1, r, direction, tol):  # noqa: N803
        up = F0 * (1 - r) + F1 * r
        r_target = 1.0 if direction * up >= 0.0 else 0.0
        B = F0 - F1
        delta = -F0
        B = 0 if abs(B) < tol else B

        if abs(B) > tol:
            F_r1 = r_target + delta / B
            F_r0 = r + delta / B
        else:
            F_r0, F_r1 = None, None

        if abs(B) < tol and abs(delta) < tol:
            ds = float("inf")
        elif B == 0:
            ds = -(r_target - r) / delta
        elif F_r1 * F_r0 < tol:
            ds = float("inf")
        else:
            ds = -1.0 / B * math.log(F_r1 / F_r0)

        if abs(ds) < tol:
            ds = float("inf")
        return ds, B, delta

    ds_x, B_x, delta_x = compute_ds(U0, U1, xsi, direction, tol)
    ds_y, B_y, delta_y = compute_ds(V0, V1, eta, direction, tol)
    if withW:
        W0 = direction * fieldset.W.data[ti, zi, yi + 1, xi + 1] * dxdy
        W1 = direction * fieldset.W.data[ti, zi + 1, yi + 1, xi + 1] * dxdy
        if withTime:
            W0 = W0 * (1 - tau) + tau * direction * fieldset.W.data[ti + 1, zi, yi + 1, xi + 1] * dxdy
            W1 = W1 * (1 - tau) + tau * direction * fieldset.W.data[ti + 1, zi + 1, yi + 1, xi + 1] * dxdy
        ds_z, B_z, delta_z = compute_ds(W0, W1, zeta, direction, tol)
    else:
        ds_z = float("inf")

    # take the minimum travel time
    s_min = min(abs(ds_x), abs(ds_y), abs(ds_z), abs(ds_t / (dxdy * dz)))

    # calculate end position in time s_min
    def compute_rs(r, B, delta, s_min):  # noqa: N803
        if abs(B) < tol:
            return -delta * s_min + r
        else:
            return (r + delta / B) * math.exp(-B * s_min) - delta / B

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
        particles.dt = max(direction * s_min * (dxdy * dz), 1e-7).astype("timedelta64[s]")
    else:
        particles.dt = min(direction * s_min * (dxdy * dz), -1e-7).astype("timedelta64[s]")
