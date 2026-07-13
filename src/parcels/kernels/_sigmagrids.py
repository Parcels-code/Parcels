import numpy as np

from parcels.kernels._advection import _constrain_dt_to_within_time_interval


def convert_z_to_sigma_croco(fieldset, t, z, y, x, particle):
    """Calculate local sigma level of the particles, by linearly interpolating the
    scaling function that maps sigma to depth (using local ocean depth h,
    sea-surface Zeta and stretching parameters Cs_w and hc).
    See also https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.grid.html#vertical-grid-parameters
    """
    h = fieldset.h.eval(t, np.zeros_like(z), y, x, particles=particle)
    zeta = fieldset.zeta.eval(t, np.zeros_like(z), y, x, particles=particle)
    sigma_levels = fieldset.U.grid.depth
    cs_w = fieldset.Cs_w.data[0, :, 0, 0].values

    z0 = fieldset.hc * sigma_levels[None, :] + (h[:, None] - fieldset.hc) * cs_w[None, :]
    zvec = z0 + zeta[:, None] * (1.0 + (z0 / h[:, None]))
    zinds = zvec <= z[:, None]
    zi = np.argmin(zinds, axis=1) - 1
    zi = np.where(zinds.all(axis=1), zvec.shape[1] - 2, zi)
    idx = np.arange(zi.shape[0])
    return sigma_levels[zi] + (z - zvec[idx, zi]) * (sigma_levels[zi + 1] - sigma_levels[zi]) / (
        zvec[idx, zi + 1] - zvec[idx, zi]
    )


def SampleOmegaCroco(particles, fieldset):
    """Sample omega field on a CROCO sigma grid by first converting z to sigma levels.

    This Kernel can be adapted to sample any other field on a CROCO sigma grid by
    replacing 'omega' with the desired field name.
    """
    sigma = convert_z_to_sigma_croco(fieldset, particles.t, particles.z, particles.y, particles.x, particles)
    particles.omega = fieldset.omega[particles.t, sigma, particles.y, particles.x, particles]


# TODO change to RK2 (once RK4 yields same results as v3)
def AdvectionRK4_3D_CROCO(particles, fieldset):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    This kernel assumes the vertical velocity is the 'w' field from CROCO output and works on sigma-layers.
    It also uses linear interpolation of the W field, which gives much better results than the default C-grid interpolation.
    """
    dt = _constrain_dt_to_within_time_interval(fieldset.time_interval, particles.t, particles.dt)
    sigma = particles.z / fieldset.h[particles.t, np.zeros_like(particles.z), particles.y, particles.x]

    sig = convert_z_to_sigma_croco(fieldset, particles.t, particles.z, particles.y, particles.x, particles)
    (u1, v1) = fieldset.UV[particles.t, sig, particles.y, particles.x, particles]
    w1 = fieldset.W[particles.t, sig, particles.y, particles.x, particles]
    w1 *= sigma / fieldset.h[particles.t, np.zeros_like(particles.z), particles.y, particles.x]
    x1 = particles.x + u1 * 0.5 * dt
    y1 = particles.y + v1 * 0.5 * dt
    sig_dep1 = sigma + w1 * 0.5 * dt
    dep1 = sig_dep1 * fieldset.h[particles.t, np.zeros_like(particles.z), y1, x1]

    sig1 = convert_z_to_sigma_croco(fieldset, particles.t + 0.5 * dt, dep1, y1, x1, particles)
    (u2, v2) = fieldset.UV[particles.t + 0.5 * dt, sig1, y1, x1, particles]
    w2 = fieldset.W[particles.t + 0.5 * dt, sig1, y1, x1, particles]
    w2 *= sig_dep1 / fieldset.h[particles.t + 0.5 * dt, np.zeros_like(particles.z), y1, x1]
    x2 = particles.x + u2 * 0.5 * dt
    y2 = particles.y + v2 * 0.5 * dt
    sig_dep2 = sigma + w2 * 0.5 * dt
    dep2 = sig_dep2 * fieldset.h[particles.t + 0.5 * dt, np.zeros_like(particles.z), y2, x2]

    sig2 = convert_z_to_sigma_croco(fieldset, particles.t + 0.5 * dt, dep2, y2, x2, particles)
    (u3, v3) = fieldset.UV[particles.t + 0.5 * dt, sig2, y2, x2, particles]
    w3 = fieldset.W[particles.t + 0.5 * dt, sig2, y2, x2, particles]
    w3 *= sig_dep2 / fieldset.h[particles.t + 0.5 * dt, np.zeros_like(particles.z), y2, x2]
    x3 = particles.x + u3 * dt
    y3 = particles.y + v3 * dt
    sig_dep3 = sigma + w3 * dt
    dep3 = sig_dep3 * fieldset.h[particles.t + dt, np.zeros_like(particles.z), y3, x3]

    sig3 = convert_z_to_sigma_croco(fieldset, particles.t + dt, dep3, y3, x3, particles)
    (u4, v4) = fieldset.UV[particles.t + dt, sig3, y3, x3, particles]
    w4 = fieldset.W[particles.t + dt, sig3, y3, x3, particles]
    w4 *= sig_dep3 / fieldset.h[particles.t + dt, np.zeros_like(particles.z), y3, x3]
    x4 = particles.x + u4 * dt
    y4 = particles.y + v4 * dt
    sig_dep4 = sigma + w4 * dt

    dep4 = sig_dep4 * fieldset.h[particles.t + dt, np.zeros_like(particles.z), y4, x4]
    particles.dx += (u1 + 2 * u2 + 2 * u3 + u4) / 6 * dt
    particles.dy += (v1 + 2 * v2 + 2 * v3 + v4) / 6 * dt
    particles.dz += (
        (dep1 - particles.z) * 2 + 2 * (dep2 - particles.z) * 2 + 2 * (dep3 - particles.z) + dep4 - particles.z
    ) / 6
