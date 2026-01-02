import numpy as np

from parcels.kernels.advection import _constrain_dt_to_within_time_interval

__all__ = [
    "AdvectionRK4_3D_CROCO",
    "SampleOmegaCroco",
    "convert_z_to_sigma_croco",
]


def convert_z_to_sigma_croco(fieldset, time, z, y, x, particle):
    """Calculate local sigma level of the particles, by linearly interpolating the
    scaling function that maps sigma to depth (using local ocean depth h,
    sea-surface Zeta and stretching parameters Cs_w and hc).
    See also https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.grid.html#vertical-grid-parameters
    """
    h = fieldset.h.eval(time, np.zeros_like(z), y, x, particles=particle, applyConversion=False)
    zeta = fieldset.zeta.eval(time, np.zeros_like(z), y, x, particles=particle, applyConversion=False)
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
    sigma = convert_z_to_sigma_croco(fieldset, particles.time, particles.z, particles.lat, particles.lon, particles)
    particles.omega = fieldset.omega[particles.time, sigma, particles.lat, particles.lon, particles]


# TODO change to RK2 (once RK4 yields same results as v3)
def AdvectionRK4_3D_CROCO(particles, fieldset):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration including vertical velocity.
    This kernel assumes the vertical velocity is the 'w' field from CROCO output and works on sigma-layers.
    """
    dt = _constrain_dt_to_within_time_interval(fieldset.time_interval, particles.time, particles.dt)
    sigma = particles.z / fieldset.h[particles.time, 0, particles.lat, particles.lon]

    sig = convert_z_to_sigma_croco(fieldset, particles.time, particles.z, particles.lat, particles.lon, particles)
    (u1, v1, w1) = fieldset.UVW[particles.time, sig, particles.lat, particles.lon, particles]
    w1 *= sigma / fieldset.h[particles.time, 0, particles.lat, particles.lon]
    lon1 = particles.lon + u1 * 0.5 * dt
    lat1 = particles.lat + v1 * 0.5 * dt
    sig_dep1 = sigma + w1 * 0.5 * dt
    dep1 = sig_dep1 * fieldset.h[particles.time, 0, lat1, lon1]

    sig1 = convert_z_to_sigma_croco(fieldset, particles.time + 0.5 * dt, dep1, lat1, lon1, particles)
    (u2, v2, w2) = fieldset.UVW[particles.time + 0.5 * dt, sig1, lat1, lon1, particles]
    w2 *= sig_dep1 / fieldset.h[particles.time, 0, lat1, lon1]
    lon2 = particles.lon + u2 * 0.5 * dt
    lat2 = particles.lat + v2 * 0.5 * dt
    sig_dep2 = sigma + w2 * 0.5 * dt
    dep2 = sig_dep2 * fieldset.h[particles.time, 0, lat2, lon2]

    sig2 = convert_z_to_sigma_croco(fieldset, particles.time + 0.5 * dt, dep2, lat2, lon2, particles)
    (u3, v3, w3) = fieldset.UVW[particles.time + 0.5 * dt, sig2, lat2, lon2, particles]
    w3 *= sig_dep2 / fieldset.h[particles.time, 0, lat2, lon2]
    lon3 = particles.lon + u3 * dt
    lat3 = particles.lat + v3 * dt
    sig_dep3 = sigma + w3 * dt
    dep3 = sig_dep3 * fieldset.h[particles.time, 0, lat3, lon3]

    sig3 = convert_z_to_sigma_croco(fieldset, particles.time + dt, dep3, lat3, lon3, particles)
    (u4, v4, w4) = fieldset.UVW[particles.time + dt, sig3, lat3, lon3, particles]
    w4 *= sig_dep3 / fieldset.h[particles.time, 0, lat3, lon3]
    lon4 = particles.lon + u4 * dt
    lat4 = particles.lat + v4 * dt
    sig_dep4 = sigma + w4 * dt

    dep4 = sig_dep4 * fieldset.h[particles.time, 0, lat4, lon4]
    particles.dlon += (u1 + 2 * u2 + 2 * u3 + u4) / 6 * dt
    particles.dlat += (v1 + 2 * v2 + 2 * v3 + v4) / 6 * dt
    particles.dz += (
        (dep1 - particles.z) * 2 + 2 * (dep2 - particles.z) * 2 + 2 * (dep3 - particles.z) + dep4 - particles.z
    ) / 6
