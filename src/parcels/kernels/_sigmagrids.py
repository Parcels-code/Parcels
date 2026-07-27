import warnings

import numpy as np


def convert_z_to_sigma_croco(fieldset, t, z, y, x, particle):
    """Calculate local sigma level of the particles, by linearly interpolating the
    scaling function that maps sigma to depth (using local ocean depth h,
    sea-surface Zeta and stretching parameters Cs_w and hc).
    See also https://croco-ocean.gitlabpages.inria.fr/croco_doc/model/model.grid.html#vertical-grid-parameters
    """
    h = fieldset.h.eval(t, np.zeros_like(z), y, x, particles=particle)
    zeta = fieldset.zeta.eval(t, np.zeros_like(z), y, x, particles=particle)
    sigma_levels = fieldset.U.grid.depth
    cs_w = fieldset.Cs_w.data.values.flatten()

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


def AdvectionRK2_3D_CROCO(particles, fieldset):  # pragma: no cover
    """Advection of particles using second-order Runge-Kutta integration including vertical velocity.
    This kernel assumes the vertical velocity is the 'w' field from CROCO output and works on sigma-layers.
    It also uses linear interpolation of the W field, which gives much better results than the default C-grid interpolation.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Sampling of velocities should normally be done using fieldset\.UV or fieldset\.UVW object; tread carefully$",
            category=RuntimeWarning,
        )  # Needed because of linear sampling of W with sigma conversion

        sigma = particles.z / fieldset.h[particles.t, np.zeros_like(particles.z), particles.y, particles.x]

        sig = convert_z_to_sigma_croco(fieldset, particles.t, particles.z, particles.y, particles.x, particles)
        (u1, v1) = fieldset.UV[particles.t, sig, particles.y, particles.x, particles]
        w1 = fieldset.W[particles.t, sig, particles.y, particles.x, particles]
        w1 *= sigma / fieldset.h[particles.t, np.zeros_like(particles.z), particles.y, particles.x]
        x1 = particles.x + u1 * 0.5 * particles.dt
        y1 = particles.y + v1 * 0.5 * particles.dt
        sig_dep1 = sigma + w1 * 0.5 * particles.dt
        dep1 = sig_dep1 * fieldset.h[particles.t, np.zeros_like(particles.z), y1, x1]

        sig1 = convert_z_to_sigma_croco(fieldset, particles.t + 0.5 * particles.dt, dep1, y1, x1, particles)
        (u2, v2) = fieldset.UV[particles.t + 0.5 * particles.dt, sig1, y1, x1, particles]
        w2 = fieldset.W[particles.t + 0.5 * particles.dt, sig1, y1, x1, particles]
        w2 *= sig_dep1 / fieldset.h[particles.t + 0.5 * particles.dt, np.zeros_like(particles.z), y1, x1]
        x2 = particles.x + u2 * 0.5 * particles.dt
        y2 = particles.y + v2 * 0.5 * particles.dt
        sig_dep2 = sigma + w2 * 0.5 * particles.dt
        dep2 = sig_dep2 * fieldset.h[particles.t + 0.5 * particles.dt, np.zeros_like(particles.z), y2, x2]

        particles.dx += u2 * particles.dt
        particles.dy += v2 * particles.dt
        particles.dz += (dep1 - particles.z) + (dep2 - particles.z)
