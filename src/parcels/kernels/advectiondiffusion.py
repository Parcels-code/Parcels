"""Collection of pre-built advection-diffusion kernels.

See `this tutorial <../examples/tutorial_diffusion.ipynb>`__ for a detailed explanation.
"""

import numpy as np

__all__ = ["AdvectionDiffusionEM", "AdvectionDiffusionM1", "DiffusionUniformKh"]


def AdvectionDiffusionM1(particles, fieldset):  # pragma: no cover
    """Kernel for 2D advection-diffusion, solved using the Milstein scheme at first order (M1).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central
    difference gradient approximation. This should be (of the order of) the
    local gridsize.

    This Milstein scheme is of strong and weak order 1, which is higher than the
    Euler-Maruyama scheme. It experiences less spurious diffusivity by
    including extra correction terms that are computationally cheap.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))

    Kxp1 = fieldset.Kh_zonal[particles.time, particles.z, particles.lat, particles.lon + fieldset.dres, particles]
    Kxm1 = fieldset.Kh_zonal[particles.time, particles.z, particles.lat, particles.lon - fieldset.dres, particles]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)

    u, v = fieldset.UV[particles.time, particles.z, particles.lat, particles.lon, particles]
    bx = np.sqrt(2 * fieldset.Kh_zonal[particles.time, particles.z, particles.lat, particles.lon, particles])

    Kyp1 = fieldset.Kh_meridional[particles.time, particles.z, particles.lat + fieldset.dres, particles.lon, particles]
    Kym1 = fieldset.Kh_meridional[particles.time, particles.z, particles.lat - fieldset.dres, particles.lon, particles]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)

    by = np.sqrt(2 * fieldset.Kh_meridional[particles.time, particles.z, particles.lat, particles.lon, particles])

    # Particle positions are updated only after evaluating all terms.
    particles.dlon += u * particles.dt + 0.5 * dKdx * (dWx**2 + particles.dt) + bx * dWx
    particles.dlat += v * particles.dt + 0.5 * dKdy * (dWy**2 + particles.dt) + by * dWy


def AdvectionDiffusionEM(particles, fieldset):  # pragma: no cover
    """Kernel for 2D advection-diffusion, solved using the Euler-Maruyama scheme (EM).

    Assumes that fieldset has fields `Kh_zonal` and `Kh_meridional`
    and variable `fieldset.dres`, setting the resolution for the central
    difference gradient approximation. This should be (of the order of) the
    local gridsize.

    The Euler-Maruyama scheme is of strong order 0.5 and weak order 1.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))

    u, v = fieldset.UV[particles.time, particles.z, particles.lat, particles.lon, particles]

    Kxp1 = fieldset.Kh_zonal[particles.time, particles.z, particles.lat, particles.lon + fieldset.dres, particles]
    Kxm1 = fieldset.Kh_zonal[particles.time, particles.z, particles.lat, particles.lon - fieldset.dres, particles]
    dKdx = (Kxp1 - Kxm1) / (2 * fieldset.dres)
    ax = u + dKdx
    bx = np.sqrt(2 * fieldset.Kh_zonal[particles.time, particles.z, particles.lat, particles.lon, particles])

    Kyp1 = fieldset.Kh_meridional[particles.time, particles.z, particles.lat + fieldset.dres, particles.lon, particles]
    Kym1 = fieldset.Kh_meridional[particles.time, particles.z, particles.lat - fieldset.dres, particles.lon, particles]
    dKdy = (Kyp1 - Kym1) / (2 * fieldset.dres)
    ay = v + dKdy
    by = np.sqrt(2 * fieldset.Kh_meridional[particles.time, particles.z, particles.lat, particles.lon, particles])

    # Particle positions are updated only after evaluating all terms.
    particles.dlon += ax * particles.dt + bx * dWx
    particles.dlat += ay * particles.dt + by * dWy


def DiffusionUniformKh(particles, fieldset):  # pragma: no cover
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.

    Assumes that fieldset has constant fields `Kh_zonal` and `Kh_meridional`.
    These can be added via e.g.
    `fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)`
    or
    `fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)`
    where mesh is either 'flat' or 'spherical'

    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
    Since the perturbation due to diffusion is in this case isotropic independent, this
    kernel contains no advection and can be used in combination with a separate
    advection kernel.

    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    # Wiener increment with zero mean and std of sqrt(dt)
    dWx = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))
    dWy = np.random.normal(0, np.sqrt(np.fabs(particles.dt)))

    bx = np.sqrt(2 * fieldset.Kh_zonal[particles])
    by = np.sqrt(2 * fieldset.Kh_meridional[particles])

    particles.dlon += bx * dWx
    particles.dlat += by * dWy
