"""Collection of time integrators for use in Parcels Kernels"""


def RK2(particles, fieldset, rhs):
    z, y, x = particles.z, particles.y, particles.x
    fields = rhs(fieldset, particles.t, z, y, x, particles)
    if len(fields) == 1:
        raise NotImplementedError("RK2 integration is not implemented for Fields that return only one component.")
    if len(fields) > 1:
        x = particles.x + fields[0] * 0.5 * particles.dt
        y = particles.y + fields[1] * 0.5 * particles.dt
    if len(fields) > 2:
        z = particles.z + fields[2] * 0.5 * particles.dt
    t = particles.t + 0.5 * particles.dt
    return rhs(fieldset, t, z, y, x, particles)


def RK4(particles, fieldset, rhs):
    z, y, x = particles.z, particles.y, particles.x
    k1 = rhs(fieldset, particles.t, z, y, x, particles)
    if len(k1) == 1:
        raise NotImplementedError("RK4 integration is not implemented for Fields that return only one component.")
    if len(k1) > 1:
        x = particles.x + k1[0] * 0.5 * particles.dt
        y = particles.y + k1[1] * 0.5 * particles.dt
    if len(k1) > 2:
        z = particles.z + k1[2] * 0.5 * particles.dt
    t = particles.t + 0.5 * particles.dt
    k2 = rhs(fieldset, t, z, y, x, particles)
    if len(k2) > 1:
        x = particles.x + k2[0] * 0.5 * particles.dt
        y = particles.y + k2[1] * 0.5 * particles.dt
    if len(k2) > 2:
        z = particles.z + k2[2] * 0.5 * particles.dt
    k3 = rhs(fieldset, t, z, y, x, particles)
    if len(k3) > 1:
        x = particles.x + k3[0] * particles.dt
        y = particles.y + k3[1] * particles.dt
    if len(k3) > 2:
        z = particles.z + k3[2] * particles.dt
    t = particles.t + particles.dt
    k4 = rhs(fieldset, t, z, y, x, particles)
    if len(k4) == 2:
        return ((k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0, (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0)
    if len(k4) == 3:
        return (
            (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0,
            (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0,
            (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0,
        )
