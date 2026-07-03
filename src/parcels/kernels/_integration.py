"""Collection of time integrators for use in Parcels Kernels"""


def RK2(fieldset, particles, rhs):
    z, lat, lon = particles.z, particles.lat, particles.lon
    fields = rhs(fieldset, particles.time, z, lat, lon, particles)
    if len(fields) == 1:
        raise NotImplementedError("RK2 integration is not implemented for Fields that return only one component.")
    if len(fields) > 1:
        lon = particles.lon + fields[0] * 0.5 * particles.dt
        lat = particles.lat + fields[1] * 0.5 * particles.dt
    if len(fields) > 2:
        z = particles.z + fields[2] * 0.5 * particles.dt
    t = particles.time + 0.5 * particles.dt
    return rhs(fieldset, t, z, lat, lon, particles)


def RK4(fieldset, particles, rhs):
    z, lat, lon = particles.z, particles.lat, particles.lon
    k1 = rhs(fieldset, particles.time, z, lat, lon, particles)
    if len(k1) == 1:
        raise NotImplementedError("RK4 integration is not implemented for Fields that return only one component.")
    if len(k1) > 1:
        lon = particles.lon + k1[0] * 0.5 * particles.dt
        lat = particles.lat + k1[1] * 0.5 * particles.dt
    if len(k1) > 2:
        z = particles.z + k1[2] * 0.5 * particles.dt
    t = particles.time + 0.5 * particles.dt
    k2 = rhs(fieldset, t, z, lat, lon, particles)
    if len(k2) > 1:
        lon = particles.lon + k2[0] * 0.5 * particles.dt
        lat = particles.lat + k2[1] * 0.5 * particles.dt
    if len(k2) > 2:
        z = particles.z + k2[2] * 0.5 * particles.dt
    t = particles.time + 0.5 * particles.dt
    k3 = rhs(fieldset, t, z, lat, lon, particles)
    if len(k3) > 1:
        lon = particles.lon + k3[0] * particles.dt
        lat = particles.lat + k3[1] * particles.dt
    if len(k3) > 2:
        z = particles.z + k3[2] * particles.dt
    t = particles.time + particles.dt
    k4 = rhs(fieldset, t, z, lat, lon, particles)
    if len(k4) == 2:
        return ((k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0, (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0)
    if len(k4) == 3:
        return (
            (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0,
            (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0,
            (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0,
        )
