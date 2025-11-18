# Interpolator explanation
Interpolation is an important functionality of Parcels. On this page we will discuss the way it is
implemented in **Parcels** and how to write a custom interpolator function.

```{note}
TODO: expand explanation (similar to Kernel loop explanation)
```

When we want to know the state of particles in an environmental field, such as temperature or velocity,
we _evaluate_ the `parcels.Field` at the particles real position in time and space (`t`, `z`, `lat`, `lon`). 
In Parcels we can do this using square brackets:
```
particles.temperature = fieldset.temperature[particles.time, particles.z, particles.lat, particles.lon]
```
The values of the `temperature` field at the particles' positions are determined using an interpolation
method. This interpolation method defines how the discretized values of the `parcels.Field` should 
relate to the value at any point within a grid cell. 

Each `parcels.Field` is defined on a (structured) `parcels.XGrid` or (unstructured) `parcels.UXGrid`.
The interpolation function takes information about the particles position relative to this grid (`grid_positions`),
as well as the `parcels.Field` values of the surrounding grid points in time and space, to calculate
the requested value at the particles location.

## Interpolator API
The interpolators included in Parcels are designed for common interpolation schemes in Parcels simulations. 
If we want to add a custom interpolation method, we need to look at the interpolator API:

We can write an interpolator function that takes a  `parcels.Field`, a dictionary with the `particle_positions` 
in real space and time, and a dictionary with the `grid_positions`.

The `particle_positions` dictionary contains:
```
particle_positions = {"time", time, "z", z, "lat", lat, "lon", lon}
```

For structured (`X`) grids, the `grid_positions` dictionary looks like:
```
grid_positions = {
    "T": {"index": ti, "bcoord": tau}, 
    "Z": {"index": zi, "bcoord": zeta}, 
    "Y": {"index": yi, "bcoord": eta}, 
    "X": {"index": xi, "bcoord", xsi},
}
```
where `index` is the grid index in the corresponding dimension, and `bcoord` is the barycentric coordinate in the grid cell.

For unstructured (`UX`) grids, the same dictionary is defined as:
```
grid_positions = {
    "T": {"index": ti, "bcoord": tau}, 
    "Z": {"index": zi, "bcoord": zeta}, 
    "FACE": {"index": fi, "bcoord": bcoord}
}
```