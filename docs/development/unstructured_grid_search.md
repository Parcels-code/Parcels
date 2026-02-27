# Unstructured Grid Search: Spatial Hashing with Morton Encoding

This page documents the algorithm used in Parcels to locate which grid cell a particle occupies on both curvilinear (`XGrid`) and unstructured (`UxGrid`) grids. The implementation lives in two files:

- `src/parcels/_core/spatialhash.py` — `SpatialHash` class and Morton encoding utilities
- `src/parcels/_core/index_search.py` — point-in-cell tests and the high-level search dispatch

---

## Motivation

On a rectilinear grid, finding which cell contains a particle is a trivial O(1) operation — compute an index from the coordinate directly. On curvilinear and unstructured grids, no such shortcut exists; each cell has an arbitrary shape and position.

The naive approach is an O(N) linear scan: compute the distance from the particle to each face centroid and find the minimum. For large meshes this becomes a bottleneck, and distance minimisation does not guarantee the nearest centroid is actually the containing cell.

Parcels uses **spatial hashing** to reduce this to an O(1) average-case lookup. The key idea is to impose a cheap, regular "hash grid" on top of the irregular source grid, assign each source-grid face to the hash cells its bounding box overlaps, and then use the hash grid to narrow candidate faces for any query point down to a small set before running an exact point-in-cell test. The key data structure that relates the cells in the hash grid to the source grid is the **hash table**. In Parcels, the hash table is a lookup table that takes as input a position and returns a short list of faces in a grid that likely contain the query point. Once a short list of candidate faces is obtained, Parcels then uses point-in-cell checks to locate which face a particle resides in.

The search methods that exist in the literature — KD-trees, BVH trees, quad-trees — give O(log N) query time. Spatial hashing achieves O(1) on average, which is particularly important when the same grid is queried millions of times per time step (one query per particle, per step).

---

## Algorithm Overview

The algorithm has two phases: **initialisation** (once, when the grid is first used) and **query** (once per particle per time step).

### Initialisation

1. Determine the bounding box of every face in the source grid.
2. Map each face's bounding box to a set of integer hash cell coordinates using **Morton encoding** (see below).
3. Sort all (face, morton-code) pairs by morton code and store them in a compact CSR-like structure: arrays of unique codes `keys`, their starting positions `starts`, hit counts `counts`, and the corresponding face indices `i`, `j`.

### Query

1. Convert each particle position to a Morton code using the same encoding as step 2 above.
2. Binary-search `keys` for the particle's code (`np.searchsorted`).
3. For each candidate face returned by the hash table, run an **exact point-in-cell test**.
4. Return the first face that passes, along with its local (computational) coordinates.

---

## Coordinate Systems

The hash grid is three-dimensional regardless of the source grid:

| Source grid | Mesh type | Hash-grid space                       |
| ----------- | --------- | ------------------------------------- |
| `XGrid`     | spherical | Cartesian unit cube (lon/lat → x,y,z) |
| `XGrid`     | flat      | 2-D lon/lat bounding box (z set to 0) |
| `UxGrid`    | spherical | Cartesian unit cube                   |
| `UxGrid`    | flat      | 2-D lon/lat bounding box              |

For spherical grids, node coordinates are converted from degrees to radians and then to Cartesian (x, y, z) on the unit sphere using `_latlon_rad_to_xyz` (`index_search.py:345`):

```
x = cos(lon) * cos(lat)
y = sin(lon) * cos(lat)
z = sin(lat)
```

The hash grid then spans the unit cube `[-1, 1]³`. Working in Cartesian space avoids the longitude wrap-around discontinuity that would otherwise cause the bounding boxes of cells crossing the antimeridian to erroneously span the entire domain.

For flat meshes the hash grid simply spans `[lon_min, lon_max] × [lat_min, lat_max]`, with z fixed at 0.

---

## Morton Encoding

Morton encoding is the core of the hashing function. It converts a 3-D floating-point coordinate into a single unsigned 32-bit integer in a way that preserves spatial locality — nearby points in 3-D space get similar Morton codes. This makes it an ideal hash function for spatial search.

The process has three steps, all implemented in `spatialhash.py`.

### Step 1: Quantization

**Function:** `quantize_coordinates` (`spatialhash.py:486`)

Each coordinate axis is mapped from its floating-point range `[min, max]` to an integer in `[0, bitwidth]` by:

```
xq = clip( floor( (x - xmin) / (xmax - xmin) * bitwidth ), 0, bitwidth )
```

With `bitwidth = 1023` (the default), each axis gets 1024 distinct integer levels. This is equivalent to overlaying a regular 1024×1024×1024 grid — the "hash grid" — on the domain. Each integer triple `(xq, yq, zq)` identifies one cell of this hash grid.

**Choosing bitwidth:** The default of 1023 was chosen because it fits in 10 bits per axis, and 3 × 10 = 30 bits, which fits exactly in a `uint32`. Increasing bitwidth beyond 1023 would require a `uint64` Morton code. In practice, 1023 provides more than enough resolution — for a global ocean model with ~1 million faces, the hash grid cells are already smaller than individual grid cells.

**Effect of domain extent:** If the quantization bounds `[xmin, xmax]` exactly match the source grid's coordinate range, then at `bitwidth = 1023` most hash cells contain at most one or two source faces. Expanding the bounds beyond the data range is equivalent to coarsening the hash grid, increasing the average number of candidates per query.

### Step 2: Bit Dilation

**Function:** `_dilate_bits` (`spatialhash.py:440`)

Before interleaving, each 10-bit integer is "dilated" so that its bits are spread out with two zero bits between each active bit:

```
Input:  b9 b8 b7 b6 b5 b4 b3 b2 b1 b0
Output: b9 0 0 b8 0 0 b7 0 0 ... b1 0 0 b0
```

This is done via a sequence of shift-and-mask operations operating on `uint32` values:

```python
n &= 0x000003FF          # keep only 10 bits
n = (n | (n << 16)) & 0xFF0000FF
n = (n | (n <<  8)) & 0x0300F00F
n = (n | (n <<  4)) & 0x030C30C3
n = (n | (n <<  2)) & 0x09249249
```

Each stage moves bits further apart. After five stages the 10 active bits are each separated by exactly two zeros, filling positions 0, 3, 6, 9, ..., 27 in the 30-bit result.

### Step 3: Bit Interleaving

**Function:** `_encode_quantized_morton3d` (`spatialhash.py:532`)

Once all three axes are dilated, the Morton code is formed by OR-ing the shifted dilated values:

```python
code = (dz << 2) | (dy << 1) | dx
```

The resulting bit layout (from LSB) is:

```
x0, y0, z0, x1, y1, z1, ..., x9, y9, z9
```

The combined encode function `_encode_morton3d` (`spatialhash.py:553`) chains all three steps (quantize → dilate → interleave) and is the function used during queries.

---

## Hash Table Construction

**Method:** `SpatialHash._initialize_hash_table` (`spatialhash.py:179`)
The hash table is built by iterating over every face in the source grid:

1. **Compute bounding boxes.** For each face, find the min and max of its node coordinates along each axis. For `XGrid`, this uses the four corner nodes at `[j,i]`, `[j,i+1]`, `[j+1,i+1]`, `[j+1,i]`. For `UxGrid`, it uses the nodes listed in `face_node_connectivity`.

2. **Quantize bounding box corners.** Both the lower-left and upper-right corners of each bounding box are quantized with `quantize_coordinates`. The difference `(xqhigh - xqlow + 1)` × `(yqhigh - yqlow + 1)` × `(zqhigh - zqlow + 1)` gives the number of hash cells the face overlaps.

3. **Generate one Morton code per (face, hash cell) pair.** For each face, every hash cell within its bounding box gets a Morton code. This is done vectorised: for each face `f` spanning a `nx × ny × nz` block of hash cells, `nx*ny*nz` Morton codes are generated by iterating over all `(xi, yi, zi)` offsets within the block. The result is a flat array `morton_codes` (one entry per face-hashcell overlap) and a parallel array `face_ids`.

4. **Sort by Morton code and compress.** `morton_codes` and `face_ids` are sorted together by code. `np.unique` then finds the unique codes and the start/count for each group, producing a CSR (Compressed Sparse Row) structure:
   - `keys` — sorted unique Morton codes
   - `starts` — index into `face_sorted` where each unique code's entries begin
   - `counts` — number of face entries for each unique code
   - `i`, `j` — the face indices (column, row) for each entry, in sorted order

This CSR layout enables O(1) hash table lookup via binary search on `keys`.

---

## Querying the Hash Table

**Method:** `SpatialHash.query` (`spatialhash.py:300`)

Given arrays of particle latitudes `y` and longitudes `x`:

1. **Convert to hash space.** For spherical grids, convert to Cartesian `(qx, qy, qz)`. For flat grids, use `(x, y, 0)` directly.

2. **Encode query Morton codes.** Call `_encode_morton3d` on the query coordinates.

3. **Binary search.** Use `np.searchsorted(keys, query_codes)` to find the position of each query code in the sorted unique-codes array. Queries whose code does not appear exactly in `keys` return no candidates (the hash cell the particle falls in has no registered faces).

4. **Gather candidates.** For each query with a valid hit, use `starts[pos]` and `counts[pos]` to gather the candidate face indices `(j_all, i_all)` from the sorted arrays. This is done fully vectorised using a CSR traversal with `np.repeat` and cumulative sums.

5. **Point-in-cell test.** Call `self._point_in_cell` (either `curvilinear_point_in_cell` or `uxgrid_point_in_cell`) on all candidates simultaneously. The first candidate per query that passes is taken as the result.

6. **Return.** Returns `(j_best, i_best, coords_best)`. For particles with no containing face found, `j_best` and `i_best` are `GRID_SEARCH_ERROR = -3`, and `coords_best` is `(-1, -1)`.

---

## Point-in-Cell Tests

### Curvilinear grids (`XGrid`)

**Function:** `curvilinear_point_in_cell` (`index_search.py:95`)

Each quadrilateral cell is parameterised by bilinear mapping from the unit square `[0,1]²` (computational coordinates `xsi`, `eta`) to physical space. The test solves the resulting quadratic equation:

```
P = a₀ + a₁·xsi + a₂·eta + a₃·xsi·eta
```

The coefficients `a` and `b` are derived from the cell's four corner longitudes and latitudes respectively. For spherical grids, the antimeridian is handled explicitly: cells spanning the ±180° boundary have their vertex longitudes adjusted so no cell has a longitude span greater than 180°.

The particle is inside the cell if `0 ≤ xsi ≤ 1` and `0 ≤ eta ≤ 1`.

### Unstructured grids (`UxGrid`)

**Function:** `uxgrid_point_in_cell` (`index_search.py:204`)

For triangular UxGrid faces, the test uses **barycentric coordinates**. For spherical geometry, the particle and face vertices are first converted to Cartesian coordinates, and the particle is projected onto the plane of the face (by removing its normal component) before computing barycentric coordinates. This makes the test valid for any triangular face on the sphere.

Barycentric coordinates are computed via **area ratios** in `_barycentric_coordinates` (`index_search.py:299`):

```
λ₀ = area(P, v₁, v₂) / area(v₀, v₁, v₂)
λ₁ = area(P, v₂, v₀) / area(v₀, v₁, v₂)
λ₂ = area(P, v₀, v₁) / area(v₀, v₁, v₂)
```

The particle is inside the face if `λ₀ ≥ 0`, `λ₁ ≥ 0`, `λ₂ ≥ 0`, and `λ₀ + λ₁ + λ₂ ≈ 1`.

```{note}
The current implementation is limited to triangular faces (K=3). Generalising to
n-sided convex polygons would require Wachspress barycentric coordinates. This is
noted as a TODO in `_barycentric_coordinates`.
```

---

## Integration with the Index Search

**Function:** `_search_indices_curvilinear_2d` (`index_search.py:148`)

The `SpatialHash` sits inside the broader index search used every time fields are interpolated onto particle positions. The flow is:

1. If a previous cell guess (`yi`, `xi`) is available (i.e., from the previous time step), first try `curvilinear_point_in_cell` at those indices. Particles that haven't moved far enough to leave their cell are resolved immediately without touching the hash table.

2. For particles not resolved by the guess (or with no guess), call `grid.get_spatial_hash().query(y, x)`.

3. The `SpatialHash` object is lazily constructed on the first call and cached on the grid object.

This two-stage approach (cheap guess check, then hash lookup) means that once particles have been located once, subsequent steps are fast for the common case where particles remain in the same cell.

---

## Design Notes and Limitations

**Hash grid resolution vs. memory.** The CSR structure stores one entry per (face, overlapping-hash-cell) pair. For a mesh with highly variable face sizes, large faces can overlap many hash cells, inflating the table size. The 10-bit quantisation (1024 levels per axis) bounds this: a face whose bounding box spans the whole domain contributes at most 1024³ ≈ 10⁹ entries, though in practice face bounding boxes are far smaller.

**Spherical geometry degeneracy.** Near the poles, lon/lat cells become highly elongated in lon/lat space. Working in Cartesian space mitigates this: all cells have similar extents in the unit cube, so the hash grid resolution is more uniform.

**Flat-grid z-coordinate.** For flat meshes, the z-coordinate is fixed at 0. The Morton code is still 3-D, but zq is always 0, so the z-bits of every code are 0. This wastes one third of the available code space but causes no correctness issues.
