import warnings

import numpy as np

from parcels._core.index_search import (
    GRID_SEARCH_ERROR,
    _latlon_rad_to_xyz,
    curvilinear_point_in_cell,
    uxgrid_point_in_cell,
)
from parcels._core.warnings import FieldSetWarning
from parcels._python import isinstance_noimport


class SpatialHash:
    """Custom data structure that is used for performing grid searches using Spatial Hashing. This class constructs an overlying
    uniformly spaced rectilinear grid, called the "hash grid" on top parcels.XGrid. It is particularly useful for grid searching
    on curvilinear grids. Faces in the Xgrid are related to the cells in the hash grid by determining the hash cells the bounding box
    of the unstructured face cells overlap with.

    Parameters
    ----------
    grid : parcels.XGrid
        Source grid used to construct the hash grid and hash table

    Note
    ----
    Does not currently support queries on periodic elements.
    """

    def __init__(
        self,
        grid,
        bitwidth=1023,
    ):
        if isinstance_noimport(grid, "XGrid"):
            self._point_in_cell = curvilinear_point_in_cell
        elif isinstance_noimport(grid, "UxGrid"):
            self._point_in_cell = uxgrid_point_in_cell
        else:
            raise ValueError("Expected `grid` to be a parcels.XGrid or parcels.UxGrid")

        self._source_grid = grid
        self._bitwidth = bitwidth  # Max integer to use per coordinate in quantization (10 bits = 0..1023)

        if isinstance_noimport(grid, "XGrid"):
            self._coord_dim = 2  # Number of computational coordinates is 2 (bilinear interpolation)
            if self._source_grid._mesh == "spherical":
                lon = np.deg2rad(self._source_grid.lon)
                lat = np.deg2rad(self._source_grid.lat)
                x, y, z = _latlon_rad_to_xyz(lat, lon)
                # Boundaries of the hash grid are the Cartesian bounding box of the
                # transformed grid, so that regional domains retain full quantization
                # resolution instead of spreading it over the whole unit cube
                self._xmin = x.min()
                self._xmax = x.max()
                self._ymin = y.min()
                self._ymax = y.max()
                self._zmin = z.min()
                self._zmax = z.max()
                _xbound = np.stack(
                    (
                        x[:-1, :-1],
                        x[:-1, 1:],
                        x[1:, 1:],
                        x[1:, :-1],
                    ),
                    axis=-1,
                )
                _ybound = np.stack(
                    (
                        y[:-1, :-1],
                        y[:-1, 1:],
                        y[1:, 1:],
                        y[1:, :-1],
                    ),
                    axis=-1,
                )
                _zbound = np.stack(
                    (
                        z[:-1, :-1],
                        z[:-1, 1:],
                        z[1:, 1:],
                        z[1:, :-1],
                    ),
                    axis=-1,
                )
                # Compute centroid locations of each cells
                self._xlow = np.min(_xbound, axis=-1)
                self._xhigh = np.max(_xbound, axis=-1)
                self._ylow = np.min(_ybound, axis=-1)
                self._yhigh = np.max(_ybound, axis=-1)
                self._zlow = np.min(_zbound, axis=-1)
                self._zhigh = np.max(_zbound, axis=-1)

                degenerate_mask = _find_degenerate_xgrid_faces(x, y, z)
                degeneracy_count = np.sum(degenerate_mask)
                if degeneracy_count > 0:
                    degen_locs = np.argwhere(degenerate_mask)  # shape (N, 2), columns are (j, i)
                    max_shown = np.min([degeneracy_count, 5])
                    shown = degen_locs[:max_shown]
                    loc_str = ", ".join(f"(j={loc[0]}, i={loc[1]})" for loc in shown)
                    warnings.warn(
                        f"Grid contains {degeneracy_count} degenerate faces that span a large portion of the "
                        "hash grid. This is most likely due to a mesh that isn't fully defined (e.g., points corresponding to land with lat/lon masked to 0). "
                        "You may experience runtime crashes due to high memory usage in the hash table or cell lookup failures for particles"
                        "in the vicinity of these degenerate cells."
                        f"First degenerate face location(s): {loc_str}.",
                        FieldSetWarning,
                        stacklevel=2,
                    )

            else:
                # Boundaries of the hash grid are the bounding box of the source grid
                self._xmin = self._source_grid.lon.min()
                self._xmax = self._source_grid.lon.max()
                self._ymin = self._source_grid.lat.min()
                self._ymax = self._source_grid.lat.max()
                # setting min and max below is needed for mesh="flat"
                self._zmin = 0.0
                self._zmax = 0.0
                x = self._source_grid.lon
                y = self._source_grid.lat

                _xbound = np.stack(
                    (
                        x[:-1, :-1],
                        x[:-1, 1:],
                        x[1:, 1:],
                        x[1:, :-1],
                    ),
                    axis=-1,
                )
                _ybound = np.stack(
                    (
                        y[:-1, :-1],
                        y[:-1, 1:],
                        y[1:, 1:],
                        y[1:, :-1],
                    ),
                    axis=-1,
                )
                # Compute bounding box of each face
                self._xlow = np.min(_xbound, axis=-1)
                self._xhigh = np.max(_xbound, axis=-1)
                self._ylow = np.min(_ybound, axis=-1)
                self._yhigh = np.max(_ybound, axis=-1)
                self._zlow = np.zeros_like(self._xlow)
                self._zhigh = np.zeros_like(self._xlow)

        elif isinstance_noimport(grid, "UxGrid"):
            self._coord_dim = grid.uxgrid.n_max_face_nodes  # Number of barycentric coordinates
            if self._source_grid._mesh == "spherical":
                # Reshape node coordinates to (nfaces, nnodes_per_face)
                nids = self._source_grid.uxgrid.face_node_connectivity.values
                lon = self._source_grid.uxgrid.node_lon.values[nids]
                lat = self._source_grid.uxgrid.node_lat.values[nids]
                _xbound, _ybound, _zbound = _latlon_rad_to_xyz(np.deg2rad(lat), np.deg2rad(lon))
                # Boundaries of the hash grid are the Cartesian bounding box of the
                # transformed grid, so that regional domains retain full quantization
                # resolution instead of spreading it over the whole unit cube
                self._xmin = _xbound.min()
                self._xmax = _xbound.max()
                self._ymin = _ybound.min()
                self._ymax = _ybound.max()
                self._zmin = _zbound.min()
                self._zmax = _zbound.max()

                # Compute bounding box of each face
                self._xlow = np.atleast_2d(np.min(_xbound, axis=-1))
                self._xhigh = np.atleast_2d(np.max(_xbound, axis=-1))
                self._ylow = np.atleast_2d(np.min(_ybound, axis=-1))
                self._yhigh = np.atleast_2d(np.max(_ybound, axis=-1))
                self._zlow = np.atleast_2d(np.min(_zbound, axis=-1))
                self._zhigh = np.atleast_2d(np.max(_zbound, axis=-1))

            else:
                # Boundaries of the hash grid are the bounding box of the source grid
                self._xmin = self._source_grid.uxgrid.node_lon.min().values
                self._xmax = self._source_grid.uxgrid.node_lon.max().values
                self._ymin = self._source_grid.uxgrid.node_lat.min().values
                self._ymax = self._source_grid.uxgrid.node_lat.max().values
                # setting min and max below is needed for mesh="flat"
                self._zmin = 0.0
                self._zmax = 0.0
                # Reshape node coordinates to (nfaces, nnodes_per_face)
                nids = self._source_grid.uxgrid.face_node_connectivity.values
                lon = self._source_grid.uxgrid.node_lon.values[nids]
                lat = self._source_grid.uxgrid.node_lat.values[nids]

                # Compute bounding box of each face
                self._xlow = np.atleast_2d(np.min(lon, axis=-1))
                self._xhigh = np.atleast_2d(np.max(lon, axis=-1))
                self._ylow = np.atleast_2d(np.min(lat, axis=-1))
                self._yhigh = np.atleast_2d(np.max(lat, axis=-1))
                self._zlow = np.zeros_like(self._xlow)
                self._zhigh = np.zeros_like(self._xlow)

        # Generate the mapping from the hash indices to unstructured grid elements
        self._hash_table = self._initialize_hash_table()

    def _initialize_hash_table(self):
        """Create a mapping that relates unstructured grid faces to hash indices by determining
        which faces overlap with which hash cells
        """
        # Quantize the bounding box in each direction
        xqlow, yqlow, zqlow = quantize_coordinates(
            self._xlow,
            self._ylow,
            self._zlow,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            self._bitwidth,
        )

        xqhigh, yqhigh, zqhigh = quantize_coordinates(
            self._xhigh,
            self._yhigh,
            self._zhigh,
            self._xmin,
            self._xmax,
            self._ymin,
            self._ymax,
            self._zmin,
            self._zmax,
            self._bitwidth,
        )
        xqlow = xqlow.ravel().astype(np.int32, copy=False)
        yqlow = yqlow.ravel().astype(np.int32, copy=False)
        zqlow = zqlow.ravel().astype(np.int32, copy=False)
        xqhigh = xqhigh.ravel().astype(np.int32, copy=False)
        yqhigh = yqhigh.ravel().astype(np.int32, copy=False)
        zqhigh = zqhigh.ravel().astype(np.int32, copy=False)
        nx = (xqhigh - xqlow + 1).astype(np.int32, copy=False)
        ny = (yqhigh - yqlow + 1).astype(np.int32, copy=False)
        nz = (zqhigh - zqlow + 1).astype(np.int32, copy=False)
        num_hash_per_face = (nx * ny * nz).astype(
            np.int32, copy=False
        )  # Since nx, ny, nz are in the 10-bit range, their product fits in int32
        # Sums over faces can exceed int32, so accumulate in int64
        total_hash_entries = int(num_hash_per_face.sum(dtype=np.int64))
        # Entry indices fit in int32 for all but extreme cases; fall back to int64 when needed
        idx_dtype = np.int64 if total_hash_entries > np.iinfo(np.int32).max else np.int32

        # Every face overlaps at least one hash cell (nx, ny, nz >= 1 since quantization
        # is monotone), and contributes one hash entry per cell of its quantized bounding
        # box. Entries are generated in face-major order: face_ids maps each entry to its
        # face, and intra enumerates the cells of that face's box (0..num_hash_per_face-1).
        nface = np.size(self._xlow)
        face_ids = np.repeat(np.arange(nface, dtype=np.uint32), num_hash_per_face)
        face_starts = np.concatenate(([0], np.cumsum(num_hash_per_face, dtype=np.int64)))[:-1]
        intra = np.arange(total_hash_entries, dtype=idx_dtype) - np.repeat(
            face_starts.astype(idx_dtype, copy=False), num_hash_per_face
        )

        # Derive (xi, yi, zi) cell offsets within each face's box from intra,
        # then shift by the per-face low corner to get quantized cell coordinates
        ny_nz = np.repeat(ny * nz, num_hash_per_face)
        nz_rep = np.repeat(nz, num_hash_per_face)

        xi = intra // ny_nz
        rem = intra % ny_nz
        yi = rem // nz_rep
        zi = rem % nz_rep

        xq = np.repeat(xqlow, num_hash_per_face) + xi
        yq = np.repeat(yqlow, num_hash_per_face) + yi
        zq = np.repeat(zqlow, num_hash_per_face) + zi

        # Vectorized morton encode for all entries at once, already in face-major order
        morton_codes = _encode_quantized_morton3d(xq, yq, zq)
        del intra, rem, xi, yi, zi, ny_nz, nz_rep, xq, yq, zq

        # Sort entries by morton code. Each (code, face) pair is fused into one uint64
        # with the code in the high 32 bits and the face id in the low 32 bits: unsigned
        # comparison then orders by code, with ties broken by ascending face id. Sorting
        # the fused array in place avoids the argsort permutation array and the gather
        # copies it would imply. Pairs are unique, so the ordering is deterministic.
        packed = morton_codes.astype(np.uint64)
        del morton_codes
        packed <<= np.uint64(32)
        np.bitwise_or(packed, face_ids, out=packed)
        del face_ids
        # Perform a single sort on the packed (morton_code | face_id ) list
        packed.sort()
        # Trunctating back to a uint32 keeps the lower 32 bits (the face_id's)
        face_sorted = packed.astype(np.uint32)
        # Purge the face ids from the packed list to retain only the morton codes
        packed >>= np.uint64(32)
        # Cast the morton codes back to uint32
        morton_codes_sorted = packed.astype(np.uint32)
        del packed
        j_sorted, i_sorted = np.unravel_index(face_sorted, self._xlow.shape)

        # Get a list of unique morton codes and their corresponding starts and counts (CSR format).
        # The codes are already sorted at this point, first by morton code, then by face_id
        # Starting indices of the matrix rows are located by finding indices where the morton codes differ
        starts = np.concatenate(([0], np.flatnonzero(morton_codes_sorted[1:] != morton_codes_sorted[:-1]) + 1))
        # The unique keys for the hash table are the unique morton codes
        keys = morton_codes_sorted[starts]
        # The number of faces per hash keys (morton codes) is easily calculated as the difference betwee the start values
        counts = np.diff(np.concatenate((starts, [morton_codes_sorted.size])))

        hash_table = {
            "keys": keys,
            "starts": starts,
            "counts": counts,
            "i": i_sorted,
            "j": j_sorted,
        }
        return hash_table

    def query(self, y, x):
        """
        Queries the hash table and finds the closes face in the source grid for each coordinate pair.

        Parameters
        ----------
        y : array_like
            y-coordinates in degrees (lat) to query of shape (N,) where N is the number of queries.
        x : array_like
            x-coordinates in degrees (lon) to query of shape (N,) where N is the number of queries.

        Returns
        -------
        j : ndarray, shape (N,)
            j-indices of the located face in the source grid for each query. If no face was found, GRID_SEARCH_ERROR is returned.
        i : ndarray, shape (N,)
            i-indices of the located face in the source grid for each query. If no face was found, GRID_SEARCH_ERROR is returned.
        coords : ndarray, shape (N, 2)
            The local coordinates (xsi, eta) of the located face in the source grid for each query.
            If no face was found, (-1.0, -1.0)
        """
        keys = self._hash_table["keys"]
        starts = self._hash_table["starts"]
        counts = self._hash_table["counts"]
        i = self._hash_table["i"]
        j = self._hash_table["j"]

        y = np.asarray(y)
        x = np.asarray(x)
        if self._source_grid._mesh == "spherical":
            # Convert coords to Cartesian coordinates (x, y, z)
            lat = np.deg2rad(y)
            lon = np.deg2rad(x)
            qx, qy, qz = _latlon_rad_to_xyz(lat, lon)
        else:
            # For Cartesian grids, use the coordinates directly
            qx = x
            qy = y
            qz = np.zeros_like(qx)

        query_codes = _encode_morton3d(
            qx, qy, qz, self._xmin, self._xmax, self._ymin, self._ymax, self._zmin, self._zmax
        ).ravel()
        num_queries = query_codes.size

        # Locate each query in the unique key array
        pos = np.searchsorted(keys, query_codes)  # pos is shape (num_queries,)

        # Valid hits: inside range with finite query coordinates and query codes give exact morton code match.
        valid = (pos < len(keys)) & np.isfinite(x) & np.isfinite(y)
        # Clip pos to valid range to avoid out-of-bounds indexing
        pos = np.clip(pos, 0, len(keys) - 1)
        # Further filter out false positives from searchsorted by checking for exact code match
        valid[valid] &= query_codes[valid] == keys[pos[valid]]

        # Pre-allocate i and j indices of the best match for each query
        # Default values to -1 (no match case)
        j_best = np.full(num_queries, GRID_SEARCH_ERROR, dtype=np.int32)
        i_best = np.full(num_queries, GRID_SEARCH_ERROR, dtype=np.int32)

        # How many matches each query has; hit_counts[i] is the number of hits for query i
        hit_counts = np.where(valid, counts[pos], 0).astype(np.int32)  # has shape (num_queries,)
        if hit_counts.sum() == 0:
            return (
                j_best.reshape(query_codes.shape),
                i_best.reshape(query_codes.shape),
                np.full((num_queries, self._coord_dim), -1.0, dtype=np.float32),
            )

        # Now, for each query, we need to gather the candidate (j,i) indices from the hash table
        # Each j,i pair needs to be repeated hit_counts[i] times, only when there are hits.

        # Boolean array for keeping track of which queries have candidates
        has_hits = hit_counts > 0  # shape (num_queries,), True for queries that had candidates

        # A quick lookup array that maps all candindates back to its query index
        q_index_for_candidate = np.repeat(
            np.arange(num_queries, dtype=np.int32), hit_counts
        )  # shape (hit_counts.sum(),)
        # Map all candidates to positions in the hash table
        hash_positions = pos[q_index_for_candidate]  # shape (hit_counts.sum(),)

        # Now that we have the positions in the hash table for each table, we can gather the (j,i) pairs for each candidate
        # We do this in a vectorized way by using a CSR-like approach
        # starts[pos[q_index_for_candidate]] gives the starting point in the hash table for each candidate
        # hit_counts gives the number of candidates for each query

        # We need to build an array that gives the offset within each query's candidates
        offsets = np.concatenate(([0], np.cumsum(hit_counts))).astype(np.int32)  # shape (num_queries+1,)
        total = int(offsets[-1])  # total number of candidates across all queries

        # Now, for each candidate, we need a simple array that tells us its "local candidate id" within its query
        # This way, we can easily take the starts[pos[q_index_for_candidate]] and add this local id to get the absolute index
        # We calculate this by computing the "global candidate number" (0..total-1) and subtracting the offsets of the corresponding query
        # This gives us an array that goes from 0..hit_counts[i]-1 for each query i
        intra = np.arange(total, dtype=np.int32) - np.repeat(offsets[:-1], hit_counts)  # shape (hit_counts.sum(),)

        # starts[pos[q_index_for_candidate]] + intra gives a list of positions in the hash table that we can
        # use to quickly gather the (i,j) pairs for each query
        source_idx = starts[hash_positions].astype(np.int32) + intra

        # Gather all candidate (j,i) pairs in one shot
        j_all = j[source_idx]
        i_all = i[source_idx]

        # Now we need to construct arrays that repeats the y and x coordinates for each candidate
        # to enable vectorized point-in-cell checks
        y_rep = np.repeat(y, hit_counts)  # shape (hit_counts.sum(),)
        x_rep = np.repeat(x, hit_counts)  # shape (hit_counts.sum(),)

        # For each query we perform a point in cell check.
        is_in_face, coordinates = self._point_in_cell(self._source_grid, y_rep, x_rep, j_all, i_all)

        coords_best = np.full((num_queries, coordinates.shape[1]), -1.0, dtype=np.float32)

        # For each query that has hits, we need to find the first candidate that was inside the face
        f_indices = np.flatnonzero(is_in_face)  # Indices of all faces that contained the point
        # For each true position, find which query it belongs to by searching offsets
        # Query index q satisfies offsets[q] <= pos < offsets[q+1].
        q = np.searchsorted(offsets[1:], f_indices, side="right")

        uniq_q, q_idx = np.unique(q, return_index=True)
        keep = has_hits[uniq_q]

        if keep.any():
            uniq_q = uniq_q[keep]
            pos_first = f_indices[q_idx[keep]]

            # Directly scatter: the code wants the first True inside each slice
            j_best[uniq_q] = j_all[pos_first]
            i_best[uniq_q] = i_all[pos_first]
            coords_best[uniq_q] = coordinates[pos_first]

        return (
            j_best.reshape(query_codes.shape),
            i_best.reshape(query_codes.shape),
            coords_best.reshape((num_queries, coordinates.shape[1])),
        )


def _dilate_bits(n):
    """
    Takes a 10-bit integer n, in range [0,1023], and "dilates" its bits so that
    there are two zeros between each bit of n in the result.

    This is a preparation step for building a 3D Morton code:
    - One axis (x, y, or z) is dilated like this.
    - Then the three dilated coordinates are bitwise interleaved
      to produce the full 30-bit Morton code.

    Example:
        Input n:  b9 b8 b7 b6 b5 b4 b3 b2 b1 b0
        Output:   b9 0 0 b8 0 0 b7 0 0 ... b0 0 0
    """
    n = np.asarray(n, dtype=np.uint32)

    # Step 1: Keep only the lowest 10 bits of n
    # Mask = 0x3FF = binary 11 1111 1111
    n &= np.uint32(0x000003FF)

    # Step 2: First spreading stage
    # Shift left by 16 and OR with original.
    # This spreads the bits apart, but introduces overlaps.
    # Mask 0xff0000ff clears out the unwanted overlaps.
    n = (n | (n << np.uint32(16))) & np.uint32(0xFF0000FF)

    # Step 3: Second spreading stage
    # Similar idea: shift left by 8, OR, then mask.
    # Now the bits are further separated.
    n = (n | (n << np.uint32(8))) & np.uint32(0x0300F00F)

    # Step 4: Third spreading stage
    # Shift by 4, OR, mask again.
    # At this point, there are 1 or 2 zeros between many of the bits.
    n = (n | (n << np.uint32(4))) & np.uint32(0x030C30C3)

    # Step 5: Final spreading stage
    # Shift by 2, OR, mask.
    # After this, each original bit is isolated with exactly two zeros
    # between it and the next bit, ready for 3D Morton interleaving.
    n = (n | (n << np.uint32(2))) & np.uint32(0x09249249)

    # Return the dilated value.
    return n


def _find_degenerate_xgrid_faces(x, y, z, threshold_factor=10):
    """Identify faces in structured grids that potentially span large portions of
    the underlying hash grid (e.g., due to the mesh being incomplete, with 0.0 stored in missing lon/lat points). Such degenerate faces can result in high memory requirements
    for the hash table.

    Detection is based on the maximum great-circle edge length of each cell.  A cell
    is flagged as degenerate when its longest edge exceeds ``threshold_factor`` multiplied by
    the 99th percentile of all edge lengths.

    Parameters
    ----------
    x, y, z : ndarray, shape (ny, nx)
        Unit-sphere Cartesian coordinates of the grid nodes.
    threshold_factor : float, optional
        Multiplier applied to the 99th-percentile edge length to set the threshold.
        Default is 10.

    Returns
    -------
    degenerate : ndarray of bool, shape (ny-1, nx-1)
        True for each cell whose maximum edge length exceeds the threshold.
    """

    # Chord length between two sets of points on the unit sphere, shape (ny-1, nx-1)
    def _chord(p1, p2):
        return np.sqrt(((p1 - p2) ** 2).sum(axis=-1))

    pts = np.stack([x, y, z], axis=-1)
    c00, c01 = pts[:-1, :-1], pts[:-1, 1:]
    c10, c11 = pts[1:, :-1], pts[1:, 1:]

    # Maximum chord across all four edges and both diagonals
    max_chord = np.maximum.reduce(
        [
            _chord(c00, c01),
            _chord(c10, c11),
            _chord(c00, c10),
            _chord(c01, c11),
            _chord(c00, c11),
            _chord(c01, c10),
        ]
    )

    threshold = threshold_factor * np.percentile(max_chord, 99)
    return max_chord > threshold


def quantize_coordinates(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth=1023):
    """
    Normalize (x, y, z) to [0, 1] over their bounding box, then quantize to 10 bits each (0..1023).

    Parameters
    ----------
    x, y, z : array_like
        Input coordinates to quantize. Can be scalars or arrays (broadcasting applies).
    xmin, xmax : float
        Minimum and maximum bounds for x coordinate.
    ymin, ymax : float
        Minimum and maximum bounds for y coordinate.
    zmin, zmax : float
        Minimum and maximum bounds for z coordinate.

    Returns
    -------
    xq, yq, zq : ndarray, dtype=uint32
        The quantized coordinates, each in range [0, 1023], same shape as the broadcasted input coordinates.
    """
    # Convert inputs to ndarray for consistent dtype/ufunc behavior.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # --- 1) Normalize each coordinate to [0, 1] over its bounding box. ---
    # Compute denominators once (avoid division by zero if bounds equal).
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    # Normalize to [0,1]; if a range is degenerate, map to 0 to avoid NaN/inf.
    with np.errstate(invalid="ignore"):
        xn = np.where(dx != 0, (x - xmin) / dx, 0.0)
        yn = np.where(dy != 0, (y - ymin) / dy, 0.0)
        zn = np.where(dz != 0, (z - zmin) / dz, 0.0)

    # --- 2) Quantize to (0..bitwidth). ---
    # Multiply by bitwidth, round down, and clip to be safe against overshoot.
    # Clip in float space before casting: out-of-range queries (e.g., points outside
    # a regional domain) would otherwise wrap around when a negative float is cast to uint32.
    # NaN queries produce arbitrary codes here; they are discarded downstream by the
    # finite-coordinate mask in SpatialHash.query.
    with np.errstate(invalid="ignore"):
        xq = np.clip(xn * bitwidth, 0, bitwidth).astype(np.uint32)
        yq = np.clip(yn * bitwidth, 0, bitwidth).astype(np.uint32)
        zq = np.clip(zn * bitwidth, 0, bitwidth).astype(np.uint32)

    return xq, yq, zq


def _encode_quantized_morton3d(xq, yq, zq):
    xq = np.asarray(xq)
    yq = np.asarray(yq)
    zq = np.asarray(zq)

    # --- 3) Bit-dilate each 10-bit number so each bit is separated by two zeros. ---
    # _dilate_bits maps:  b9..b0  ->  b9 0 0 b8 0 0 ... b0 0 0
    dx3 = _dilate_bits(xq).astype(np.uint32)
    dy3 = _dilate_bits(yq).astype(np.uint32)
    dz3 = _dilate_bits(zq).astype(np.uint32)

    # --- 4) Interleave the dilated bits into a single Morton code. ---
    # Bit layout (from LSB upward): x0,y0,z0, x1,y1,z1, ..., x9,y9,z9
    # We shift z's bits by 2, y's by 1, x stays at 0, then OR them together.
    # Cast to a wide type before shifting/OR to be safe when arrays are used.
    code = (dz3 << 2) | (dy3 << 1) | dx3

    # Since our compact type fits in 30 bits, uint32 is enough.
    return code.astype(np.uint32)


def _encode_morton3d(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth=1023):
    """
    Quantize (x, y, z) to 10 bits each (0..1023), dilate the bits so there are
    two zeros between successive bits, and interleave them into a 3D Morton code.

    Parameters
    ----------
    x, y, z : array_like
        Input coordinates to encode. Can be scalars or arrays (broadcasting applies).
    xmin, xmax : float
        Minimum and maximum bounds for x coordinate.
    ymin, ymax : float
        Minimum and maximum bounds for y coordinate.
    zmin, zmax : float
        Minimum and maximum bounds for z coordinate.

    Returns
    -------
    code : ndarray, dtype=uint32
        The resulting Morton codes, same shape as the broadcasted input coordinates.

    Notes
    -----
    - Works with scalars or NumPy arrays (broadcasting applies).
    - Output is up to 30 bits returned as uint32.
    """
    # Convert inputs to ndarray for consistent dtype/ufunc behavior.
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    xq, yq, zq = quantize_coordinates(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax, bitwidth)

    # --- 3) Bit-dilate each 10-bit number so each bit is separated by two zeros. ---
    # _dilate_bits maps:  b9..b0  ->  b9 0 0 b8 0 0 ... b0 0 0
    dx3 = _dilate_bits(xq).astype(np.uint32)
    dy3 = _dilate_bits(yq).astype(np.uint32)
    dz3 = _dilate_bits(zq).astype(np.uint32)

    # --- 4) Interleave the dilated bits into a single Morton code. ---
    # Bit layout (from LSB upward): x0,y0,z0, x1,y1,z1, ..., x9,y9,z9
    # We shift z's bits by 2, y's by 1, x stays at 0, then OR them together.
    # Cast to a wide type before shifting/OR to be safe when arrays are used.
    code = (dz3 << 2) | (dy3 << 1) | dx3

    # Since our compact type fits in 30 bits, uint32 is enough.
    return code.astype(np.uint32)
