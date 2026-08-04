# Remove xgcm Dependency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove xgcm as a runtime dependency by replacing the XgcmLike adapter layer with direct use of SGRID metadata throughout `xgrid.py`.

**Architecture:** Three sequential tasks: (1) add new SGRID-native types and helpers as pure additions, (2) refactor xgrid.py to use them instead of the adapter layer, (3) delete the now-dead bridge functions and fix the two test files that called xgcm directly.

**Tech Stack:** Python 3.11+, xarray, parcels SGRID metadata (`SGrid2DMetadata`, `FaceNodePadding`, `Padding`)

## Global Constraints

- Never import `xgcm` outside of `TYPE_CHECKING` blocks — and those blocks are being removed too
- Preserve all existing function names (they are imported by `model.py` and may be used elsewhere)
- All commits must include `Co-authored-by: Claude <noreply@anthropic.com>`
- Run `pytest tests/sgrid/ tests/datasets/ -x` after each task before committing

---

## File Map

| File                                | Change                                                                                 |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `src/parcels/_typing.py`            | Add `GridPosition`; remove `XgcmAxisPosition`, `XgcmAxes`, xgcm TYPE_CHECKING import   |
| `src/parcels/_sgrid/accessor.py`    | Add `get_dim_position()`                                                               |
| `src/parcels/_core/xgrid.py`        | Major refactor — remove adapter classes, update all functions to use `SGrid2DMetadata` |
| `src/parcels/_sgrid/core.py`        | Remove `SGRID_PADDING_TO_XGCM_POSITION` and `xgcm_parse_sgrid()`                       |
| `src/parcels/_sgrid/__init__.py`    | Remove `xgcm_parse_sgrid` from imports and `__all__`                                   |
| `tests/sgrid/test_sgrid.py`         | Remove xgcm import + two xgcm tests; add `test_get_dim_position`                       |
| `tests/datasets/test_structured.py` | Replace `xgcm.Grid` calls with SGRID-native assertions                                 |
| `pyproject.toml`                    | Remove `"xgcm >=0.9.0"`                                                                |

---

### Task 1: Add `GridPosition` type and `get_dim_position()` helper

Pure additions — nothing is deleted. All existing code continues to work.

**Files:**

- Modify: `src/parcels/_typing.py`
- Modify: `src/parcels/_sgrid/accessor.py`
- Modify: `tests/sgrid/test_sgrid.py`

**Interfaces:**

- Produces: `GridPosition = Literal["face"] | Padding` in `_typing.py`
- Produces: `get_dim_position(grid: SGrid2DMetadata, dim: str) -> Literal["face"] | Padding` in `_sgrid/accessor.py`

- [ ] **Step 1: Write failing tests for `get_dim_position`**

Add to `tests/sgrid/test_sgrid.py` (after the existing imports, before existing test functions):

```python
from parcels._sgrid.accessor import get_dim_position


def test_get_dim_position_face_dims():
    """Face dimensions return 'face'."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=False, with_node_coordinates=False)
    # face_dimensions = (FaceNodePadding("face_dimension1", "node_dimension1", Padding.LOW), ...)
    assert get_dim_position(metadata, "face_dimension1") == "face"
    assert get_dim_position(metadata, "face_dimension2") == "face"


def test_get_dim_position_node_dims():
    """Node dimensions return their Padding value."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=False, with_node_coordinates=False)
    assert get_dim_position(metadata, "node_dimension1") == sgrid.Padding.LOW
    assert get_dim_position(metadata, "node_dimension2") == sgrid.Padding.LOW


def test_get_dim_position_vertical():
    """Vertical face and node dimensions are handled."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=True, with_node_coordinates=False)
    # vertical_dimensions = (FaceNodePadding("vertical_dimensions_dim1", "vertical_dimensions_dim2", Padding.LOW),)
    assert get_dim_position(metadata, "vertical_dimensions_dim1") == "face"
    assert get_dim_position(metadata, "vertical_dimensions_dim2") == sgrid.Padding.LOW


def test_get_dim_position_unknown_dim_raises():
    """Unknown dimensions raise ValueError."""
    metadata = create_example_grid2dmetadata(with_vertical_dimensions=False, with_node_coordinates=False)
    with pytest.raises(ValueError, match="not a spatial SGRID dimension"):
        get_dim_position(metadata, "nonexistent_dim")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/sgrid/test_sgrid.py::test_get_dim_position_face_dims -x -v
```

Expected: `ImportError` or `AttributeError` — `get_dim_position` does not exist yet.

- [ ] **Step 3: Add `GridPosition` to `_typing.py`**

In `src/parcels/_typing.py`, after the existing imports block, add:

```python
from parcels._sgrid.core import Padding
```

Then after `XgcmAxisDirection = CfAxisSpatial | Literal["T"]`, add:

```python
GridPosition = Literal["face"] | Padding
```

No existing lines are removed yet — that happens in Task 2.

- [ ] **Step 4: Add `get_dim_position` to `accessor.py`**

In `src/parcels/_sgrid/accessor.py`, add this function after `_get_axis_info`:

```python
def get_dim_position(grid: SGrid2DMetadata, dim: str) -> "Literal['face'] | Padding":
    """Returns 'face' if dim is a face dimension, or the Padding value if it is a node dimension.

    Replaces xgcm's position string vocabulary ('center', 'left', 'right', 'inner', 'outer')
    with SGRID-native types.
    """
    axis_info = _get_axis_info(grid)
    if dim not in axis_info:
        raise ValueError(f"Dimension {dim!r} is not a spatial SGRID dimension in this grid.")
    fnp, is_node = axis_info[dim]
    return fnp.padding if is_node else "face"
```

Add the `Literal` import to the existing `from typing import Any, Literal, cast` line in accessor.py (it already has `Literal` — confirm and leave as-is if so).

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/sgrid/test_sgrid.py::test_get_dim_position_face_dims tests/sgrid/test_sgrid.py::test_get_dim_position_node_dims tests/sgrid/test_sgrid.py::test_get_dim_position_vertical tests/sgrid/test_sgrid.py::test_get_dim_position_unknown_dim_raises -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Run full test subset to confirm no regressions**

```bash
pytest tests/sgrid/ tests/datasets/ -x -q
```

Expected: all pass (same as before this task).

- [ ] **Step 7: Commit**

```bash
git add src/parcels/_typing.py src/parcels/_sgrid/accessor.py tests/sgrid/test_sgrid.py
git commit -m "feat: add GridPosition type and get_dim_position() SGRID helper

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 2: Refactor `xgrid.py` to use SGRID metadata directly

Remove `XgcmLikeAxis`, `XgcmLikeGrid`, `construct_xgcm_axes_object`, and `self.xgcm_grid`. Update every function in `xgrid.py` that referenced the adapter layer to use `SGrid2DMetadata` / `FaceNodePadding` / `Padding` directly. Also clean up the now-dead type aliases from `_typing.py`.

**Files:**

- Modify: `src/parcels/_core/xgrid.py`
- Modify: `src/parcels/_typing.py`

**Interfaces:**

- Consumes: `get_dim_position(grid, dim)` from Task 1
- Consumes: `GridPosition` from Task 1
- Consumes: `_get_dim_to_axis_mapping(metadata)` from `_sgrid/accessor.py` (already imported)
- Produces: All public function signatures updated to take `SGrid2DMetadata` instead of `XgcmAxes`/`xgcm.Grid`

- [ ] **Step 1: Confirm baseline tests pass before touching anything**

```bash
pytest tests/sgrid/ tests/datasets/ -x -q
```

Expected: all pass. If not, do not proceed — fix the cause first.

- [ ] **Step 2: Rewrite `xgrid.py` imports and module-level constants**

At the top of `src/parcels/_core/xgrid.py`, make these changes:

Remove the `TYPE_CHECKING` block entirely:

```python
# DELETE these lines:
if TYPE_CHECKING:
    import xgcm.axis
```

Remove the `_DEFAULT_XGCM_KWARGS` constant:

```python
# DELETE this line:
_DEFAULT_XGCM_KWARGS: dict[str, Any] = {"padding": "fill"}
```

Remove the import of `SGRID_PADDING_TO_XGCM_POSITION`:

```python
# DELETE this line:
from parcels._sgrid.core import SGRID_PADDING_TO_XGCM_POSITION
```

Add `get_dim_position` to the existing accessor import:

```python
# Change this:
from parcels._sgrid.accessor import _get_dim_to_axis_mapping
# To:
from parcels._sgrid.accessor import _get_dim_to_axis_mapping, get_dim_position
```

Add `FaceNodePadding` and `Padding` to the sgrid imports. The existing `import parcels._sgrid as sgrid` is kept — use `sgrid.Padding` and `sgrid.FaceNodePadding` in the code below.

Also update the `_typing` import to add `GridPosition`:

```python
# The existing line:
import parcels._typing as ptyping
# No change to the line itself — access GridPosition as ptyping.GridPosition
```

Remove `Any` from the typing imports if it's only used by `_DEFAULT_XGCM_KWARGS` (check first — if used elsewhere, keep it).

- [ ] **Step 3: Replace standalone helper functions**

Replace `get_cell_count_along_dim` (lines 29-33):

```python
def get_cell_count_along_dim(ds: xr.Dataset, fnp: sgrid.FaceNodePadding) -> int:
    return ds[fnp.face].size - 1
```

Replace `get_time` (lines 36-37):

```python
def get_time(ds: xr.Dataset, time_dim: str) -> npt.NDArray:
    return ds[time_dim].values
```

Replace `_get_xgrid_axes` (lines 40-42):

```python
def _get_xgrid_axes(metadata: sgrid.SGrid2DMetadata, ds_dims: set[str]) -> list[ptyping.XgridAxis]:
    dim_to_axis = _get_dim_to_axis_mapping(metadata)
    present = {axis for dim, axis in dim_to_axis.items() if dim in ds_dims}
    return sorted(present, key=_XGRID_AXES_ORDERING.index)
```

Replace `assert_all_field_dims_have_axis` (lines 54-76). Note: `model.py` imports this by name so the name must not change:

```python
def assert_all_field_dims_have_axis(da: xr.DataArray, metadata: sgrid.SGrid2DMetadata) -> None:
    dim_to_axis = _get_dim_to_axis_mapping(metadata) | {"time": "T"}
    ax_dims = [(dim_to_axis.get(str(dim)), str(dim)) for dim in da.dims]

    for ax, dim_name in ax_dims:
        if ax is None:
            raise ValueError(
                f'Dimension "{dim_name}" has no axis attribute. '
                f'HINT: You may want to add an {{"axis": A}} to your DataSet["{dim_name}"], where A is one of "X", "Y", "Z" or "T"'
            )

    seen_axes: dict[str, str] = {}
    for ax, dim_name in ax_dims:
        if ax in seen_axes:
            raise ValueError(
                f"Two dimensions ({dim_name!r} and {seen_axes[ax]!r}) provide values in the axis direction {ax!r}. "
                "This is not possible, a field cannot have two dimensions on a single axis."
            )
        seen_axes[ax] = dim_name
    assert len(ax_dims) <= 4, (
        "The input dataset appears to have more than 4 dimensions after conversion. Execution should never reach this point. Please file an issue sharing more about your input dataset."
    )
```

- [ ] **Step 4: Delete the adapter classes and construct function**

Delete `XgcmLikeAxis` (lines 116-118), `XgcmLikeGrid` (lines 121-129), and `construct_xgcm_axes_object` (lines 132-159) entirely.

- [ ] **Step 5: Rewrite `XGrid.__init__`**

Replace the `__init__` body from `grid = XgcmLikeGrid(...)` onward:

```python
def __init__(self, model_data: xr.Dataset, mesh: Literal["flat", "spherical"] | SphericalMesh):
    self.sgrid_metadata = model_data.sgrid.metadata
    self._ds = model_data
    self._mesh = get_mesh(mesh)
    self._spatialhash = None
    ds = model_data

    if "lon" in ds:
        ds.set_coords("lon")
    if "lat" in ds:
        ds.set_coords("lat")

    axes = self.axes  # uses _get_xgrid_axes(self.sgrid_metadata, set(self._ds.dims))
    if len(set(axes) & {"X", "Y"}) > 0:
        assert_valid_lat_lon(ds["lat"], ds["lon"], self.sgrid_metadata)

    if "Z" in axes:
        assert_valid_depth(ds["depth"])

    self._ds = ds
```

- [ ] **Step 6: Rewrite `XGrid` properties and methods**

Replace the `axes` property:

```python
@property
def axes(self) -> list[ptyping.XgridAxis]:
    return _get_xgrid_axes(self.sgrid_metadata, set(self._ds.dims))
```

Replace the `lon` property:

```python
@property
def lon(self):
    """
    Note
    ----
    Included for compatibility with v3 codebase. May be removed in future.
    TODO v4: Evaluate
    """
    if "X" not in self.axes:
        return np.zeros(1)
    if is_dask_collection(self._ds["lon"].data):
        self._ds["lon"].load()
    return self._ds["lon"].values
```

Replace the `lat` property:

```python
@property
def lat(self):
    """
    Note
    ----
    Included for compatibility with v3 codebase. May be removed in future.
    TODO v4: Evaluate
    """
    if "Y" not in self.axes:
        return np.zeros(1)
    if is_dask_collection(self._ds["lat"].data):
        self._ds["lat"].load()
    return self._ds["lat"].values
```

Replace the `depth` property:

```python
@property
def depth(self):
    """
    Note
    ----
    Included for compatibility with v3 codebase. May be removed in future.
    TODO v4: Evaluate
    """
    if "Z" not in self.axes:
        return np.zeros(1)
    return self._ds["depth"].values
```

Replace the `_datetimes` property:

```python
@property
def _datetimes(self):
    if "time" not in self._ds.dims:
        return np.zeros(1)
    return get_time(self._ds, "time")
```

Replace `get_axis_dim`:

```python
def get_axis_dim(self, axis: ptyping.XgridAxis) -> int:
    if axis not in self.axes:
        raise ValueError(f"Axis {axis!r} is not part of this grid. Available axes: {self.axes}")

    fnp_x, fnp_y = self.sgrid_metadata.face_dimensions
    if axis == "X":
        return get_cell_count_along_dim(self._ds, fnp_x)
    if axis == "Y":
        return get_cell_count_along_dim(self._ds, fnp_y)
    # axis == "Z"
    assert self.sgrid_metadata.vertical_dimensions is not None
    return get_cell_count_along_dim(self._ds, self.sgrid_metadata.vertical_dimensions[0])
```

Replace `localize`:

```python
def localize(
    self, position: dict[ptyping.XgridAxis, tuple[int, float]], dims: list[str]
) -> dict[str, tuple[int, float]]:
    """
    Uses the grid context (i.e., the staggering of the grid) to convert a position relative
    to the F-points in the grid to a position relative to the staggered grid the array
    of interest is defined on.

    Uses dimensions of the DataArray to determine the staggered grid.

    WARNING: This API is unstable and subject to change in future versions.

    Parameters
    ----------
    position : dict
        A mapping of the axis to a tuple of (index, barycentric coordinate) for the
        F-points in the grid.
    dims : list[str]
        A list of dimension names that the DataArray is defined on. This is used to determine
        the staggering of the grid and which axis each dimension corresponds to.

    Returns
    -------
    dict[str, tuple[int, float]]
        A mapping of the dimension names to a tuple of (index, barycentric coordinate) for
        the staggered grid the DataArray is defined on.

    Example
    -------
    >>> position = {'X': (5, 0.51), 'Y': (
        10, 0.25), 'Z': (3, 0.75)}
    >>> dims = ['time', 'depth', 'YC', 'XC']
    >>> grid.localize(position, dims)
    {'depth': (3, 0.75), 'YC': (9, 0.75), 'XC': (5, 0.01)}
    """
    dim_to_axis = _get_dim_to_axis_mapping(self.sgrid_metadata) | {"time": "T"}
    axis_to_var = {dim_to_axis[dim]: dim for dim in dims if dim in dim_to_axis}
    var_positions = {
        axis: get_dim_position(self.sgrid_metadata, dim)
        for axis, dim in axis_to_var.items()
        if axis != "T"
    }
    return {
        axis_to_var[axis]: _convert_center_pos_to_fpoint(
            index=index,
            bcoord=bcoord,
            position=var_positions[axis],
            f_point_position=self._fpoint_info[axis],
        )
        for axis, (index, bcoord) in position.items()
    }
```

Replace `_fpoint_info`:

```python
@cached_property
def _fpoint_info(self) -> dict[ptyping.XgridAxis, sgrid.Padding]:
    """Returns a mapping of the spatial axes in the Grid to their Padding values (node positions)."""
    metadata = self.sgrid_metadata
    fnp_x, fnp_y = metadata.face_dimensions
    result: dict[ptyping.XgridAxis, sgrid.Padding] = {}
    axes = self.axes
    if "X" in axes:
        result["X"] = fnp_x.padding
    if "Y" in axes:
        result["Y"] = fnp_y.padding
    if "Z" in axes and metadata.vertical_dimensions:
        result["Z"] = metadata.vertical_dimensions[0].padding
    return result
```

Replace `get_axis_dim_mapping`:

```python
def get_axis_dim_mapping(self, dims: Sequence[Hashable]) -> dict[ptyping.XgridAxis, str]:
    """
    Maps xarray dimension names to their corresponding axis (X, Y, Z).

    WARNING: This API is unstable and subject to change in future versions.

    Parameters
    ----------
    dims : Sequence[Hashable]
        Sequence of xarray dimension names

    Returns
    -------
    dict[_XGRID_AXES, str]
        Dictionary mapping axes (X, Y, Z) to their corresponding dimension names

    Examples
    --------
    >>> grid.get_axis_dim_mapping(['time', 'lat', 'lon'])
    {'Y': 'lat', 'X': 'lon'}

    Notes
    -----
    Only returns mappings for spatial axes (X, Y, Z) that are present in the grid.
    """
    dim_to_axis = _get_dim_to_axis_mapping(self.sgrid_metadata)
    result = {}
    for dim in dims:
        axis = dim_to_axis.get(str(dim))
        if axis in self.axes:
            result[cast(ptyping.XgridAxis, axis)] = str(dim)
    return result
```

- [ ] **Step 7: Rewrite module-level functions below `XGrid`**

Replace `get_axis_from_dim_name` (lines 459-464):

```python
def get_axis_from_dim_name(metadata: sgrid.SGrid2DMetadata, dim: Hashable) -> ptyping.XgcmAxisDirection | None:
    """For a given dimension name in a grid, returns the direction axis it is on."""
    dim_to_axis = _get_dim_to_axis_mapping(metadata) | {"time": "T"}
    return dim_to_axis.get(str(dim))
```

Replace `get_xgcm_position_from_dim_name` (lines 467-473). Keep the name as-is since it may be imported externally:

```python
def get_xgcm_position_from_dim_name(metadata: sgrid.SGrid2DMetadata, dim: str) -> ptyping.GridPosition | None:
    """For a given dimension, returns the GridPosition of the variable in the grid."""
    try:
        return get_dim_position(metadata, dim)
    except ValueError:
        return None
```

Replace `assert_all_dimensions_correspond_with_axis` (lines 477-484):

```python
def assert_all_dimensions_correspond_with_axis(da: xr.DataArray, metadata: sgrid.SGrid2DMetadata) -> None:
    dim_to_axis = _get_dim_to_axis_mapping(metadata)
    for dim in da.dims:
        if dim not in dim_to_axis:
            raise ValueError(
                f"Dimension {dim!r} for DataArray {da.name!r} with dims {da.dims} is not associated with a direction on the provided grid."
            )
```

Replace `assert_valid_field_array` (lines 487-509):

```python
def assert_valid_field_array(da: xr.DataArray, metadata: sgrid.SGrid2DMetadata):
    """
    Asserts that for a data array:
    - All dimensions are associated with a direction on the grid
    - These directions are T, Z, Y, X and the array is ordered as T, Z, Y, X
    """
    dim_to_axis = _get_dim_to_axis_mapping(metadata) | {"time": "T"}

    for dim in da.dims:
        if dim not in dim_to_axis:
            raise ValueError(
                f"Dimension {dim!r} for DataArray {da.name!r} with dims {da.dims} is not associated with a direction on the provided grid."
            )

    dim_to_axis_for_da = {dim: dim_to_axis[dim] for dim in da.dims}
    dim_to_axis_for_da = cast(dict[Hashable, ptyping.XgcmAxisDirection], dim_to_axis_for_da)

    if set(dim_to_axis_for_da.values()) != {"T", "Z", "Y", "X"}:
        raise ValueError(
            f"DataArray {da.name!r} with dims {da.dims} has directions {tuple(dim_to_axis_for_da.values())}."
            "Expected directions of 'T', 'Z', 'Y', and 'X'."
        )

    if list(dim_to_axis_for_da.values()) != ["T", "Z", "Y", "X"]:
        raise ValueError(
            f"Dimension order for array {da.name!r} is not valid. Got {tuple(dim_to_axis_for_da.keys())} with associated directions of {tuple(dim_to_axis_for_da.values())}.  Expected directions of ('T', 'Z', 'Y', 'X'). Transpose your array accordingly."
        )
```

Replace `assert_valid_lat_lon` (lines 512-579):

```python
def assert_valid_lat_lon(da_lat, da_lon, metadata: sgrid.SGrid2DMetadata):
    """
    Asserts that the provided longitude and latitude DataArrays are defined appropriately
    on the F points to match the internal representation in Parcels.

    - Longitude and latitude must be 1D or 2D (both must have the same dimensionality)
    - Both are defined on the node points (i.e., not the face/center)
    - If 1D:
      - Longitude is associated with the X axis
      - Latitude is associated with the Y axis
    - If 2D:
      - Lon and lat are defined on the same dimensions
      - Lon and lat are transposed such they're Y, X
    """
    assert_all_dimensions_correspond_with_axis(da_lon, metadata)
    assert_all_dimensions_correspond_with_axis(da_lat, metadata)

    for dim in da_lon.dims:
        if get_dim_position(metadata, dim) == "face":
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} is defined on the center of the grid, but must be defined on the F points."
            )
    for dim in da_lat.dims:
        if get_dim_position(metadata, dim) == "face":
            raise ValueError(
                f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} is defined on the center of the grid, but must be defined on the F points."
            )

    if da_lon.ndim != da_lat.ndim:
        raise ValueError(
            f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} have different dimensionalities."
        )
    if da_lon.ndim not in (1, 2):
        raise ValueError(
            f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be 1D or 2D."
        )

    dim_to_axis = _get_dim_to_axis_mapping(metadata)

    if da_lon.ndim == 1:
        if dim_to_axis.get(da_lon.dims[0]) != "X":
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} is not associated with the X axis."
            )
        if dim_to_axis.get(da_lat.dims[0]) != "Y":
            raise ValueError(
                f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} is not associated with the Y axis."
            )

        if not np.all(np.diff(da_lon.values) > 0):
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} must be strictly increasing."
            )
        if not np.all(np.diff(da_lat.values) > 0):
            raise ValueError(f"Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be strictly increasing.")

    if da_lon.ndim == 2:
        if da_lon.dims != da_lat.dims:
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be defined on the same dimensions."
            )

        lon_axes = [dim_to_axis.get(dim) for dim in da_lon.dims]
        if lon_axes != ["Y", "X"]:
            raise ValueError(
                f"Longitude DataArray {da_lon.name!r} with dims {da_lon.dims} and Latitude DataArray {da_lat.name!r} with dims {da_lat.dims} must be defined on the X and Y axes and transposed to have dimensions in order of Y, X."
            )
```

Replace `_convert_center_pos_to_fpoint` (lines 590-616):

```python
def _convert_center_pos_to_fpoint(
    *,
    index: int,
    bcoord: float,
    position: ptyping.GridPosition,
    f_point_position: sgrid.Padding,
) -> tuple[int, float]:
    """Converts a physical position relative to the cell edges defined in the grid to be relative to the center point.

    This is used to "localize" a position to be relative to the staggered grid at which the field is defined, so that
    it can be easily interpolated.

    This also handles different model input cell edges and centers are staggered in different directions (e.g., with NEMO and MITgcm).
    """
    if position != "face":  # Data is already defined on the F points
        return index, bcoord

    bcoord = bcoord - 0.5
    if bcoord < 0:
        bcoord += 1.0
        index -= 1

    # Correct relative to the f-point position
    # Padding.BOTH was "inner", Padding.LOW was "right" in xgcm vocabulary
    if f_point_position in (sgrid.Padding.BOTH, sgrid.Padding.LOW):
        index += 1

    return index, bcoord
```

- [ ] **Step 8: Clean up `_typing.py`**

Remove these lines from `src/parcels/_typing.py`:

```python
# DELETE the TYPE_CHECKING block for xgcm:
if TYPE_CHECKING:
    import xgcm

# DELETE these two type aliases:
XgcmAxisPosition = Literal["center", "left", "right", "inner", "outer"]
XgcmAxes = Mapping[XgcmAxisDirection, "xgcm.Axis"]
```

Also remove `Mapping` from the `collections.abc` import if it is now unused (check by searching for other uses of `Mapping` in the file).

- [ ] **Step 9: Run tests**

```bash
pytest tests/sgrid/ tests/datasets/ -x -q
```

Expected: all pass. If any fail, fix before proceeding.

- [ ] **Step 10: Commit**

```bash
git add src/parcels/_core/xgrid.py src/parcels/_typing.py
git commit -m "refactor: replace XgcmLike adapter layer with direct SGRID metadata usage

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 3: Remove bridge functions, fix tests, remove xgcm dep

Delete the now-dead bridge code in `_sgrid/core.py`, fix the two test files, and drop xgcm from `pyproject.toml`.

**Files:**

- Modify: `src/parcels/_sgrid/core.py`
- Modify: `src/parcels/_sgrid/__init__.py`
- Modify: `tests/sgrid/test_sgrid.py`
- Modify: `tests/datasets/test_structured.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: `get_dim_position` from Task 1

- [ ] **Step 1: Remove `SGRID_PADDING_TO_XGCM_POSITION` and `xgcm_parse_sgrid` from `_sgrid/core.py`**

Delete lines 41-47 (the `SGRID_PADDING_TO_XGCM_POSITION` dict):

```python
# DELETE:
SGRID_PADDING_TO_XGCM_POSITION = {
    Padding.LOW: "right",
    Padding.HIGH: "left",
    Padding.BOTH: "inner",
    Padding.NONE: "outer",
    # "center" position is not used in SGrid, in SGrid this would just be the edges/faces themselves
}
```

Delete lines 470-492 (the `xgcm_parse_sgrid` function):

```python
# DELETE:
def xgcm_parse_sgrid(ds: xr.Dataset):
    # Function similar to that provided in `xgcm.metadata_parsers.
    # Might at some point be upstreamed to xgcm directly
    grid = ds.sgrid.metadata
    ...
    return (ds, {"coords": xgcm_coords})
```

- [ ] **Step 2: Update `_sgrid/__init__.py`**

Remove `xgcm_parse_sgrid` from the import and `__all__`:

```python
# Change from:
from .core import (
    FaceNodePadding,
    Padding,
    SGrid2DMetadata,
    SGrid3DMetadata,
    _attach_sgrid_metadata,
    dump_mappings,
    get_n_faces,
    get_n_nodes,
    load_mappings,
    xgcm_parse_sgrid,
)

__all__ = [
    "FaceNodePadding",
    "Padding",
    "SGrid2DMetadata",
    "SGrid3DMetadata",
    "SgridAccessor",
    "_attach_sgrid_metadata",
    "dump_mappings",
    "get_n_faces",
    "get_n_nodes",
    "load_mappings",
    "xgcm_parse_sgrid",
]

# To:
from .core import (
    FaceNodePadding,
    Padding,
    SGrid2DMetadata,
    SGrid3DMetadata,
    _attach_sgrid_metadata,
    dump_mappings,
    get_n_faces,
    get_n_nodes,
    load_mappings,
)

__all__ = [
    "FaceNodePadding",
    "Padding",
    "SGrid2DMetadata",
    "SGrid3DMetadata",
    "SgridAccessor",
    "_attach_sgrid_metadata",
    "dump_mappings",
    "get_n_faces",
    "get_n_nodes",
    "load_mappings",
]
```

- [ ] **Step 3: Fix `tests/sgrid/test_sgrid.py`**

Remove the `import xgcm` line (line 7).

Remove `SGRID_PADDING_TO_XGCM_POSITION` from the import on line 12:

```python
# Change from:
from parcels._sgrid.core import SGRID_PADDING_TO_XGCM_POSITION, _get_unique_names, parse_grid_attrs

# To:
from parcels._sgrid.core import _get_unique_names, parse_grid_attrs
```

Delete the entire `test_parse_sgrid_2d` function (lines 253-273) and `test_parse_sgrid_3d` function (lines 276-290) — these test the now-deleted `xgcm_parse_sgrid` bridge function. The SGRID parsing itself is still exercised by the existing `test_Grid2DMetadata_roundtrip`, `test_parse_grid_attrs`, and other tests.

- [ ] **Step 4: Fix `tests/datasets/test_structured.py`**

Replace the entire file content:

```python
import parcels._sgrid as sgrid
from parcels._datasets.structured.generic import datasets
from parcels._sgrid.accessor import get_dim_position


def test_left_indexed_dataset():
    """Checks that 'ds_2d_left' uses HIGH padding (MITgcm / left-indexed style)."""
    ds = datasets["ds_2d_left"]
    metadata = ds.sgrid.metadata
    for fnp in metadata.face_dimensions:
        assert get_dim_position(metadata, fnp.face) == "face"
        assert get_dim_position(metadata, fnp.node) == sgrid.Padding.HIGH


def test_right_indexed_dataset():
    """Checks that 'ds_2d_right' uses LOW padding (NEMO / right-indexed style)."""
    ds = datasets["ds_2d_right"]
    metadata = ds.sgrid.metadata
    for fnp in metadata.face_dimensions:
        assert get_dim_position(metadata, fnp.face) == "face"
        assert get_dim_position(metadata, fnp.node) == sgrid.Padding.LOW
```

- [ ] **Step 5: Remove xgcm from `pyproject.toml`**

Delete the line `"xgcm >=0.9.0",` from the `dependencies` list in `pyproject.toml`.

- [ ] **Step 6: Run the full test subset**

```bash
pytest tests/sgrid/ tests/datasets/ -x -q
```

Expected: all pass. The two deleted tests (`test_parse_sgrid_2d`, `test_parse_sgrid_3d`) are gone; four new tests from Task 1 and two rewritten tests from `test_structured.py` all pass.

- [ ] **Step 7: Verify xgcm is no longer imported anywhere in src/**

```bash
grep -r "import xgcm" src/parcels --include="*.py"
```

Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add src/parcels/_sgrid/core.py src/parcels/_sgrid/__init__.py \
        tests/sgrid/test_sgrid.py tests/datasets/test_structured.py \
        pyproject.toml
git commit -m "feat: remove xgcm dependency — use SGRID metadata natively throughout

Co-authored-by: Claude <noreply@anthropic.com>"
```
