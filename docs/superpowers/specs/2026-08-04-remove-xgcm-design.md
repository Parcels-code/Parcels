# Remove xgcm Dependency — Design Spec

**Date:** 2026-08-04
**Branch:** remove-xgcm
**Approach:** Option B — full SGRID-native refactor (remove XgcmLike* adapter layer)

## Motivation

All grid topology information previously sourced from xgcm (axis directions, staggering positions, face/node relationships) is now fully expressed in SGRID metadata attached to the dataset. The XgcmLike* adapter layer was always a temporary bridge. Removing xgcm simplifies the dependency graph, removes COMODO metadata coupling, and lets the codebase work entirely with the standardised SGRID model.

---

## Section 1: New Position Vocabulary

Replace xgcm position strings (`"center"`, `"left"`, `"right"`, `"inner"`, `"outer"`) with a new `GridPosition` type in `_typing.py`:

```python
from parcels._sgrid.core import Padding

GridPosition = Literal["face"] | Padding
```

| Old (xgcm) | New (SGRID)    |
| ---------- | -------------- |
| `"center"` | `"face"`       |
| `"right"`  | `Padding.LOW`  |
| `"left"`   | `Padding.HIGH` |
| `"inner"`  | `Padding.BOTH` |
| `"outer"`  | `Padding.NONE` |

**Removed from `_typing.py`:**

- `XgcmAxisPosition`
- `XgcmAxes`
- `import xgcm` (TYPE_CHECKING block)

**Removed from `_sgrid/core.py`:**

- `SGRID_PADDING_TO_XGCM_POSITION` dict
- `xgcm_parse_sgrid()` function

**Added to `_sgrid/accessor.py`:**

```python
def get_dim_position(grid: SGrid2DMetadata, dim: str) -> GridPosition:
    """Returns 'face' or the Padding value for a given dimension."""
```

This replaces `get_xgcm_position_from_dim_name`. Uses the existing `_get_axis_info` helper internally.

---

## Section 2: `xgrid.py` Changes

### Removed entirely

- `XgcmLikeAxis` dataclass
- `XgcmLikeGrid` class
- `construct_xgcm_axes_object` function
- `self.xgcm_grid` attribute on `XGrid`
- `_DEFAULT_XGCM_KWARGS`
- `import xgcm` and `import xgcm.axis` (TYPE_CHECKING blocks)
- Import of `SGRID_PADDING_TO_XGCM_POSITION`

`self.sgrid_metadata` (already stored on `XGrid`) becomes the sole source of grid topology truth.

### Changed function signatures

| Function                                     | Old signature                                                                 | New signature                                           |
| -------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| `get_cell_count_along_dim`                   | `(ds, axis: xgcm.axis.Axis)`                                                  | `(ds, fnp: FaceNodePadding)` — uses `ds[fnp.face].size` |
| `get_time`                                   | `(ds, axis: xgcm.axis.Axis)`                                                  | `(ds, time_dim: str)` — uses `ds[time_dim].values`      |
| `_get_xgrid_axes`                            | `(grid: xgcm.Grid)`                                                           | `(metadata: SGrid2DMetadata)`                           |
| `assert_all_field_dims_have_axis`            | `(da, xgcm_grid: xgcm.Grid)`                                                  | `(da, metadata: SGrid2DMetadata)`                       |
| `assert_valid_lat_lon`                       | `(da_lat, da_lon, axes: XgcmAxes)`                                            | `(da_lat, da_lon, metadata: SGrid2DMetadata)`           |
| `assert_all_dimensions_correspond_with_axis` | `(da, axes: XgcmAxes)`                                                        | `(da, metadata: SGrid2DMetadata)`                       |
| `assert_valid_field_array`                   | `(da, axes: XgcmAxes)`                                                        | `(da, metadata: SGrid2DMetadata)`                       |
| `_convert_center_pos_to_fpoint`              | `(xgcm_position: XgcmAxisPosition, f_points_xgcm_position: XgcmAxisPosition)` | `(position: GridPosition, f_point_position: Padding)`   |

### Internal logic changes

- `get_axis_from_dim_name(axes, dim)` → replaced by `_get_dim_to_axis_mapping(metadata).get(dim)` from accessor
- `get_xgcm_position_from_dim_name(axes, dim)` → replaced by `get_dim_position(metadata, dim)` from accessor
- `_fpoint_info` → returns `dict[XgridAxis, Padding]` (was `dict[XgridAxis, str]` with xgcm strings)
- `localize()` → uses `self.sgrid_metadata` instead of `self.xgcm_grid`
- `get_axis_dim_mapping()` → uses `_get_dim_to_axis_mapping(self.sgrid_metadata)` directly
- `_convert_center_pos_to_fpoint`: `"center"` branch becomes `"face"` check; `"inner"/"right"` checks become `Padding.BOTH / Padding.LOW`
- `XGrid.lon/lat/depth/_datetimes` — currently check `self.xgcm_grid.axes["X"/"Y"/"Z"/"T"]` to determine axis presence; replaced by checking `_get_dim_to_axis_mapping(self.sgrid_metadata)` for spatial axes, and `"time" in self._ds.dims` for the time axis

---

## Section 3: Test Changes

### `tests/datasets/test_structured.py`

Replace `xgcm.Grid` calls with SGRID-native checks:

```python
# Old
grid = xgcm.Grid(ds, **_DEFAULT_XGCM_KWARGS)
for _axis_name, axis in grid.axes.items():
    for pos, _dim_name in axis.coords.items():
        assert pos in ["left", "center"]

# New
metadata = ds.sgrid.metadata
for fnp in metadata.face_dimensions:
    assert get_dim_position(metadata, fnp.face) == "face"
    assert get_dim_position(metadata, fnp.node) in (Padding.HIGH, Padding.LOW, Padding.BOTH, Padding.NONE)
```

The specific padding assertion per test depends on the dataset fixture (`ds_2d_left` vs `ds_2d_right`).

### `tests/sgrid/test_sgrid.py`

- Delete the two tests at lines 259–285 that call `xgcm_parse_sgrid()` and `xgcm.Grid(...)` — they were testing the now-removed bridge function
- Remove `import xgcm` and `SGRID_PADDING_TO_XGCM_POSITION` from imports

### `pyproject.toml`

Remove `"xgcm >=0.9.0"` from the dependencies list.

---

## Out of Scope

- Changing interpolation logic or search algorithms in `xgrid.py`
- Renaming `XgcmAxisDirection` / `CfAxis` type aliases (they don't reference xgcm at runtime)
- Any changes to SGRID parsing or accessor logic beyond adding `get_dim_position`
