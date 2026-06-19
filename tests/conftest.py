import pytest

from parcels import FieldSet
from parcels._datasets.structured.generic import datasets as datasets_structured

SKIP_BY_DEFAULT = {"validation", "flaky"}


def pytest_collection_modifyitems(config, items):
    if not config.getoption("-m"):
        for item in items:
            skip_by_default = list(SKIP_BY_DEFAULT & set(item.keywords))
            if skip_by_default:
                skip_marker = skip_by_default[0]  # get first marker in case of multiple
                item.add_marker(
                    pytest.mark.skip(reason=f"{skip_marker} tests skipped by default, use `-m {skip_marker}` to run")
                )


@pytest.fixture
def tmp_parquet(tmp_path):
    return tmp_path / "tmp.parquet"


@pytest.fixture
def fieldset() -> FieldSet:
    """FieldSet with U and V"""
    ds = datasets_structured["ds_2d_left"].copy()
    ds = ds[["U_A_grid", "V_A_grid", "grid"]].rename(
        {
            "U_A_grid": "U",
            "V_A_grid": "V",
        }
    )
    return FieldSet.from_sgrid_conventions(ds, mesh="flat")
