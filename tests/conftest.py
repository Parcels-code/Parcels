import pytest

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
