import pytest


@pytest.fixture
def tmp_parquet(tmp_path):
    return tmp_path / "tmp.parquet"
