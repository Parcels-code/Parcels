import pytest

zarr_filterwarning_consolidated_metadata = pytest.mark.filterwarnings(
    "ignore:Consolidated metadata is currently not part in the Zarr format 3 specification"
)
