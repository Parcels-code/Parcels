import pytest

from parcels._core.utils.unstructured import (
    get_vertical_dim_name_from_location,
    get_vertical_location_from_dims,
)


def test_get_vertical_location_from_dims():
    # Test with zc dimension
    assert get_vertical_location_from_dims(("zc", "time")) == "center"

    # Test with zf dimension
    assert get_vertical_location_from_dims(("zf", "time")) == "face"

    # Test with both dimensions
    with pytest.raises(ValueError):
        get_vertical_location_from_dims(("zc", "zf", "time"))

    # Test with no vertical dimension
    with pytest.raises(ValueError):
        get_vertical_location_from_dims(("time", "x", "y"))


def test_get_vertical_dim_name_from_location():
    # Test with center location
    assert get_vertical_dim_name_from_location("center") == "zc"

    # Test with face location
    assert get_vertical_dim_name_from_location("face") == "zf"

    with pytest.raises(KeyError):
        get_vertical_dim_name_from_location("invalid_location")
