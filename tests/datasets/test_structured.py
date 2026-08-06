from parcels._datasets.structured.generic import datasets
from parcels._sgrid.accessor import get_dim_position
from parcels._sgrid.core import Padding


def test_left_indexed_dataset():
    """Checks that 'ds_2d_left' has HIGH padding (MITgcm/left-indexed) on all spatial axes."""
    ds = datasets["ds_2d_left"]
    metadata = ds.sgrid.metadata
    for fnp in metadata.face_dimensions:
        assert get_dim_position(metadata, fnp.face) == "face"
        assert get_dim_position(metadata, fnp.node) == Padding.HIGH


def test_right_indexed_dataset():
    """Checks that 'ds_2d_right' has LOW padding (NEMO/right-indexed) on all spatial axes."""
    ds = datasets["ds_2d_right"]
    metadata = ds.sgrid.metadata
    for fnp in metadata.face_dimensions:
        assert get_dim_position(metadata, fnp.face) == "face"
        assert get_dim_position(metadata, fnp.node) == Padding.LOW
