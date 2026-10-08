from parcels._repr_utils import _format_list_items_multiline


def test_format_list_items_multiline():
    expected = """[
    item1,
    item2,
    item3
]"""
    assert _format_list_items_multiline(["item1", "item2", "item3"], 1) == expected
    assert _format_list_items_multiline([], 1) == "[]"
