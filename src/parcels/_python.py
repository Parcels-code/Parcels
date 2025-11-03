# Generic Python helpers
import inspect
from collections.abc import Callable


def isinstance_noimport(obj, class_or_tuple):
    """A version of isinstance that does not require importing the class.
    This is useful to avoid circular imports.
    """
    return (
        type(obj).__name__ == class_or_tuple
        if isinstance(class_or_tuple, str)
        else type(obj).__name__ in class_or_tuple
    )


def assert_same_function_signature(f: Callable, *, ref: Callable, context: str) -> None:
    """Ensures a function `f` has the same signature as the reference function `ref`."""
    sig_ref = inspect.signature(ref)
    sig = inspect.signature(f)

    if len(sig_ref.parameters) != len(sig.parameters):
        raise ValueError(
            f"{context} function must have {len(sig_ref.parameters)} parameters, got {len(sig.parameters)}"
        )

    for param1, param2 in zip(sig_ref.parameters.values(), sig.parameters.values(), strict=False):
        if param1.kind != param2.kind:
            raise ValueError(
                f"Parameter '{param2.name}' has incorrect parameter kind. Expected {param1.kind}, got {param2.kind}"
            )
        if param1.name != param2.name:
            raise ValueError(
                f"Parameter '{param2.name}' has incorrect name. Expected '{param1.name}', got '{param2.name}'"
            )


def test_isinstance_noimport():
    class A:
        pass

    class B:
        pass

    a = A()
    b = B()

    assert isinstance_noimport(a, "A")
    assert not isinstance_noimport(a, "B")
    assert isinstance_noimport(b, ("A", "B"))
    assert not isinstance_noimport(b, "C")
