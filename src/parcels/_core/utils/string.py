from keyword import iskeyword, kwlist


def _assert_str_and_python_varname(name):
    if not isinstance(name, str):
        raise TypeError(f"Expected a string for variable name, got {type(name).__name__} instead.")

    msg = f"Received invalid Python variable name {name!r}: "

    if not name.isidentifier():
        msg += "not a valid identifier. HINT: avoid using spaces, special characters, and starting with a number."
        raise ValueError(msg)

    if iskeyword(name):
        msg += f"it is a reserved keyword. HINT: avoid using the following names: {', '.join(kwlist)}"
        raise ValueError(msg)
