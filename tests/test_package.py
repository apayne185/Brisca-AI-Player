import brisca


def test_version_is_exposed() -> None:
    assert isinstance(brisca.__version__, str)
    assert brisca.__version__
