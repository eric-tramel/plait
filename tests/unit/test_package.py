"""Tests for package structure and imports."""

from pathlib import Path


def test_package_imports() -> None:
    """Verify the inf_engine package can be imported."""
    import inf_engine

    assert inf_engine is not None


def test_package_has_version() -> None:
    """Verify the package exposes a version string."""
    import inf_engine

    assert hasattr(inf_engine, "__version__")
    assert isinstance(inf_engine.__version__, str)
    assert len(inf_engine.__version__) > 0


def test_package_has_all() -> None:
    """Verify the package has __all__ defined for explicit exports."""
    import inf_engine

    assert hasattr(inf_engine, "__all__")
    assert isinstance(inf_engine.__all__, list)


def test_py_typed_marker_exists() -> None:
    """Verify the py.typed marker exists for PEP 561 compliance."""
    import inf_engine

    assert inf_engine.__file__ is not None, "Package __file__ should not be None"
    package_dir = Path(inf_engine.__file__).parent
    py_typed_path = package_dir / "py.typed"
    assert py_typed_path.exists(), "py.typed marker file should exist for PEP 561"
