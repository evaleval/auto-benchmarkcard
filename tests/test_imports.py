"""Smoke test: the package and its public workflow entry point import cleanly."""


def test_package_imports():
    import auto_benchmarkcard  # noqa: F401


def test_build_workflow_importable():
    from auto_benchmarkcard.workflow import build_workflow

    assert callable(build_workflow)
