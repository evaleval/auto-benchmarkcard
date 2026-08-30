"""Composer public facade without eagerly importing the full composition tool."""

from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from .composer_tool import compose_benchmark_card

__all__ = ["compose_benchmark_card"]

globals().pop("compose_benchmark_card", None)


def __getattr__(name: str):
    if name != "compose_benchmark_card":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .composer_tool import compose_benchmark_card

    globals()[name] = compose_benchmark_card
    return compose_benchmark_card


def __dir__():
    return sorted(set(globals()) | set(__all__))
