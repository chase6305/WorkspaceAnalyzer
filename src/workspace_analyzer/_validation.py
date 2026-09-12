"""Shared validation for discrete configuration values."""

from numbers import Integral


def require_integer(value, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
