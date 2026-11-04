"""Shared validation for discrete configuration values."""

from numbers import Integral


def require_integer(value, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_port(value) -> None:
    """Allow TCP ports, including zero for an OS-assigned temporary port."""
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= value <= 65535
    ):
        raise ValueError("port must be an integer between 0 and 65535")
