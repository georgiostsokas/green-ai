"""Interval arithmetic for uncertain quantities.

Every quantity handled by Green AI is non-negative and every formula is
monotonically increasing in its inputs, so an interval can be propagated by
combining lower bounds with lower bounds and upper bounds with upper bounds.
This mirrors the convention used by EcoLogits and keeps results identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Union

Number = Union[int, float]


@dataclass(frozen=True)
class Range:
    """A closed interval ``[lo, hi]`` of a non-negative physical quantity."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo < 0:
            raise ValueError("Range bounds must be finite and non-negative")
        if self.lo > self.hi:
            raise ValueError(f"Range lower bound {self.lo} is above upper bound {self.hi}")

    # construction -----------------------------------------------------------------

    @classmethod
    def of(cls, value: Any) -> Range:
        """Coerce a number, a ``Range`` or a ``{"min": .., "max": ..}`` mapping."""
        if isinstance(value, Range):
            return value
        if isinstance(value, dict):
            return cls(float(value["min"]), float(value["max"]))
        return cls(float(value), float(value))

    # properties -------------------------------------------------------------------

    @property
    def mean(self) -> float:
        return (self.lo + self.hi) / 2

    @property
    def is_point(self) -> bool:
        return self.lo == self.hi

    # arithmetic -------------------------------------------------------------------

    def __add__(self, other: Value) -> Range:
        o = Range.of(other)
        return Range(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __mul__(self, other: Value) -> Range:
        o = Range.of(other)
        return Range(self.lo * o.lo, self.hi * o.hi)

    __rmul__ = __mul__

    def __truediv__(self, other: Number) -> Range:
        if isinstance(other, Range):
            raise TypeError("Division by a Range is not monotone and is not supported")
        return Range(self.lo / other, self.hi / other)

    def __format__(self, spec: str) -> str:
        if self.is_point:
            return format(self.lo, spec)
        return f"{format(self.mean, spec)} [{format(self.lo, spec)} to {format(self.hi, spec)}]"

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "min": self.lo, "max": self.hi}


Value = Union[Number, Range]


def lo(v: Value) -> float:
    return v.lo if isinstance(v, Range) else float(v)


def hi(v: Value) -> float:
    return v.hi if isinstance(v, Range) else float(v)


def mean(v: Value) -> float:
    return v.mean if isinstance(v, Range) else float(v)


def add(a: Value, b: Value) -> Value:
    if isinstance(a, Range) or isinstance(b, Range):
        return Range.of(a) + Range.of(b)
    return a + b


def mul(a: Value, b: Value) -> Value:
    if isinstance(a, Range) or isinstance(b, Range):
        return Range.of(a) * Range.of(b)
    return a * b


def parse(value: Any) -> Value:
    """Parse JSON style values: numbers stay numbers, ``{"min","max"}`` becomes a Range."""
    if isinstance(value, dict):
        r = Range.of(value)
        return r.lo if r.is_point else r
    if isinstance(value, Range):
        return value
    return float(value)
