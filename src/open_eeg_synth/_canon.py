"""Plain Python values for the frozen spec dataclasses (DESIGN §7.2).

A spec's identity is its canonical JSON (``CaseSpec.digest``), so two specs that compare equal must
serialise identically: ``duration_s=8`` and ``duration_s=8.0`` must not give two case ids, and a
numpy scalar must not make ``json.dumps`` fail. Each spec normalises its own fields once, on
construction, from their annotations.
"""

from __future__ import annotations

import json
import math
import numbers
import operator
from dataclasses import fields
from typing import Any


def _scalar(base: str, obj: Any, name: str, v: Any) -> Any:
    """``v`` coerced to the plain Python type ``base`` names, or a ``TypeError`` naming the
    field: a wrong type used to pass silently here (``bool("false")`` is truthy, ``float("256")``
    parses a numeric string, ``operator.index(True)`` accepts a bool as an int) instead of being
    rejected. ``numbers.Integral``/``numbers.Real`` (not the bare ``int``/``float`` builtins) so
    numpy scalars keep working exactly as before - they are registered with both ABCs - while an
    actual ``bool`` (itself an ``int`` and a ``numbers.Integral``) is always refused except for a
    genuinely ``bool``-typed field.
    """
    is_bool = isinstance(v, bool)
    if base == "bool":
        if not is_bool:
            raise TypeError(f"{type(obj).__name__}.{name} must be bool, got {v!r}")
        return v
    if base == "int":
        if is_bool or not isinstance(v, numbers.Integral):
            raise TypeError(f"{type(obj).__name__}.{name} must be an int, got {v!r}")
        return operator.index(v)
    if base == "float":
        if is_bool or not isinstance(v, numbers.Real):
            raise TypeError(f"{type(obj).__name__}.{name} must be a number, got {v!r}")
        return float(v)
    if not isinstance(v, str):
        raise TypeError(f"{type(obj).__name__}.{name} must be a str, got {v!r}")
    return v


def canonical_fields(obj: Any) -> None:
    """Coerce every scalar-annotated field of a (frozen) dataclass to its plain Python type.

    Handles ``float``, ``int`` (no silent truncation: ``operator.index``), ``bool``, ``str``,
    their ``| None`` forms, ``dict[str, float]``, ``tuple[float, float]`` and ``tuple[str, ...]``;
    other fields are left to the class itself.
    """
    for f in fields(obj):
        t = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        v = getattr(obj, f.name)
        optional = t.endswith(" | None")
        base = t[: -len(" | None")] if optional else t
        if optional and v is None:
            continue
        if base in ("bool", "int", "float", "str"):
            new = _scalar(base, obj, f.name, v)
        elif base == "dict[str, float]":
            new = {str(k): float(x) for k, x in v.items()}
        elif base == "tuple[float, float]":
            new = tuple(float(x) for x in v)
            if len(new) != 2:
                raise ValueError(f"{type(obj).__name__}.{f.name} needs two values, got {v!r}")
        elif base == "tuple[str, ...]":
            new = tuple(str(x) for x in v)
        else:
            continue
        object.__setattr__(obj, f.name, new)


def json_plain(value: Any) -> Any:
    """``value`` as JSON reads it back: numpy scalars and arrays become Python, tuples lists."""

    def default(o: Any) -> Any:
        if hasattr(o, "tolist"):  # numpy scalar or array
            return o.tolist()
        raise TypeError(f"{type(o).__name__} is not JSON-serialisable: {o!r}")

    return json.loads(json.dumps(value, default=default, allow_nan=False))


def inf_to_json(x: float) -> float | None:
    """Strict JSON has no infinity: an unbounded time is written as null."""
    return None if math.isinf(x) else x


def inf_from_json(x: float | None) -> float:
    return math.inf if x is None else float(x)
