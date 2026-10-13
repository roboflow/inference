"""Deprecated import path for ``roboflow_workflows.v2.data.kinds``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    BUILTIN_KINDS,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    WILDCARD_KIND_NAME,
    Kind,
    OutputConverter,
    PayloadCodec,
    PayloadValidator,
    kinds_compatible,
    normalize_kinds,
)

__all__ = [
    "BOOLEAN_KIND",
    "BUILTIN_KINDS",
    "DICTIONARY_KIND",
    "FLOAT_KIND",
    "INTEGER_KIND",
    "Kind",
    "LIST_OF_VALUES_KIND",
    "OutputConverter",
    "PayloadCodec",
    "PayloadValidator",
    "STRING_KIND",
    "WILDCARD_KIND",
    "WILDCARD_KIND_NAME",
    "kinds_compatible",
    "normalize_kinds",
]
