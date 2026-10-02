"""Real-V1 reference side of the sequential parity examples.

Importing this package never imports the V1 engine. Use ``REFERENCE_CASES``
for definitions, inputs and comparison labels, and
``collect_v1_observations`` to execute them on V1 in a child process.
"""

from .catalogue import (
    INPUT_PREPARATION_MUTATION,
    MATCH,
    REFERENCE_CASES,
    V1_ANOMALY,
    V1_DEFECT,
    V1_QUIRK,
    V2_BOUNDARY,
    Comparison,
    ReferenceCase,
    Run,
    case_ids,
    get_case,
)
from .v1_observations import V1ReferenceError, collect_v1_observations

__all__ = [
    "INPUT_PREPARATION_MUTATION",
    "MATCH",
    "REFERENCE_CASES",
    "V1_ANOMALY",
    "V1_DEFECT",
    "V1_QUIRK",
    "V2_BOUNDARY",
    "Comparison",
    "ReferenceCase",
    "Run",
    "V1ReferenceError",
    "case_ids",
    "collect_v1_observations",
    "get_case",
]
