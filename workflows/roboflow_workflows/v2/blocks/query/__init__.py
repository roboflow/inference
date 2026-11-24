"""Internal V2 helper for existing UQL (Roboflow query language) JSON.

Validate a literal query once, then evaluate it per runtime values::

    condition = compile_condition(definition, declared_names={"_", "threshold"})
    passed = condition.evaluate({"_": 0.7, "threshold": 0.5})

    operations = compile_operations(
        [{"type": "DetectionsFilter", "filter_operation": {...}}],
        declared_names={"classes"},
    )
    kept = operations.apply(detections, parameters={"classes": ["car"]})

Every current UQL literal is supported: 36 operations, 16 comparator
spellings, 6 unary operators and 5 structural types. Detection operations
accept native ``Detections``, ``InstanceDetections``, ``(KeyPoints,
Detections)`` tuples, ``PredictionRows`` and ``SegmentationView``; image
operations accept ``ImageData``. ``parameters`` reach nested filters and
string ``class_map`` / ``strict`` names of ``DetectionsRename``.
"""

from roboflow_workflows.v2.blocks.query.conditions import (
    CompiledCondition,
    compile_condition,
)
from roboflow_workflows.v2.blocks.query.detections import DetectionRow
from roboflow_workflows.v2.blocks.query.errors import (
    QueryDefinitionError,
    QueryError,
    QueryRuntimeError,
)
from roboflow_workflows.v2.blocks.query.operations import (
    DETECTIONS_PRESERVING_OPERATIONS,
    CompiledOperations,
    compile_operations,
)
from roboflow_workflows.v2.blocks.query.schema import (
    COMPARATOR_TYPES,
    DETECTION_PROPERTIES,
    SUPPORTED_OPERATION_TYPES,
    UNARY_OPERATOR_TYPES,
    BinaryStatement,
    DynamicOperand,
    Operation,
    StatementGroup,
    StaticOperand,
    UnaryStatement,
)

__all__ = [
    "COMPARATOR_TYPES",
    "DETECTIONS_PRESERVING_OPERATIONS",
    "DETECTION_PROPERTIES",
    "SUPPORTED_OPERATION_TYPES",
    "UNARY_OPERATOR_TYPES",
    "BinaryStatement",
    "CompiledCondition",
    "CompiledOperations",
    "DetectionRow",
    "DynamicOperand",
    "Operation",
    "QueryDefinitionError",
    "QueryError",
    "QueryRuntimeError",
    "StatementGroup",
    "StaticOperand",
    "UnaryStatement",
    "compile_condition",
    "compile_operations",
]
