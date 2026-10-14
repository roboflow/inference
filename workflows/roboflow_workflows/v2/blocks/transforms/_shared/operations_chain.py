"""Chain compiler used by the detections filter and transformation blocks.

The ``roboflow_core/detections_filter@v1`` and
``roboflow_core/detections_transformation@v1`` block definitions (ports of
the V1 blocks) call ``compile_chain``. As in V1, both run the same chain and
differ only in name and description::

    predictions ──▶ step 1 ──▶ step 2 ──▶ ... ──▶ predictions (same carrier)

Allowed top-level steps keep a detections carrier: ``DetectionsFilter``,
``DetectionsSelection``, ``SortDetections``, ``DetectionsOffset``,
``DetectionsShift``, ``DetectionsRename`` and ``PickDetectionsByParentClass``.
Compilation rejects other steps, an invalid chain, an operand name missing
from ``operations_parameters`` and the reserved name ``_``.

Rows keep their masks, keypoints, row metadata and row record. Host work is
the query helper's: a filter condition runs in Python per row and copies each
box column it reads to the host once per call.

Each block declares its own parameters, outputs and run; this module holds
the chain validation (``compile_chain``) and the schema example both use.
"""

from typing import Any, List, Tuple
from roboflow_workflows.v2.blocks.query import (
    DETECTIONS_PRESERVING_OPERATIONS,
    CompiledOperations,
    compile_operations,
)
from roboflow_workflows.v2.blocks.query.schema import DEFAULT_OPERAND_NAME

OPERATIONS_EXAMPLE = [
    {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "class_name",
                            }
                        ],
                    },
                    "comparator": {"type": "in (Sequence)"},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "classes",
                    },
                }
            ],
        },
    }
]


def compile_chain(
    operations: List[Any], *, parameter_names: Tuple[str, ...]
) -> CompiledOperations:
    """Validate and compile a detections-preserving operation chain.

    Args:
        operations: UQL steps, as JSON objects or validated models.
        parameter_names: Names declared in ``operations_parameters``.

    Returns:
        The compiled chain.

    Raises:
        QueryDefinitionError: On an invalid chain or an undeclared operand.
        ValueError: When ``parameter_names`` contains ``_`` or a step does
            not return detections.
    """
    if DEFAULT_OPERAND_NAME in parameter_names:
        raise ValueError(
            f"operations_parameters cannot use the name '{DEFAULT_OPERAND_NAME}': "
            "it is reserved for the detection a DetectionsFilter evaluates"
        )

    chain = compile_operations(
        operations, path="$.operations", declared_names=set(parameter_names)
    )
    for index, operation in enumerate(operations):
        operation_type = (
            operation["type"] if isinstance(operation, dict) else operation.type
        )
        if operation_type not in DETECTIONS_PRESERVING_OPERATIONS:
            raise ValueError(
                f"$.operations[{index}]: {operation_type} does not return "
                "detections; use only "
                f"{', '.join(sorted(DETECTIONS_PRESERVING_OPERATIONS))}"
            )

    return chain
