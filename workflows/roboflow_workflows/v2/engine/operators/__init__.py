"""Operators: explicit transitions between pulse domains of an active plan.

This package exposes the class-owned operator contract. The two built-in
operators live in their own modules and are collected explicitly, like
blocks, by ``v2.blocks.create_catalogue``::

    operators.alignment.Align     roboflow_core/align@v1
    operators.window.Window       roboflow_core/window@v1

They are not imported here: the catalogue imports the contract, and the
built-ins build run-time entries, whose package imports the catalogue.
"""

from roboflow_workflows.v2.engine.operators.contract import (
    INPUT_MAP_ROLES,
    Arrival,
    InputRole,
    Operator,
    OperatorCounters,
    OperatorDeclarationError,
    OperatorInput,
    OperatorParams,
    OperatorPort,
    OperatorPulse,
    OperatorSpec,
    TerminationReason,
    operator_step_path,
    spec_of_operator,
)

__all__ = [
    "INPUT_MAP_ROLES",
    "Arrival",
    "InputRole",
    "Operator",
    "OperatorCounters",
    "OperatorDeclarationError",
    "OperatorInput",
    "OperatorParams",
    "OperatorPort",
    "OperatorPulse",
    "OperatorSpec",
    "TerminationReason",
    "operator_step_path",
    "spec_of_operator",
]
