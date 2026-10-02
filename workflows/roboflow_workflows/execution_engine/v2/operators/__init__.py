"""Operators: explicit transitions between pulse domains of an active plan.

This package exposes the class-owned operator contract. The two built-in
operators live in their own modules and are collected explicitly, like
blocks, by ``v2.blocks.create_catalogue``::

    operators.alignment.Align     v2/align@v1
    operators.window.Window       v2/window@v1

They are not imported here: the catalogue imports the contract, and the
built-ins build run-time entries, whose package imports the catalogue.
"""

from roboflow_workflows.execution_engine.v2.operators.contract import (
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
