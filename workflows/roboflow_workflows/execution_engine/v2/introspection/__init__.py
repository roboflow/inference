"""Structural introspection of V2 catalogues and compiled plans.

Four read-only questions, answered from class declarations and the compiled
plan alone; no block is constructed, no resource provider is called and no
submitted dynamic code runs::

    describe_catalogue(catalogue)  what blocks, kinds, fields, defaults exist
    describe_workflow(plan)        inputs, scoped steps, parameters, sources, axes
    discover_connections(plan)     data, control and output edges
    discover_workload(plan)        resources, operations, restrictions (Discovery)
"""

from roboflow_workflows.execution_engine.v2.introspection.catalogue import (
    describe_catalogue,
)
from roboflow_workflows.execution_engine.v2.introspection.workflow import (
    Connection,
    describe_workflow,
    discover_connections,
)
from roboflow_workflows.execution_engine.v2.introspection.workload import (
    ResourceUsage,
    StepWorkload,
    WorkloadReport,
    discover_workload,
)

__all__ = [
    "Connection",
    "ResourceUsage",
    "StepWorkload",
    "WorkloadReport",
    "describe_catalogue",
    "describe_workflow",
    "discover_connections",
    "discover_workload",
]
