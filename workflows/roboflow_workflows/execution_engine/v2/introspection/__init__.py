"""Structural introspection of V2 catalogues and compiled plans.

Four read-only questions, answered from class declarations and the compiled
plan alone; no block or source is constructed, no resource provider is called
and no submitted dynamic code runs::

    describe_catalogue(catalogue)  what blocks, sources, kinds, fields, defaults exist
    describe_workflow(plan)        inputs, declared sources, scoped steps with their
                                   causal domain, parameters, output groups, axes
    discover_connections(plan)     data, control, output, anchor and group edges
    discover_workload(plan)        resources, operations, restrictions (Discovery),
                                   source constructor resources
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
    SourceWorkload,
    StepWorkload,
    WorkloadReport,
    discover_workload,
)

__all__ = [
    "Connection",
    "ResourceUsage",
    "SourceWorkload",
    "StepWorkload",
    "WorkloadReport",
    "describe_catalogue",
    "describe_workflow",
    "discover_connections",
    "discover_workload",
]
