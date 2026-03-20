"""Regression tests for workflow schema containing DetectionsProperty enum.

Ensures the schema returned by get_workflow_schema_description() includes
area_px and area_converted in the DetectionsProperty enum (used by
DetectionsPropertyExtract and other UQL operations).
"""

import pytest

from inference.core.workflows.execution_engine.v1.compiler import syntactic_parser


@pytest.fixture(autouse=True)
def clear_schema_cache():
    """Clear schema cache so schema is rebuilt from current block/enum definitions."""
    syntactic_parser.clear_cache()
    yield
    syntactic_parser.clear_cache()


def test_workflow_schema_includes_area_px_and_area_converted_in_detections_property():
    """DetectionsProperty enum in schema must include area_px and area_converted."""
    from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
        get_workflow_schema_description,
    )

    desc = get_workflow_schema_description()
    schema = desc.schema
    defs = schema.get("$defs", schema.get("definitions", {}))

    detections_property_schema = defs.get("DetectionsProperty")
    assert detections_property_schema is not None, (
        "Schema should define DetectionsProperty (e.g. under $defs)"
    )
    enum_values = detections_property_schema.get("enum", [])
    assert "area_px" in enum_values, (
        "DetectionsProperty enum in workflow schema must include 'area_px' "
        "(from DetectionsProperty.AREA)"
    )
    assert "area_converted" in enum_values, (
        "DetectionsProperty enum in workflow schema must include 'area_converted' "
        "(from DetectionsProperty.AREA_CONVERTED)"
    )
