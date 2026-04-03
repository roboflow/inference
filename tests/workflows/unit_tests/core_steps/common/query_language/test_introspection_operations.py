"""Tests for UQL operations introspection (prepare_operations_descriptions)."""

from inference.core.workflows.core_steps.common.query_language.introspection.core import (
    prepare_operations_descriptions,
)


def test_detections_property_extract_includes_property_name_options_with_area_fields():
    """Describe endpoint UQL operations must include area_px and area_converted for DetectionsPropertyExtract."""
    operations = prepare_operations_descriptions()
    detections_extract = next(
        (op for op in operations if op.operation_type == "DetectionsPropertyExtract"),
        None,
    )
    assert detections_extract is not None
    assert detections_extract.property_name_options is not None
    assert "area_px" in detections_extract.property_name_options
    assert "area_converted" in detections_extract.property_name_options
