import numpy as np

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
)
from inference.core.workflows.core_steps.common.query_language.operations.detection.base import (
    extract_detection_property,
)
from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
)


def test_extract_detection_property_with_area_px() -> None:
    # given
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.float32(0.9),
        np.int64(1),
        None,
        {
            "class_name": np.array("leaf"),
            AREA_KEY_IN_SV_DETECTIONS: np.float32(400.0),
            AREA_CONVERTED_KEY_IN_SV_DETECTIONS: np.float32(4.0),
        },
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.AREA,
        execution_context="<test>",
    )

    # then
    assert result == 400.0


def test_extract_detection_property_with_area_converted() -> None:
    # given
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.float32(0.9),
        np.int64(1),
        None,
        {
            "class_name": np.array("leaf"),
            AREA_KEY_IN_SV_DETECTIONS: np.float32(400.0),
            AREA_CONVERTED_KEY_IN_SV_DETECTIONS: np.float32(4.0),
        },
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.AREA_CONVERTED,
        execution_context="<test>",
    )

    # then
    assert result == 4.0
