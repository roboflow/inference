import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("inference_models")

from inference.core.workflows.core_steps.formatters.vlm_as_detector.v1_tensor import (
    VLMAsDetectorBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections


def test_formatter_for_florence2_open_vocabulary_object_detection() -> None:
    # Locks the `florence_task_type == "<OPEN_VOCABULARY_DETECTION>"` exact-match
    # semantics ported from the numpy source (previously an accidental substring
    # test via `in`): the OVD branch must execute for the exact task type and map
    # class ids from the caller-provided `classes` list by index.
    # given
    block = VLMAsDetectorBlockV1()
    image = WorkflowImageData(
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="parent"),
    )
    vlm_output = """
{"bboxes": [[434.0, 30.848499298095703, 760.4000244140625, 530.4144897460938], [0.4000000059604645, 96.13949584960938, 528.4000244140625, 564.5574951171875]], "bboxes_labels": ["cat", "dog"]}
"""

    # when
    result = block.run(
        image=image,
        vlm_output=vlm_output,
        classes=["cat", "dog"],
        model_type="florence-2",
        task_type="open-vocabulary-object-detection",
    )

    # then
    assert result["error_status"] is False
    assert isinstance(result["predictions"], Detections)
    assert len(result["inference_id"]) > 0
    assert np.allclose(
        result["predictions"].xyxy.cpu().numpy(),
        np.array([[434, 30.848, 760.4, 530.41], [0.4, 96.139, 528.4, 564.56]]),
        atol=1e-1,
    ), "Expected coordinates to be the same as given in raw input"
    assert result["predictions"].class_id.cpu().tolist() == [0, 1]
    assert np.allclose(
        result["predictions"].confidence.cpu().numpy(), np.array([1.0, 1.0])
    )
    assert [entry["class"] for entry in result["predictions"].bboxes_metadata] == [
        "cat",
        "dog",
    ]
    assert result["predictions"].image_metadata["class_names"] == {0: "cat", 1: "dog"}
