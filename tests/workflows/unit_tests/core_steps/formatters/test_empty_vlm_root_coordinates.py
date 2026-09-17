import numpy as np
import pytest

from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.anthropic_detection_parsing import (
    parse_anthropic_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.gemini_detection_parsing import (
    parse_gemini_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.muse_detection_parsing import (
    parse_muse_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.openai_detection_parsing import (
    parse_openai_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.qwen_detection_parsing import (
    parse_qwen_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.spacexai_detection_parsing import (
    parse_spacexai_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2 import (
    parse_llm_object_detection_response,
)
from inference.core.workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.executor.output_constructor import (
    _prepare_data_piece_for_output,
)


@pytest.mark.parametrize(
    "parser,response",
    [
        (parse_anthropic_object_detection_response, []),
        (parse_gemini_object_detection_response, []),
        (parse_muse_object_detection_response, []),
        (parse_openai_object_detection_response, []),
        (parse_qwen_object_detection_response, []),
        (parse_spacexai_object_detection_response, []),
        (parse_llm_object_detection_response, {"detections": []}),
    ],
)
@pytest.mark.parametrize(
    "size,offset",
    [((60, 80), (50, 100)), ((60, 80), (0, 0)), ((480, 640), (0, 0))],
)
def test_empty_vlm_output_honors_coordinate_system(parser, response, size, offset):
    height, width = size
    image = WorkflowImageData(
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
        parent_metadata=ImageParentMetadata(parent_id="crop"),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id="root",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=offset[0],
                left_top_y=offset[1],
                origin_width=640,
                origin_height=480,
            ),
        ),
    )
    detections = parser(
        image=image, parsed_data=response, classes=["cat"], inference_id="test"
    )
    own = _prepare_data_piece_for_output(
        data_piece=detections,
        resolve_output_futures=False,
        convert_to_parent_coordinates=False,
    )
    parent = _prepare_data_piece_for_output(
        data_piece=detections,
        resolve_output_futures=False,
        convert_to_parent_coordinates=True,
    )

    assert serialise_sv_detections(parent) == {
        "image": {"width": 640, "height": 480},
        "predictions": [],
    }
    assert serialise_sv_detections(own) == {
        "image": {"width": width, "height": height},
        "predictions": [],
    }
    assert detections.metadata[IMAGE_DIMENSIONS_KEY] == [height, width]
    assert detections.metadata[ROOT_PARENT_COORDINATES_KEY] == list(offset)
    assert parent.metadata[ROOT_PARENT_COORDINATES_KEY] == [0, 0]
    assert parent.metadata[ROOT_PARENT_DIMENSIONS_KEY] == [480, 640]
    assert parent.metadata[ROOT_PARENT_ID_KEY] == "root"
    assert parent.confidence.tolist() == []
    assert parent.class_id.tolist() == []
