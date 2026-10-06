"""Block -> adapter -> request differential tests for Task 11.14's SAM family
(SAM2, SAM3 interactive, SAM3 v1/v2/v3, SAM3-3D v1/v1_tensor).

Each test drives a block's real ``run_locally`` through a real
``ModelManagerModelsProvider`` wrapping a ``MagicMock`` ``ModelManager``, then
compares the pydantic request the adapter built against the exact request
construction the block used to run inline before Task 11.14 (this task's
actual BASE, ``8cde35edf``, is the immediate parent of Task 11.14's commit on
this branch - the pre-port construction below is copied straight from
``git show 8cde35edf:<path>``, i.e. the source this task started from; that
tree is byte-identical to ``1c535758e``'s for every block file this suite
reads, since the one commit between them touched only
``test_sam_prompts.py``).

One case per distinct request-building shape the task touches:
``Sam2SegmentationRequest`` built from a box prompt (``segment_anything2/v1.py``)
and from a point prompt with the ``model_id=`` selector shape
(``segment_anything3_interactive/v1.py``); ``Sam3SegmentationRequest`` in its
v1 (bare), v2 (``nms_iou_threshold``) and v3 (``format``) shapes; and
``Sam3_3D_Objects_InferenceRequest`` for both the numpy block
(``segment_anything3_3d/v1.py``) and its tensor-native sibling
(``segment_anything3_3d/v1_tensor.py``, which materialises masks from
``InstanceDetections`` instead of ``sv.Detections`` before the same request
shape).

Every case asserts both ``model_dump()`` equality AND ``model_fields_set``
equality: an omitted field must stay omitted, not be re-supplied as its own
default.
"""

from unittest.mock import MagicMock

import numpy as np
import supervision as sv
import torch

from inference.core.entities.requests.sam2 import (
    Box,
    Point,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.requests.sam3 import Sam3Prompt, Sam3SegmentationRequest
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.tensor_native import (
    instance_mask_to_numpy,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    SegmentAnything2BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v1 import (
    SegmentAnything3BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2 import (
    SegmentAnything3BlockV2,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    SegmentAnything3BlockV3,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
    SegmentAnything3_3D_ObjectsBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1_tensor import (
    SegmentAnything3_3D_ObjectsBlockV1 as SegmentAnything3_3D_ObjectsTensorBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1 import (
    SAM3_INTERACTIVE_MODEL_ID,
    SegmentAnything3InteractiveBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.prototypes.models_provider import CORE_MODEL_ENDPOINT_TYPE
from inference_models.models.base.instance_segmentation import InstanceDetections


def _make_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="p"),
        numpy_image=np.zeros((20, 10, 3), dtype=np.uint8),
    )


def _batch_of_one(item) -> Batch:
    return Batch(content=[item], indices=[(0,)])


def _manager() -> MagicMock:
    return MagicMock()


def _captured_request(manager: MagicMock):
    assert manager.infer_from_request_sync.call_count == 1
    call = manager.infer_from_request_sync.call_args
    return call.kwargs["request"] if "request" in call.kwargs else call.args[1]


def _box_detections() -> sv.Detections:
    detections = sv.Detections(
        xyxy=np.array([[10.0, 10.0, 50.0, 50.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["object"]),
            "detection_id": np.array(["d1"]),
        },
    )
    return detections


def test_sam2_v1_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 8cde35edf:.../segment_anything2/v1.py`:
    # `Sam2SegmentationRequest(image=..., sam2_version_id=version, api_key=...,
    # source="workflow-execution", prompts=Sam2PromptSet(prompts=prompts),
    # threshold=threshold, multimask_output=multimask_output)`, with `prompts`
    # built from the box-centre arithmetic (`cx`/`cy`/`width`/`height` of
    # [10, 10, 50, 50] -> centre (30, 30), width/height 40).
    manager = _manager()
    manager.infer_from_request_sync.return_value = MagicMock(predictions=[])
    image = _make_image()
    block = SegmentAnything2BlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=_batch_of_one(image),
        boxes=_batch_of_one(_box_detections()),
        version="hiera_large",
        threshold=0.0,
        multimask_output=True,
    )

    manager.add_model.assert_called_once_with(
        model_id="sam2/hiera_large", api_key="k", endpoint_type=CORE_MODEL_ENDPOINT_TYPE
    )
    request = _captured_request(manager)
    expected = Sam2SegmentationRequest(
        image=image.to_inference_format(numpy_preferred=True),
        sam2_version_id="hiera_large",
        api_key="k",
        source="workflow-execution",
        prompts=Sam2PromptSet(
            prompts=[Sam2Prompt(box=Box(x=30.0, y=30.0, width=40.0, height=40.0))]
        ),
        threshold=0.0,
        multimask_output=True,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_sam3_interactive_v1_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 8cde35edf:.../segment_anything3_interactive/v1.py`:
    # `Sam2SegmentationRequest(image=..., model_id=SAM3_INTERACTIVE_MODEL_ID,
    # api_key=..., source="workflow-execution",
    # prompts=Sam2PromptSet(prompts=group.prompts), multimask_output=...)` -
    # this family identifies the model by `model_id=`, not `sam2_version_id=`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = MagicMock(predictions=[])
    image = _make_image()
    block = SegmentAnything3InteractiveBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=_batch_of_one(image),
        points=[{"x": 320, "y": 240, "positive": True}],
        boxes=None,
        threshold=0.0,
        multimask_output=True,
    )

    manager.add_model.assert_called_once_with(
        model_id=SAM3_INTERACTIVE_MODEL_ID, api_key="k"
    )
    request = _captured_request(manager)
    expected = Sam2SegmentationRequest(
        image=image.to_inference_format(numpy_preferred=True),
        model_id=SAM3_INTERACTIVE_MODEL_ID,
        api_key="k",
        source="workflow-execution",
        prompts=Sam2PromptSet(
            prompts=[Sam2Prompt(points=[Point(x=320.0, y=240.0, positive=True)])]
        ),
        multimask_output=True,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_sam3_v1_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 8cde35edf:.../segment_anything3/v1.py`:
    # `Sam3SegmentationRequest(image=..., model_id=model_id, api_key=...,
    # prompts=unified_prompts, output_prob_thresh=threshold)`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = MagicMock(prompt_results=[])
    image = _make_image()
    block = SegmentAnything3BlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=_batch_of_one(image),
        model_id="sam3/sam3_final",
        class_names=["cat"],
        threshold=0.5,
    )

    manager.add_model.assert_called_once_with(model_id="sam3/sam3_final", api_key="k")
    request = _captured_request(manager)
    expected = Sam3SegmentationRequest(
        image=image.to_inference_format(numpy_preferred=True),
        model_id="sam3/sam3_final",
        api_key="k",
        prompts=[Sam3Prompt(type="text", text="cat")],
        output_prob_thresh=0.5,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_sam3_v2_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 8cde35edf:.../segment_anything3/v2.py`:
    # `Sam3SegmentationRequest(image=..., model_id=model_id, api_key=...,
    # prompts=unified_prompts, output_prob_thresh=confidence,
    # nms_iou_threshold=nms_iou_threshold if apply_nms else None)`. No
    # `format=` - v2 never sets it, so the request's own default applies.
    manager = _manager()
    manager.infer_from_request_sync.return_value = MagicMock(prompt_results=[])
    image = _make_image()
    block = SegmentAnything3BlockV2(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=_batch_of_one(image),
        model_id="sam3/sam3_final",
        class_names=["cat"],
        confidence=0.4,
        per_class_confidence=None,
        apply_nms=True,
        nms_iou_threshold=0.9,
    )

    manager.add_model.assert_called_once_with(model_id="sam3/sam3_final", api_key="k")
    request = _captured_request(manager)
    expected = Sam3SegmentationRequest(
        image=image.to_inference_format(numpy_preferred=True),
        model_id="sam3/sam3_final",
        api_key="k",
        prompts=[Sam3Prompt(type="text", text="cat", output_prob_thresh=None)],
        output_prob_thresh=0.4,
        nms_iou_threshold=0.9,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_sam3_v2_disables_nms_when_apply_nms_is_false() -> None:
    """`nms_iou_threshold=nms_iou_threshold if apply_nms else None` - the
    request stores an explicit `None`, matching the pydantic default, but the
    block always forwards it (round-5 defect 1 territory)."""
    manager = _manager()
    manager.infer_from_request_sync.return_value = MagicMock(prompt_results=[])
    image = _make_image()
    block = SegmentAnything3BlockV2(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=_batch_of_one(image),
        model_id="sam3/sam3_final",
        class_names=["cat"],
        confidence=0.4,
        per_class_confidence=None,
        apply_nms=False,
        nms_iou_threshold=0.9,
    )

    request = _captured_request(manager)
    expected = Sam3SegmentationRequest(
        image=image.to_inference_format(numpy_preferred=True),
        model_id="sam3/sam3_final",
        api_key="k",
        prompts=[Sam3Prompt(type="text", text="cat", output_prob_thresh=None)],
        output_prob_thresh=0.4,
        nms_iou_threshold=None,
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_sam3_v3_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 8cde35edf:.../segment_anything3/v3.py`:
    # `Sam3SegmentationRequest(image=..., model_id=model_id, api_key=...,
    # prompts=unified_prompts, output_prob_thresh=confidence,
    # nms_iou_threshold=nms_iou_threshold if apply_nms else None,
    # format=model_format)` - v3 alone sets `format`.
    manager = _manager()
    manager.infer_from_request_sync.return_value = MagicMock(prompt_results=[])
    image = _make_image()
    block = SegmentAnything3BlockV3(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(
        images=_batch_of_one(image),
        model_id="sam3/sam3_final",
        class_names=["cat"],
        confidence=0.5,
        per_class_confidence=None,
        apply_nms=True,
        nms_iou_threshold=0.9,
        output_format="rle",
    )

    manager.add_model.assert_called_once_with(model_id="sam3/sam3_final", api_key="k")
    request = _captured_request(manager)
    expected = Sam3SegmentationRequest(
        image=image.to_inference_format(numpy_preferred=True),
        model_id="sam3/sam3_final",
        api_key="k",
        prompts=[Sam3Prompt(type="text", text="cat", output_prob_thresh=None)],
        output_prob_thresh=0.5,
        nms_iou_threshold=0.9,
        format="rle",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_sam3_3d_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 8cde35edf:.../segment_anything3_3d/v1.py`:
    # `Sam3_3D_Objects_InferenceRequest(image=..., mask_input=converted_mask,
    # api_key=..., model_id=model_id)`. `mask_input` is a plain polygon list,
    # so `extract_masks_from_input` passes it through unchanged.
    manager = _manager()
    response = MagicMock()
    response.mesh_glb = b"mesh"
    response.gaussian_ply = b"gaussian"
    response.time = 1.0
    response.objects = []
    manager.infer_from_request_sync.return_value = response
    image = _make_image()
    mask_input = [10, 10, 100, 10, 100, 100, 10, 100]
    block = SegmentAnything3_3D_ObjectsBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=_batch_of_one(image), mask_input=_batch_of_one(mask_input))

    manager.add_model.assert_called_once_with(model_id="sam3-3d-objects", api_key="k")
    request = _captured_request(manager)
    expected = Sam3_3D_Objects_InferenceRequest(
        image=image.to_inference_format(numpy_preferred=True),
        mask_input=mask_input,
        api_key="k",
        model_id="sam3-3d-objects",
    )
    assert request.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    assert request.model_fields_set == expected.model_fields_set


def test_sam3_3d_tensor_request_matches_the_pre_port_construction() -> None:
    # Copied verbatim from `git show 8cde35edf:.../segment_anything3_3d/v1_tensor.py`:
    # `Sam3_3D_Objects_InferenceRequest(image=..., mask_input=converted_mask,
    # api_key=..., model_id=model_id)` - identical shape to the numpy sibling,
    # but `converted_mask` here comes from `extract_masks_from_input` reading a
    # tensor-native `InstanceDetections` (`instance_mask_to_numpy` per
    # instance), not a plain polygon list.
    manager = _manager()
    response = MagicMock()
    response.mesh_glb = b"mesh"
    response.gaussian_ply = b"gaussian"
    response.time = 1.0
    response.objects = []
    manager.infer_from_request_sync.return_value = response
    image = _make_image()
    dense_mask = torch.zeros((1, 8, 8), dtype=torch.bool)
    dense_mask[0, 2:6, 3:7] = True
    mask_input = InstanceDetections(
        xyxy=torch.tensor([[3.0, 2.0, 7.0, 6.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=dense_mask,
        image_metadata={"class_names": {0: "object"}, "image_dimensions": [8, 8]},
        bboxes_metadata=[{"detection_id": "d1"}],
    )
    block = SegmentAnything3_3D_ObjectsTensorBlockV1(
        model_manager=ModelManagerModelsProvider(manager),
        api_key="k",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block.run_locally(images=_batch_of_one(image), mask_input=_batch_of_one(mask_input))

    manager.add_model.assert_called_once_with(model_id="sam3-3d-objects", api_key="k")
    request = _captured_request(manager)
    # The exact mask conversion the block performs before the call
    # (`extract_masks_from_input` -> `instance_mask_to_numpy` per instance).
    converted_mask = [instance_mask_to_numpy(mask_input, 0)]
    expected = Sam3_3D_Objects_InferenceRequest(
        image=image.to_inference_format(numpy_preferred=True),
        mask_input=converted_mask,
        api_key="k",
        model_id="sam3-3d-objects",
    )
    # `mask_input` holds numpy arrays - dict/model_dump equality on those
    # raises ("truth value of an array is ambiguous"), so it is compared
    # separately with array equality; every other field goes through the
    # same dump comparison as the rest of this suite.
    assert request.model_dump(exclude={"id", "mask_input"}) == expected.model_dump(
        exclude={"id", "mask_input"}
    )
    assert request.model_fields_set == expected.model_fields_set
    assert len(request.mask_input) == len(expected.mask_input) == 1
    assert np.array_equal(request.mask_input[0], expected.mask_input[0])
