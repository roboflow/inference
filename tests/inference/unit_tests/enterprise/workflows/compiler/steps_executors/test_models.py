import json
import time
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from inference.core.entities.responses.cogvlm import CogVLMResponse
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors import models
from inference.enterprise.workflows.complier.steps_executors.models import (
    construct_http_client_configuration_for_classification_step,
    construct_http_client_configuration_for_detection_step,
    construct_http_client_configuration_for_keypoints_detection_step,
    construct_http_client_configuration_for_segmentation_step,
    execute_gpt_4v_request,
    filter_out_unwanted_classes,
    get_cogvlm_generations_from_remote_api,
    get_cogvlm_generations_locally,
    resolve_model_api_url,
    run_cog_vlm_prompting,
    run_qr_code_detection_step,
    run_barcode_detection_step,
    try_parse_json,
    try_parse_lmm_output_to_json,
)
from inference.enterprise.workflows.entities.steps import (
    ClassificationModel,
    ClipComparison,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    LMMConfig,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    QRCodeDetection,
    BarcodeDetection,
)


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "self-hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_self_hosted_api_is_chosen() -> None:
    # given
    some_step = ClassificationModel(
        type="ClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "http://127.0.0.1:9001"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_classification_model() -> (
    None
):
    # given
    some_step = ClassificationModel(
        type="ClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://classify.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_multi_label_classification_model() -> (
    None
):
    # given
    some_step = MultiLabelClassificationModel(
        type="MultiLabelClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://classify.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_detection_model() -> None:
    # given
    some_step = ObjectDetectionModel(
        type="ObjectDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://detect.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_keypoints_model() -> None:
    # given
    some_step = KeypointsDetectionModel(
        type="KeypointsDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://detect.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_segmentation_model() -> (
    None
):
    # given
    some_step = InstanceSegmentationModel(
        type="InstanceSegmentationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://outline.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_ocr_model() -> None:
    # given
    some_step = OCRModel(
        type="OCRModel",
        name="some",
        image="$inputs.image",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://infer.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_clip_model() -> None:
    # given
    some_step = ClipComparison(
        type="ClipComparison", name="some", image="$inputs.image", text=["a", "b"]
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://infer.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_classification_step() -> None:
    # given
    some_step = ClassificationModel(
        type="ClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
    )
    runtime_parameters = {
        "confidence": 0.7,
    }

    # when
    result = construct_http_client_configuration_for_classification_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_detection_step_step() -> None:
    # given
    some_step = ObjectDetectionModel(
        type="ObjectDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
        iou_threshold="$inputs.iou_threshold",
    )
    runtime_parameters = {"confidence": 0.7, "iou_threshold": 0.1}

    # when
    result = construct_http_client_configuration_for_detection_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert (
        result.iou_threshold == 0.1
    ), "Iou threshold must be resolved from runtime params"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_segmentation_step() -> None:
    # given
    some_step = InstanceSegmentationModel(
        type="InstanceSegmentationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
        iou_threshold="$inputs.iou_threshold",
    )
    runtime_parameters = {"confidence": 0.7, "iou_threshold": 0.1}

    # when
    result = construct_http_client_configuration_for_segmentation_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert (
        result.iou_threshold == 0.1
    ), "Iou threshold must be resolved from runtime params"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_keypoints_detection_step_step() -> (
    None
):
    # given
    some_step = KeypointsDetectionModel(
        type="KeypointsDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
        iou_threshold="$inputs.iou_threshold",
        keypoint_confidence=1.0,
    )
    runtime_parameters = {"confidence": 0.7, "iou_threshold": 0.1}

    # when
    result = construct_http_client_configuration_for_keypoints_detection_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert (
        result.iou_threshold == 0.1
    ), "Iou threshold must be resolved from runtime params"
    assert (
        result.keypoint_confidence_threshold == 1.0
    ), "keypoints threshold must be as defined in step initialisation"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"


def test_try_parse_json_when_input_is_not_json_parsable() -> None:
    # when
    result = try_parse_json(
        content="for sure not a valid JSON",
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "not_detected",
        "field_b": "not_detected",
    }, "No field detected is expected output"


def test_try_parse_json_when_input_is_json_parsable_and_some_fields_are_missing() -> (
    None
):
    # when
    result = try_parse_json(
        content=json.dumps({"field_a": "XXX", "field_c": "additional_field"}),
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "XXX",
        "field_b": "not_detected",
    }, "field_a must be extracted, `field_b` is missing and field_c should be ignored"


def test_try_parse_json_when_input_is_json_parsable_and_all_values_are_delivered() -> (
    None
):
    # when
    result = try_parse_json(
        content=json.dumps({"field_a": "XXX", "field_b": "YYY"}),
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "XXX",
        "field_b": "YYY",
    }, "Both fields must be detected with values specified in content"


def test_try_parse_lmm_output_to_json_when_no_json_to_be_found_in_input() -> None:
    # when
    result = try_parse_lmm_output_to_json(
        output="for sure not a valid JSON",
        expected_output={"field_a": "my field", "field_b": "other_field"},
    )

    # then
    assert result == {
        "field_a": "not_detected",
        "field_b": "not_detected",
    }, "No field detected is expected output"


def test_try_parse_lmm_output_to_json_when_single_json_markdown_block_with_linearised_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{"field_a": 1, "field_b": 37}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == {"field_a": 1, "field_b": 37}


def test_try_parse_lmm_output_to_json_when_single_json_markdown_block_with_multi_line_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{
    "field_a": 1, 
    "field_b": 37
}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == {"field_a": 1, "field_b": 37}


def test_try_parse_lmm_output_to_json_when_single_json_without_markdown_spotted() -> (
    None
):
    # given
    output = """
{
    "field_a": 1, 
    "field_b": 37
}
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == {"field_a": 1, "field_b": 37}


def test_try_parse_lmm_output_to_json_when_multiple_json_markdown_blocks_with_linearised_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{"field_a": 1, "field_b": 37}
```
some other comment

```json
{"field_a": 2, "field_b": 47}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == [{"field_a": 1, "field_b": 37}, {"field_a": 2, "field_b": 47}]


def test_try_parse_lmm_output_to_json_when_multiple_json_markdown_blocks_with_multi_line_document_found() -> (
    None
):
    # given
    output = """
This is some comment produced by LLM

```json
{
    "field_a": 1, 
    "field_b": 37
}
```

Some other comment
```json
{
    "field_a": 2, 
    "field_b": 47
}
```
""".strip()
    # when
    result = try_parse_lmm_output_to_json(
        output=output, expected_output={"field_a": "my field", "field_b": "other_field"}
    )

    # then
    assert result == [{"field_a": 1, "field_b": 37}, {"field_a": 2, "field_b": 47}]


@pytest.mark.asyncio
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "self-hosted")
@mock.patch.object(models.InferenceHTTPClient, "init")
async def test_get_cogvlm_generations_from_remote_api(
    inference_client_init_mock: MagicMock,
) -> None:
    # given
    client_mock = AsyncMock()
    client_mock.prompt_cogvlm_async.side_effect = [
        {"response": "Response 1: 42"},
        {"response": "Response 2: 42"},
        {"response": "Response 3: 42"},
    ]
    inference_client_init_mock.return_value = client_mock

    # when
    result = await get_cogvlm_generations_from_remote_api(
        image=[
            {"type": "numpy_object", "value": np.zeros((192, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((193, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((194, 168, 3), dtype=np.uint8)},
        ],
        prompt="What is the meaning of life?",
        api_key="some",
    )

    # then
    assert result == [
        {"content": "Response 1: 42", "image": {"width": 168, "height": 192}},
        {"content": "Response 2: 42", "image": {"width": 168, "height": 193}},
        {"content": "Response 3: 42", "image": {"width": 168, "height": 194}},
    ]


@pytest.mark.asyncio
@mock.patch.object(models, "load_core_model", MagicMock())
async def test_get_cogvlm_generations_locally() -> None:
    # given
    model_manager = AsyncMock()
    model_manager.infer_from_request.side_effect = [
        CogVLMResponse.parse_obj({"response": "Response 1: 42"}),
        CogVLMResponse.parse_obj({"response": "Response 2: 42"}),
        CogVLMResponse.parse_obj({"response": "Response 3: 42"}),
    ]

    # when
    result = await get_cogvlm_generations_locally(
        image=[
            {"type": "numpy_object", "value": np.zeros((192, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((193, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((194, 168, 3), dtype=np.uint8)},
        ],
        prompt="What is the meaning of life?",
        model_manager=model_manager,
        api_key="some",
    )

    # then
    assert result == [
        {"content": "Response 1: 42", "image": {"width": 168, "height": 192}},
        {"content": "Response 2: 42", "image": {"width": 168, "height": 193}},
        {"content": "Response 3: 42", "image": {"width": 168, "height": 194}},
    ]


@pytest.mark.asyncio
@mock.patch.object(models, "load_core_model", MagicMock())
async def test_run_cog_vlm_prompting_when_local_execution_chosen_and_no_expected_output_structure() -> (
    None
):
    # given
    model_manager = AsyncMock()
    model_manager.infer_from_request.side_effect = [
        CogVLMResponse.parse_obj({"response": "Response 1: 42"}),
        CogVLMResponse.parse_obj({"response": "Response 2: 42"}),
        CogVLMResponse.parse_obj({"response": "Response 3: 42"}),
    ]

    # when
    result = await run_cog_vlm_prompting(
        image=[
            {"type": "numpy_object", "value": np.zeros((192, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((193, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((194, 168, 3), dtype=np.uint8)},
        ],
        prompt="What is the meaning of life?",
        expected_output=None,
        model_manager=model_manager,
        api_key="some",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then
    assert len(result) == 2, "Result mus be 2-tuple"
    assert result[0] == [
        {"content": "Response 1: 42", "image": {"width": 168, "height": 192}},
        {"content": "Response 2: 42", "image": {"width": 168, "height": 193}},
        {"content": "Response 3: 42", "image": {"width": 168, "height": 194}},
    ]
    assert result[1] == [{}, {}, {}], "No meaningful parsed response expected"


@pytest.mark.asyncio
@mock.patch.object(models, "load_core_model", MagicMock())
async def test_run_cog_vlm_prompting_when_local_execution_chosen_and_json_output_structure_expected() -> (
    None
):
    # given
    model_manager = AsyncMock()
    model_manager.infer_from_request.side_effect = [
        CogVLMResponse.parse_obj({"response": json.dumps({"value": 42})}),
        CogVLMResponse.parse_obj({"response": json.dumps({"value": 43})}),
        CogVLMResponse.parse_obj({"response": json.dumps({"value": 44})}),
    ]

    # when
    result = await run_cog_vlm_prompting(
        image=[
            {"type": "numpy_object", "value": np.zeros((192, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((193, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((194, 168, 3), dtype=np.uint8)},
        ],
        prompt="What is the meaning of life?",
        expected_output={"value": "field with answer"},
        model_manager=model_manager,
        api_key="some",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then
    assert len(result) == 2, "Result mus be 2-tuple"
    assert result[0] == [
        {"content": json.dumps({"value": 42}), "image": {"width": 168, "height": 192}},
        {"content": json.dumps({"value": 43}), "image": {"width": 168, "height": 193}},
        {"content": json.dumps({"value": 44}), "image": {"width": 168, "height": 194}},
    ]
    assert result[1] == [
        {"value": 42},
        {"value": 43},
        {"value": 44},
    ], "Parsed objects expected"


@pytest.mark.asyncio
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "self-hosted")
@mock.patch.object(models.InferenceHTTPClient, "init")
async def test_run_cog_vlm_prompting_when_remote_execution_chosen_and_no_expected_output_structure(
    inference_client_init_mock: MagicMock,
) -> None:
    # given
    client_mock = AsyncMock()
    client_mock.prompt_cogvlm_async.side_effect = [
        {"response": "Response 1: 42"},
        {"response": "Response 2: 42"},
        {"response": "Response 3: 42"},
    ]
    inference_client_init_mock.return_value = client_mock

    # when
    result = await run_cog_vlm_prompting(
        image=[
            {"type": "numpy_object", "value": np.zeros((192, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((193, 168, 3), dtype=np.uint8)},
            {"type": "numpy_object", "value": np.zeros((194, 168, 3), dtype=np.uint8)},
        ],
        prompt="What is the meaning of life?",
        expected_output=None,
        model_manager=AsyncMock(),
        api_key="some",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    # then
    assert len(result) == 2, "Result mus be 2-tuple"
    assert result[0] == [
        {"content": "Response 1: 42", "image": {"width": 168, "height": 192}},
        {"content": "Response 2: 42", "image": {"width": 168, "height": 193}},
        {"content": "Response 3: 42", "image": {"width": 168, "height": 194}},
    ]
    assert result[1] == [{}, {}, {}], "No meaningful parsed response expected"


@pytest.mark.asyncio
async def test_execute_gpt_4v_request() -> None:
    # given
    client = AsyncMock()
    client.chat.completions.create.return_value = ChatCompletion(
        id="38",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="This is content from GPT",
                ),
            )
        ],
        created=int(time.time()),
        model="gpt-4-vision-preview",
        object="chat.completion",
    )

    # when
    result = await execute_gpt_4v_request(
        client=client,
        image={
            "type": "numpy_object",
            "value": np.zeros((192, 168, 3), dtype=np.uint8),
        },
        prompt="My prompt",
        lmm_config=LMMConfig(gpt_image_detail="low", max_tokens=120),
    )

    # then
    assert result == {
        "content": "This is content from GPT",
        "image": {"width": 168, "height": 192},
    }
    call_kwargs = client.chat.completions.create.call_args[1]
    assert call_kwargs["model"] == "gpt-4-vision-preview"
    assert call_kwargs["max_tokens"] == 120
    assert (
        len(call_kwargs["messages"]) == 1
    ), "Only single message is expected to be prompted"
    assert (
        call_kwargs["messages"][0]["content"][0]["text"] == "My prompt"
    ), "Text prompt is expected to be injected without modification"
    assert (
        call_kwargs["messages"][0]["content"][1]["image_url"]["detail"] == "low"
    ), "Image details level expected to be set to `low` as in LMMConfig"


@pytest.mark.asyncio
async def test_qr_code_detection() -> None:
    # given
    step = QRCodeDetection(
        type="QRCodeDetection",
        name="some",
        image="$inputs.image",
    )

    image = cv2.imread(
        "./tests/inference/unit_tests/enterprise/workflows/assets/qr.png"
    )

    # when
    _, result = await run_qr_code_detection_step(
        step=step,
        runtime_parameters={
            "image": [
                {"type": "numpy_object", "value": image, "parent_id": "$inputs.image"}
            ]
        },
        outputs_lookup={},
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then
    actual_parent_id = result["$steps.some"]["parent_id"]
    assert actual_parent_id == ["$inputs.image"]

    actual_predictions = result["$steps.some"]["predictions"][0]
    assert len(actual_predictions) == 3
    for prediction in actual_predictions:
        assert prediction["class"] == "qr_code"
        assert prediction["class_id"] == 0
        assert prediction["confidence"] == 1.0
        assert prediction["x"] > 0
        assert prediction["y"] > 0
        assert prediction["width"] > 0
        assert prediction["height"] > 0
        assert prediction["detection_id"] is not None
        assert prediction["data"] == "https://www.qrfy.com/LEwG_Gj"
        assert prediction["parent_id"] == "$inputs.image"

    actual_image = result["$steps.some"]["image"]
    assert len(actual_image) == 1
    assert actual_image[0]["height"] == 1018
    assert actual_image[0]["width"] == 2470

    actual_prediction_type = result["$steps.some"]["prediction_type"]
    assert actual_prediction_type == "qrcode-detection"


@pytest.mark.asyncio
async def test_barcode_detection() -> None:
    # given
    step = BarcodeDetection(
        type="BarcodeDetection",
        name="some",
        image="$inputs.image",
    )

    image = cv2.imread(
        "./tests/inference/unit_tests/enterprise/workflows/assets/barcodes.png"
    )

    # when
    _, result = await run_barcode_detection_step(
        step=step,
        runtime_parameters={
            "image": [
                {"type": "numpy_object", "value": image, "parent_id": "$inputs.image"}
            ]
        },
        outputs_lookup={},
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # then
    actual_parent_id = result["$steps.some"]["parent_id"]
    assert actual_parent_id == ["$inputs.image"]

    values = ["47205255193", "37637448832", "21974251554", "81685630817"]
    actual_predictions = result["$steps.some"]["predictions"][0]
    assert len(actual_predictions) == 4
    for prediction in actual_predictions:
        assert prediction["class"] == "barcode"
        assert prediction["class_id"] == 0
        assert prediction["confidence"] == 1.0
        assert prediction["x"] > 0
        assert prediction["y"] > 0
        assert prediction["width"] > 0
        assert prediction["height"] > 0
        assert prediction["detection_id"] is not None
        assert prediction["data"] in values
        assert prediction["parent_id"] == "$inputs.image"

    actual_image = result["$steps.some"]["image"]
    assert len(actual_image) == 1
    assert actual_image[0]["height"] == 480
    assert actual_image[0]["width"] == 800

    actual_prediction_type = result["$steps.some"]["prediction_type"]
    assert actual_prediction_type == "barcode-detection"


def test_filter_out_unwanted_classes_when_empty_results_provided() -> None:
    # when
    result = filter_out_unwanted_classes(
        serialised_result=[], classes_to_accept=["a", "b"]
    )

    # then
    assert result == []


def test_filter_out_unwanted_classes_when_no_class_filter_provided() -> None:
    # given
    serialised_result = [
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "a", "field": "b"},
                {"class": "b", "field": "b"},
            ],
        }
    ]

    # when
    result = filter_out_unwanted_classes(
        serialised_result=serialised_result,
        classes_to_accept=None,
    )

    # then
    assert result == [
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "a", "field": "b"},
                {"class": "b", "field": "b"},
            ],
        }
    ]


def test_filter_out_unwanted_classes_when_there_are_classes_to_be_filtered_out() -> (
    None
):
    # given
    serialised_result = [
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "a", "field": "b"},
                {"class": "b", "field": "b"},
            ],
        },
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "c", "field": "b"},
            ],
        },
    ]

    # when
    result = filter_out_unwanted_classes(
        serialised_result=serialised_result,
        classes_to_accept=["b"],
    )

    # then
    assert result == [
        {
            "image": {"height": 100, "width": 200},
            "predictions": [
                {"class": "b", "field": "b"},
            ],
        },
        {"image": {"height": 100, "width": 200}, "predictions": []},
    ]
