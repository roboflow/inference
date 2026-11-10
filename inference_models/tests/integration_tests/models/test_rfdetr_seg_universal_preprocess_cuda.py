"""Device qualification for the segmentation adapter; no model download needed."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from inference_models.models.common.roboflow.model_packages import ImagePreProcessing
from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)
from inference_models.models.rfdetr.pre_processing import pre_process_network_input
from tests.unit_tests.models.rfdetr.test_triton_universal_preprocess_runtime import (
    _network_input,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _model(network=None):
    from inference_models.models.rfdetr.rfdetr_instance_segmentation_trt import (
        RFDetrForInstanceSegmentationTRT,
    )

    model = RFDetrForInstanceSegmentationTRT(
        engine=None,
        input_name="images",
        output_names=["boxes", "labels", "masks"],
        class_names=["object"],
        classes_re_mapping=None,
        inference_config=SimpleNamespace(
            image_pre_processing=ImagePreProcessing(),
            network_input=network if network is not None else _network_input(),
        ),
        trt_config=None,
        device=torch.device("cuda:0"),
        cuda_context=None,
        execution_context=None,
        trt_cuda_graph_cache=None,
        execution_plan=RFDetrExecutionPlan(
            preprocessor_id="triton-universal-v1",
            postprocessor_id="base",
            allow_compatibility_fallback=False,
            allow_runtime_failure_fallback=False,
        ),
    )

    return model


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("floating", [False, True])
def test_exact_resize_metadata_and_delayed_producer(batch, floating):
    model = _model()
    producer = torch.cuda.Stream(device=model._device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(1000000)
        images = torch.randint(
            0, 256, (batch, 3, 1080, 1920), dtype=torch.uint8, device=model._device
        )
        if floating:
            images = images.float() / 255
        actual, metadata = model.pre_process(images, input_color_format="bgr")
    expected, expected_metadata = pre_process_network_input(
        images=images,
        image_pre_processing=model._inference_config.image_pre_processing,
        network_input=model._inference_config.network_input,
        target_device=model._device,
        input_color_format="bgr",
    )
    assert torch.equal(actual, expected)
    assert metadata == expected_metadata
    assert model.last_preprocessor_selection["effective_id"] == "triton-universal-v1"
    assert model.last_preprocessor_selection["fallback_reason"] is None


def test_async_handoff_after_second_preprocess_and_input_release(monkeypatch):
    from inference_models.models.rfdetr import (
        rfdetr_instance_segmentation_trt as adapter,
    )

    model = _model()
    # Prepare reference outputs before submitting delayed work. FP32 CUDA
    # inputs avoid the uint8 staging runtime's intentional scratch-reuse wait,
    # so both preprocessing outputs can remain outstanding at the handoff.
    expected_first, _ = model.pre_process(
        torch.zeros(3, 480, 640, device=model._device)
    )
    expected_second, _ = model.pre_process(
        torch.ones(3, 480, 640, device=model._device)
    )
    producer = torch.cuda.Stream(device=model._device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(10000000)
        first, _ = model.pre_process(
            torch.zeros(3, 480, 640, device=model._device),
            independent_stage_execution=False,
        )
        second, _ = model.pre_process(
            torch.ones(3, 480, 640, device=model._device),
            independent_stage_execution=False,
        )
    # Fake only engine arithmetic, preserving the real adapter event wait and
    # allocator lifetime handoff on the engine stream.
    outputs = []

    def infer(*, pre_processed_images, stream, **kwargs):
        with torch.cuda.stream(stream):
            output = pre_processed_images.clone()
        outputs.append(output)
        return output, output, output

    monkeypatch.setattr(adapter, "infer_from_trt_engine", infer)
    monkeypatch.setattr(adapter, "use_cuda_context", lambda **kwargs: nullcontext())
    model._trt_cuda_graph_cache = object()
    future = model.forward_async(first, [])
    actual_second = model.forward(second)[0]
    del first, second
    model._inference_stream.synchronize()
    assert future is not None
    assert torch.equal(outputs[0], expected_first)
    assert torch.equal(actual_second, expected_second)
    assert not torch.equal(expected_first, expected_second)


def test_uint8_resized_bytes_match_pil_before_normalization(monkeypatch):
    """Independently diagnose resize bytes without normalization arithmetic."""
    import numpy as np
    from inference_models.models.rfdetr import (
        triton_universal_preprocess_runtime as runtime,
    )
    from PIL import Image

    # An identity byte LUT exposes the kernel's existing rounded PIL byte.
    monkeypatch.setattr(
        runtime,
        "_exact_uint8_normalization_lut",
        lambda **kwargs: torch.arange(256, dtype=torch.float32)
        .expand(3, 256)
        .contiguous(),
    )
    model = _model()
    source = np.random.default_rng(20261028).integers(
        0, 256, (1080, 1920, 3), dtype=np.uint8
    )
    actual, _ = model.pre_process(
        torch.from_numpy(source).to(model._device), input_color_format="rgb"
    )
    expected = np.array(
        Image.fromarray(source).resize((64, 64), Image.Resampling.BILINEAR)
    )
    expected = (
        torch.from_numpy(expected)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=model._device, dtype=torch.float32)
    )
    assert torch.equal(actual, expected)


def test_all_uint8_values_normalize_exactly_on_device():
    import numpy as np
    from inference_models.models.common.roboflow.model_packages import TrainingInputSize

    # No resize changes these bytes; every channel sees every possible value.
    network = _network_input().model_copy(
        update={"training_input_size": TrainingInputSize(height=16, width=16)}
    )
    model = _model(network=network)
    source = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), 3, axis=2)
    actual, _ = model.pre_process(
        torch.from_numpy(source).to(model._device), input_color_format="rgb"
    )
    expected, _ = pre_process_network_input(
        images=source,
        image_pre_processing=ImagePreProcessing(),
        network_input=network,
        target_device=model._device,
        input_color_format="rgb",
    )
    assert torch.equal(actual, expected)
