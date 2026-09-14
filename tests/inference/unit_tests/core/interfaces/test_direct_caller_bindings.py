"""The two direct Python callers must install the server's workflow services.

`test_workflows_composition_roots.py` pins the inventory of engine
construction sites; the four server/CLI roots are driven in
`test_image_codec_binding.py`.
This drives the callers' own code - model loading, video decode, the platform
fetch and `ExecutionEngine.init` stubbed - captures the arguments that actually
reach the engine, and then uses the dictionary the caller built: a real model
block through the compiler, and both image injection paths. A hand-written copy
of the intended dictionary would not cover that seam, which is the one the
decontamination broke.

Per-service depth belongs to the neighbouring files: usage rows to
`test_workflows_observer_row_parity.py`, the URL/local-file denial matrix to
`test_image_codec_local_file_paths.py` and `test_image_codec_binding.py`,
caller overrides of the platform bindings to
`test_workflows_platform_bindings.py`. Here it is identity
of what is bound, plus one execution through each path it enables.
"""

import importlib.util
import runpy
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import supervision as sv

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
)
from inference.core.interfaces.roboflow_platform_client import (
    workflows_platform_bindings,
)
from inference.core.interfaces.workflows_configuration import (
    server_workflows_configuration,
)
from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.interfaces.workflows_image_codec import GUARDED_IMAGE_CODEC
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.interfaces.workflows_step_error_handlers import (
    resolve_step_error_handler,
)
from inference.core.managers.base import ModelManager
from inference.core.utils import image_utils
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_image_kind,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.object_detection.v1 import (
    BlockManifest,
    RoboflowObjectDetectionModelBlockV1,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
    initialise_step,
)
from inference.core.workflows.prototypes.image_codec import (
    get_image_codec,
    reset_image_codec,
)

# tests/inference/unit_tests/core/interfaces/<this file> -> five levels up is the repo root
REPO_ROOT = Path(__file__).resolve().parents[5]
EXAMPLE = REPO_ROOT / "examples" / "run_perspective_correction.py"
BENCHMARK = (
    REPO_ROOT / "development" / "stream_interface" / "benchmark_engine_throughput.py"
)
API_KEY = "direct-caller-key"
MODEL_ID = "some-project/1"
OBJECT_DETECTION_BLOCK = "roboflow_core/roboflow_object_detection_model@v1"


@pytest.fixture(autouse=True)
def _clean_registry():
    # The callers install a PROCESS-wide codec; leaving it behind would make a
    # later test's install conflict. Same pattern as test_image_codec_binding.py.
    reset_image_codec()
    yield
    reset_image_codec()


class _CapturedEngine:
    """Stands in for the engine and remembers how the caller built it."""

    def __init__(self, init_kwargs: dict):
        self.init_kwargs = init_kwargs
        self.runs = []

    def run(self, **kwargs):
        self.runs.append(kwargs)
        # The example treats a result without dynamic zones as "nothing
        # detected" and exits before it opens any window.
        return [{}]


def _capture_engine_init(monkeypatch) -> list:
    captured = []

    def fake_init(**kwargs):
        engine = _CapturedEngine(kwargs)
        captured.append(engine)
        return engine

    monkeypatch.setattr(ExecutionEngine, "init", fake_init)
    return captured


def _fake_manager() -> MagicMock:
    """A mock with the real manager's API surface, and only that.

    `spec=ModelManager` is the point: the manager has `add_model`,
    `infer_from_request_sync` and `__contains__`, and has no `run_*` block
    methods - the gap the provider exists to close.
    """
    manager = MagicMock(spec=ModelManager)
    manager.__contains__.return_value = True  # MagicMock's default is False
    manager.infer_from_request_sync.return_value = ObjectDetectionInferenceResponse(
        image=InferenceResponseImage(width=4, height=4), predictions=[]
    )
    assert not hasattr(manager, "run_object_detection")
    return manager


def _assert_server_services_are_bound(init_kwargs: dict, manager: MagicMock) -> dict:
    init_parameters = init_kwargs["init_parameters"]

    provider = init_parameters["workflows_core.model_manager"]
    assert isinstance(provider, ModelManagerModelsProvider)
    assert provider._model_manager is manager

    assert init_parameters["workflows_core.api_key"] == API_KEY
    assert (
        init_parameters["workflows_core.step_execution_mode"] is StepExecutionMode.LOCAL
    )

    for key, value in workflows_platform_bindings().items():
        assert init_parameters[key] is value, key

    # One codec object on both injection paths: the engine's deserializer takes
    # it from here, every later reference reload takes it from the registry.
    codec = init_parameters["workflows_core.image_codec"]
    assert codec is GUARDED_IMAGE_CODEC
    assert get_image_codec() is codec

    assert isinstance(
        init_parameters["workflows_core.execution_observer"],
        UsageTrackingExecutionObserver,
    )
    assert init_parameters["workflows_core.configuration"] is (
        server_workflows_configuration()
    )
    assert init_kwargs["step_error_handler"] == resolve_step_error_handler()
    return init_parameters


def _run_real_detection_block(init_parameters: dict, manager: MagicMock) -> list:
    """A real block, built by the compiler out of the caller's own parameters.

    The block reaches `run_object_detection`, which the spec'd manager does not
    have; only the provider bound above turns that into the request the manager
    validates. Options are the shape a caller sends, not None - the adapter
    forwards them verbatim and `confidence` is a required request field.
    """
    initialised = initialise_step(
        step_manifest=BlockManifest.model_construct(
            name="detection", type=OBJECT_DETECTION_BLOCK
        ),
        block_specification=BlockSpecification(
            block_source="workflows_core",
            identifier=OBJECT_DETECTION_BLOCK,
            block_class=RoboflowObjectDetectionModelBlockV1,
            manifest_class=BlockManifest,
        ),
        explicit_init_parameters=init_parameters,
        initializers={},
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    result = initialised.step.run(
        images=Batch(content=[image], indices=[(0,)]),
        model_id=MODEL_ID,
        class_agnostic_nms=False,
        class_filter=None,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        disable_active_learning=None,
        active_learning_target_dataset=None,
    )
    manager.add_model.assert_called_once_with(model_id=MODEL_ID, api_key=API_KEY)
    request = manager.infer_from_request_sync.call_args.kwargs["request"]
    assert isinstance(request, ObjectDetectionInferenceRequest)
    assert request.model_id == MODEL_ID
    assert request.api_key == API_KEY
    assert request.confidence == 0.4
    assert len(result) == 1
    assert isinstance(result[0]["predictions"], sv.Detections)
    return result


def _exercise_both_image_paths(init_parameters: dict, image_input: dict) -> None:
    """Decode the caller's own file input, then re-load it from its reference.

    Path A is the engine's deserializer with the bound codec; Path B is a
    reference-only image reading through the process registry
    (`entities/base.py` -> `utils/images.ensure_local_image_load_allowed`). The
    spy sits on the module attribute the codec calls, so it counts exactly the
    permission checks that went through the server's policy owner. The
    allow/deny matrix itself is covered by the image-codec tests.
    """
    codec = init_parameters["workflows_core.image_codec"]
    with mock.patch.object(
        image_utils, "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM", True
    ), mock.patch.object(
        image_utils,
        "ensure_local_file_load_allowed",
        wraps=image_utils.ensure_local_file_load_allowed,
    ) as policy:
        deserialized = deserialize_image_kind(
            parameter="image", image=image_input, image_codec=codec
        )
        assert deserialized.numpy_image.shape == (16, 24, 3)
        # The reference is private; `to_inference_format()` is how a downstream
        # block sees it, and it round-trips to the caller's own input dict.
        assert deserialized.to_inference_format() == image_input
        assert policy.call_count == 1

        reloaded = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="reload"),
            image_reference=image_input["value"],
        )
        assert reloaded.numpy_image.shape == (16, 24, 3)
        assert policy.call_count == 2


def test_example_binds_the_server_services_and_reaches_a_real_model_block(
    monkeypatch, tmp_path
) -> None:
    manager = _fake_manager()
    source = tmp_path / "scene.png"
    assert cv2.imwrite(str(source), np.zeros((16, 24, 3), dtype=np.uint8))
    monkeypatch.setenv("ROBOFLOW_API_KEY", API_KEY)
    monkeypatch.setattr(
        "inference.core.managers.base.ModelManager", lambda **kwargs: manager
    )
    monkeypatch.setattr(
        sys, "argv", ["run_perspective_correction.py", "--source-path", str(source)]
    )
    captured = _capture_engine_init(monkeypatch)

    # The script's work sits under `if __name__ == "__main__"`, so it is driven
    # the way a user runs it; the empty result makes it exit before any window.
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert exit_info.value.code == 0

    assert len(captured) == 1
    engine = captured[0]
    init_parameters = _assert_server_services_are_bound(engine.init_kwargs, manager)
    image_input = engine.runs[0]["runtime_parameters"]["image"]
    assert image_input == {"type": "file", "value": str(source)}
    _exercise_both_image_paths(init_parameters, image_input)
    _run_real_detection_block(init_parameters, manager)


@pytest.mark.parametrize("tensor_mode", [False, True])
def test_benchmark_binds_the_server_services_and_reuses_predecoded_input(
    monkeypatch, tmp_path, tensor_mode: bool
) -> None:
    torch = pytest.importorskip("torch")  # the script imports it at module scope
    spec = importlib.util.spec_from_file_location(
        "benchmark_engine_throughput_under_test", BENCHMARK
    )
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    manager = _fake_manager()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    # Both data representations, on the CPU: the tensor branch is the benchmark's
    # reason to exist and must survive the rewiring without a GPU.
    monkeypatch.setattr(benchmark, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor_mode)
    monkeypatch.setattr(benchmark, "WORKFLOWS_IMAGE_TENSOR_DEVICE", torch.device("cpu"))
    monkeypatch.setattr(benchmark, "build_model_manager", lambda: manager)
    monkeypatch.setattr(benchmark, "decode_frames", lambda video_path, count: frames)
    monkeypatch.setattr(
        benchmark,
        "get_workflow_specification",
        lambda **kwargs: {"version": "1.0", "inputs": [], "steps": [], "outputs": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_engine_throughput.py",
            "--video",
            str(video),
            "--workspace",
            "some-workspace",
            "--workflow-id",
            "some-workflow",
            "--api-key",
            API_KEY,
            "--decoded-frames",
            "1",
            "--engine-runs",
            "2",
            "--warmup",
            "0",
        ],
    )
    captured = _capture_engine_init(monkeypatch)

    benchmark.main()

    assert len(captured) == 1
    engine = captured[0]
    init_parameters = _assert_server_services_are_bound(engine.init_kwargs, manager)
    assert engine.init_kwargs["workflow_id"] == "some-workflow"

    assert len(engine.runs) == 2
    prepared = engine.runs[0]["runtime_parameters"]["image"][0]
    if tensor_mode:
        assert isinstance(prepared, torch.Tensor)
        assert tuple(prepared.shape) == (3, 4, 4)
        assert prepared.dtype is torch.uint8
        assert prepared.device.type == "cpu"
    else:
        assert prepared is frames[0]
    # The materialization stays out of the timed loop: the same object again.
    assert engine.runs[1]["runtime_parameters"]["image"][0] is prepared

    if not tensor_mode:
        # Once is enough - the provider path does not depend on this caller's
        # image representation, and the example already covers the other one.
        _run_real_detection_block(init_parameters, manager)
