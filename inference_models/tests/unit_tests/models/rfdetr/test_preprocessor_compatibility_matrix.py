"""Compatibility matrix for the widened `triton-universal-v1` RF-DETR gate.

`UniversalFastPreprocessRuntime.check_model_compatibility` used to reject model
packages that carried `dataset_version_resize_dimensions` or an `auto-orient`
flag. Both are now accepted. The tests below pin down where that change can and
cannot have an effect, so the widened gate can be reviewed without a GPU.

Scope of the change, by RF-DETR entry point:

| RF-DETR entry point                     | Preprocessor                   | Affected | Proven by                            |
| --------------------------------------- | ------------------------------ | -------- | ------------------------------------ |
| Object detection, TensorRT              | `triton-universal-v1` (Triton) | yes      | CPU here + CUDA parity test (GPU)    |
| Object detection, TensorRT (fallback)   | reference / `threaded-exact-v1`| no       | CPU here                             |
| Object detection, ONNX                  | reference                      | no       | CPU here                             |
| Object detection, PyTorch               | reference                      | no       | CPU here                             |
| Instance segmentation, TensorRT         | `FastPreprocessRuntime`        | no       | CPU here, behaviourally (gate unchanged) |
| Key-points detection, TensorRT / ONNX   | reference                      | no       | CPU here                             |

What is proven on any platform, including CPU-only CI:

* the reference preprocessor produces byte-identical tensors and metadata with
  and without `dataset_version_resize_dimensions` whenever the resize mode is
  `STRETCH_TO`, which is the only mode the Triton gate accepts;
* the reference preprocessor never reads `auto-orient`, for every resize mode
  and for both NumPy and tensor inputs;
* the widened gate is still closed for every non-`STRETCH_TO` resize mode, where
  `dataset_version_resize_dimensions` does change reference pixels;
* every other static restriction of the gate is unchanged;
* no RF-DETR entry point other than object-detection TensorRT can reach the
  widened gate, and the sibling `FastPreprocessRuntime` gate used by instance
  segmentation still rejects dataset-version resize.

Two assertions here are source-text scans rather than executed behaviour,
because `rfdetr_instance_segmentation_trt` and `rfdetr_key_points_detection_trt`
cannot be imported without `pycuda`. That test says so in its name; every other
test in this file executes the code it describes.

What CPU execution cannot prove, and is therefore covered by the CUDA-gated
parity tests in `test_triton_universal_preprocess_runtime.py`: that the Triton
kernel itself reproduces the reference output for the newly accepted packages.
Those tests skip without CUDA, and a skip is not a pass.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    Contrast,
    ContrastType,
    Grayscale,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    StaticCrop,
    TrainingInputSize,
)
from inference_models.models.rfdetr import triton_preprocess_runtime
from inference_models.models.rfdetr.optimization.catalog import (
    RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
)
from inference_models.models.rfdetr.optimization.ids import (
    RFDETR_PREPROCESSOR_BASE,
    RFDETR_PREPROCESSOR_THREADED_EXACT_V1,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
)
from inference_models.models.rfdetr.pre_processing import (
    _needs_two_step_resize,
    pre_process_network_input,
)
from inference_models.models.rfdetr.triton_universal_preprocess_runtime import (
    UniversalFastPreprocessRuntime,
)

# Importing an ONNX entry point pulls in `onnxruntime`, which arrives with the
# `onnx-*` extras. The inference-models unit-test job installs only
# `[torch-cpu,test,model-blob-cache]`, so those imports raise there. The PyTorch
# entry point below carries the same assertion and still runs everywhere, and the
# ONNX ones run wherever the extra is installed. A skip here is an environment
# fact, not a pass.
try:
    import onnxruntime  # noqa: F401

    _ONNXRUNTIME_AVAILABLE = True
except ImportError:
    _ONNXRUNTIME_AVAILABLE = False

requires_onnxruntime = pytest.mark.skipif(
    not _ONNXRUNTIME_AVAILABLE,
    reason="onnxruntime is not installed (requires the onnx-* extra)",
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_NON_STRETCH_RESIZE_MODES = (
    ResizeMode.LETTERBOX,
    ResizeMode.CENTER_CROP,
    ResizeMode.LETTERBOX_REFLECT_EDGES,
    ResizeMode.FIT_LONGER_EDGE,
)
_DATASET_DIMENSIONS = ((32, 32), (96, 48), (48, 96))
_IMAGE_SHAPES = ((48, 80), (64, 64), (91, 37))


def _module_source(module_name: str) -> str:
    """Read a module's source without importing it.

    Only the three RF-DETR TensorRT modules
    (`rfdetr_object_detection_trt`, `rfdetr_instance_segmentation_trt`,
    `rfdetr_key_points_detection_trt`) fail to import on a CPU-only machine:
    they pull in `inference_models.models.common.cuda`, which raises
    `MissingDependencyError` when `pycuda` is absent. Every other RF-DETR
    module imports fine, and the tests below import the ones they can.

    A source scan is weaker evidence than an executed assertion: it would miss
    a behavioural change that keeps the same identifiers, and it breaks on
    harmless renames. It is used here only for the two TensorRT modules that
    cannot be imported, and the test that uses it says so in its name.
    """
    spec = importlib.util.find_spec(f"inference_models.models.rfdetr.{module_name}")
    assert spec is not None and spec.origin is not None, module_name

    return Path(spec.origin).read_text(encoding="utf-8")


def _network_input(
    *,
    resize_mode: ResizeMode = ResizeMode.STRETCH_TO,
    dataset_version_resize_dimensions=None,
) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=dataset_version_resize_dimensions,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=resize_mode,
        input_channels=3,
        scaling_factor=255,
        normalization=[list(_IMAGENET_MEAN), list(_IMAGENET_STD)],
    )


def _image(shape, *, as_tensor: bool):
    height, width = shape
    array = np.random.default_rng(11).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    if not as_tensor:
        return array
    return torch.from_numpy(array).permute(2, 0, 1).to(torch.float32) / 255.0


def _reference(image, *, image_pre_processing, network_input, input_color_format):
    return pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format=input_color_format,
    )


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("image_shape", _IMAGE_SHAPES)
@pytest.mark.parametrize("dataset_dimensions", _DATASET_DIMENSIONS)
@pytest.mark.parametrize("input_color_format", ["rgb", "bgr"])
def test_stretch_to_reference_output_ignores_dataset_version_dimensions(
    as_tensor, image_shape, dataset_dimensions, input_color_format
) -> None:
    height, width = dataset_dimensions
    image = _image(image_shape, as_tensor=as_tensor)
    plain = _network_input()
    with_dimensions = _network_input(
        dataset_version_resize_dimensions=TrainingInputSize(height=height, width=width)
    )

    expected, expected_metadata = _reference(
        image,
        image_pre_processing=ImagePreProcessing(),
        network_input=plain,
        input_color_format=input_color_format,
    )
    actual, actual_metadata = _reference(
        image,
        image_pre_processing=ImagePreProcessing(),
        network_input=with_dimensions,
        input_color_format=input_color_format,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_metadata == expected_metadata


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("image_shape", _IMAGE_SHAPES)
@pytest.mark.parametrize(
    "resize_mode", (ResizeMode.STRETCH_TO,) + _NON_STRETCH_RESIZE_MODES
)
@pytest.mark.parametrize("input_color_format", ["rgb", "bgr"])
def test_reference_preprocessing_never_reads_auto_orient(
    as_tensor, image_shape, resize_mode, input_color_format
) -> None:
    image = _image(image_shape, as_tensor=as_tensor)
    network_input = _network_input(resize_mode=resize_mode)

    expected, expected_metadata = _reference(
        image,
        image_pre_processing=ImagePreProcessing(),
        network_input=network_input,
        input_color_format=input_color_format,
    )
    actual, actual_metadata = _reference(
        image,
        image_pre_processing=ImagePreProcessing.model_validate(
            {"auto-orient": {"enabled": True}}
        ),
        network_input=network_input,
        input_color_format=input_color_format,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_metadata == expected_metadata


@pytest.mark.parametrize("resize_mode", _NON_STRETCH_RESIZE_MODES)
@pytest.mark.parametrize("dataset_dimensions", _DATASET_DIMENSIONS)
def test_non_stretch_dataset_dimensions_change_numpy_reference_output(
    resize_mode, dataset_dimensions
) -> None:
    # The boundary the widened gate must not cross: for these modes the
    # dataset-version resize is replayed on the NumPy path, so the reference
    # pixels really do depend on the dimensions. Individual geometries can
    # coincide (a fit-longer-edge intermediate can land on the same pixels as a
    # direct resize), so the difference is asserted over a set of image shapes.
    height, width = dataset_dimensions
    plain = _network_input(resize_mode=resize_mode)
    with_dimensions = _network_input(
        resize_mode=resize_mode,
        dataset_version_resize_dimensions=TrainingInputSize(height=height, width=width),
    )
    assert not _needs_two_step_resize(plain)
    assert _needs_two_step_resize(with_dimensions)

    differences = []
    for image_shape in _IMAGE_SHAPES:
        image = _image(image_shape, as_tensor=False)
        expected, _ = _reference(
            image,
            image_pre_processing=ImagePreProcessing(),
            network_input=plain,
            input_color_format="rgb",
        )
        actual, _ = _reference(
            image,
            image_pre_processing=ImagePreProcessing(),
            network_input=with_dimensions,
            input_color_format="rgb",
        )
        differences.append(not torch.equal(actual, expected))

    assert any(differences), _IMAGE_SHAPES
    compatibility = UniversalFastPreprocessRuntime.check_model_compatibility(
        image_pre_processing=ImagePreProcessing(),
        network_input=with_dimensions,
    )
    assert not compatibility.supported
    assert any("resize_mode" in reason for reason in compatibility.reasons)


@pytest.mark.parametrize("dataset_dimensions", (None,) + _DATASET_DIMENSIONS)
def test_stretch_to_never_takes_the_two_step_resize_path(dataset_dimensions) -> None:
    dimensions = (
        None
        if dataset_dimensions is None
        else TrainingInputSize(
            height=dataset_dimensions[0], width=dataset_dimensions[1]
        )
    )
    network_input = _network_input(dataset_version_resize_dimensions=dimensions)

    assert not _needs_two_step_resize(network_input)


@pytest.mark.parametrize("dataset_dimensions", (None,) + _DATASET_DIMENSIONS)
@pytest.mark.parametrize("auto_orient", [False, True])
def test_widened_gate_accepts_dataset_dimensions_and_auto_orient(
    dataset_dimensions, auto_orient
) -> None:
    dimensions = (
        None
        if dataset_dimensions is None
        else TrainingInputSize(
            height=dataset_dimensions[0], width=dataset_dimensions[1]
        )
    )
    transforms = ImagePreProcessing.model_validate(
        {"auto-orient": {"enabled": auto_orient}}
    )

    compatibility = UniversalFastPreprocessRuntime.check_model_compatibility(
        image_pre_processing=transforms,
        network_input=_network_input(dataset_version_resize_dimensions=dimensions),
    )

    assert compatibility.supported, compatibility.reasons


@pytest.mark.parametrize(
    "transforms, network_input, expected_reason_fragment",
    [
        (
            ImagePreProcessing.model_validate(
                {
                    "static-crop": StaticCrop(
                        enabled=True, x_min=0, y_min=0, x_max=100, y_max=100
                    )
                }
            ),
            _network_input(),
            "static crop",
        ),
        (
            ImagePreProcessing(
                contrast=Contrast(enabled=True, type=ContrastType.CONTRAST_STRETCHING)
            ),
            _network_input(),
            "contrast",
        ),
        (
            ImagePreProcessing(grayscale=Grayscale(enabled=True)),
            _network_input(),
            "grayscale",
        ),
        (
            ImagePreProcessing(),
            _network_input(resize_mode=ResizeMode.LETTERBOX),
            "resize_mode",
        ),
        (
            ImagePreProcessing(),
            _network_input().model_copy(update={"input_channels": 1}),
            "input_channels",
        ),
        (
            ImagePreProcessing(),
            _network_input().model_copy(update={"scaling_factor": 1}),
            "scaling_factor",
        ),
        (
            ImagePreProcessing(),
            _network_input().model_copy(update={"normalization": None}),
            "missing normalization",
        ),
    ],
)
def test_widened_gate_keeps_every_other_restriction(
    transforms, network_input, expected_reason_fragment
) -> None:
    compatibility = UniversalFastPreprocessRuntime.check_model_compatibility(
        image_pre_processing=transforms,
        network_input=network_input,
    )

    assert not compatibility.supported
    assert any(
        expected_reason_fragment in reason for reason in compatibility.reasons
    ), compatibility.reasons


def test_registry_exposes_exactly_three_preprocessor_implementations() -> None:
    assert set(RFDETR_PREPROCESSOR_IMPLEMENTATIONS) == {
        RFDETR_PREPROCESSOR_BASE,
        RFDETR_PREPROCESSOR_THREADED_EXACT_V1,
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    }


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param("rfdetr_object_detection_onnx", marks=requires_onnxruntime),
        "rfdetr_object_detection_pytorch",
        pytest.param("rfdetr_instance_segmentation_onnx", marks=requires_onnxruntime),
        pytest.param("rfdetr_key_points_detection_onnx", marks=requires_onnxruntime),
    ],
)
def test_importable_non_tensorrt_backends_bind_the_reference_preprocessor(
    module_name,
) -> None:
    module = importlib.import_module(f"inference_models.models.rfdetr.{module_name}")

    assert module.pre_process_network_input is pre_process_network_input
    assert not [
        name
        for name, value in vars(module).items()
        if value is UniversalFastPreprocessRuntime
    ]


def test_tensorrt_sibling_modules_do_not_name_the_universal_preprocessor_source_scan() -> (
    None
):
    # Source-text evidence, not behavioural: these two modules cannot be
    # imported without `pycuda`. See `_module_source`.
    for module_name in (
        "rfdetr_instance_segmentation_trt",
        "rfdetr_key_points_detection_trt",
    ):
        source = _module_source(module_name)
        assert "UniversalFastPreprocessRuntime" not in source, module_name
        assert "TritonUniversalPreprocessor" not in source, module_name


@pytest.mark.parametrize("dataset_dimensions", _DATASET_DIMENSIONS)
def test_instance_segmentation_triton_runtime_still_rejects_dataset_version_resize(
    monkeypatch, dataset_dimensions
) -> None:
    """Behavioural check on the sibling Triton runtime's unchanged gate.

    `FastPreprocessRuntime._unsupported_reason` short-circuits on
    `not _TRITON_AVAILABLE` and on a non-CUDA device before it reaches the
    `dataset_version_resize_dimensions` guard, so both short-circuits are
    neutralised here: Triton availability is patched and the runtime is
    constructed with a CUDA `torch.device`. `__init__` only stores the device,
    so no GPU is touched and `_unsupported_reason` performs no device work.
    """
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", True)
    runtime = triton_preprocess_runtime.FastPreprocessRuntime(
        device=torch.device("cuda")
    )
    image = _image((48, 80), as_tensor=False)
    height, width = dataset_dimensions

    accepted = runtime._unsupported_reason(
        images=image,
        image_size=None,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(),
    )
    rejected = runtime._unsupported_reason(
        images=image,
        image_size=None,
        image_pre_processing=ImagePreProcessing(),
        network_input=_network_input(
            dataset_version_resize_dimensions=TrainingInputSize(
                height=height, width=width
            )
        ),
    )

    # The only difference between the two calls is the dataset-version
    # dimensions, so the reason below is attributable to that guard alone.
    assert accepted is None
    assert rejected == "dataset-version resize is unsupported"


@pytest.mark.parametrize("auto_orient", [False, True])
def test_instance_segmentation_triton_runtime_ignores_auto_orient(
    monkeypatch, auto_orient
) -> None:
    monkeypatch.setattr(triton_preprocess_runtime, "_TRITON_AVAILABLE", True)
    runtime = triton_preprocess_runtime.FastPreprocessRuntime(
        device=torch.device("cuda")
    )

    reason = runtime._unsupported_reason(
        images=_image((48, 80), as_tensor=False),
        image_size=None,
        image_pre_processing=ImagePreProcessing.model_validate(
            {"auto-orient": {"enabled": auto_orient}}
        ),
        network_input=_network_input(),
    )

    assert reason is None
