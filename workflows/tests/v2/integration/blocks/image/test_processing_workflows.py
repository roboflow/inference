"""Compiled workflows over the six V2 image processing blocks.

These run the real compiler, session and executor; nothing is called directly.
The blocks are added to ``create_catalogue()`` here because the shared
catalogue does not list them yet. Expected pixels come from the OpenCV,
scikit-image and V1 calls that V1 made, on V1's NumPy layout (2-D gray or
``(height, width, 3)`` BGR). Every comparison is exact.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.classical_cv.contrast_enhancement.v1 import (
    enhance_contrast,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.image.blur import ImageBlurBlock
from roboflow_workflows.v2.blocks.image.contrast_enhancement import (
    ContrastEnhancementBlock,
)
from roboflow_workflows.v2.blocks.image.contrast_equalization import (
    ContrastEqualizationBlock,
)
from roboflow_workflows.v2.blocks.image.grayscale import ConvertGrayscaleBlock
from roboflow_workflows.v2.blocks.image.morphology import (
    MorphologicalTransformationBlock,
)
from roboflow_workflows.v2.blocks.image.threshold import ImageThresholdBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    ResolvedParameterError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions
from skimage import exposure

CATALOGUE = create_catalogue().with_blocks(
    [
        ConvertGrayscaleBlock,
        ImageBlurBlock,
        ImageThresholdBlock,
        MorphologicalTransformationBlock,
        ContrastEnhancementBlock,
        ContrastEqualizationBlock,
    ]
)
MODES = ["run", "phases"]
IMAGES_INPUT = {"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}
MORPHOLOGY_CODES = {
    "Erosion": cv2.MORPH_ERODE,
    "Dilation": cv2.MORPH_DILATE,
    "Closing": cv2.MORPH_CLOSE,
}


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _parameter(name: str, kind: Optional[str] = None) -> Dict[str, Any]:
    parameter = {"type": "WorkflowParameter", "name": name}
    if kind is not None:
        parameter["kind"] = [kind]

    return parameter


def _definition(steps: List[dict], outputs: Dict[str, str], inputs=()) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [IMAGES_INPUT, *inputs],
        "steps": steps,
        "outputs": [_output(name, selector) for name, selector in outputs.items()],
    }

    return definition


def _compile(definition: dict, mode: str = "run", **options: Any):
    plan = compile_workflow(
        definition,
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode, **options),
    )

    return plan


def _scene(seed: int, height: int = 48, width: int = 64) -> torch.Tensor:
    """Dark noisy background, a bright noisy rectangle and isolated dots."""
    generator = np.random.default_rng(seed)
    pixels = generator.normal(70, 12, (3, height, width))
    pixels[:, 12:36, 16:48] += 120
    pixels[:, generator.integers(0, height, 20), generator.integers(0, width, 20)] = 255
    pixels[0] += 15

    scene = torch.from_numpy(np.clip(pixels, 0, 255).astype(np.uint8))

    return scene


def _to_v1(pixels: torch.Tensor) -> np.ndarray:
    array = pixels.permute(1, 2, 0).numpy()
    array = array[:, :, 0] if array.shape[2] == 1 else array[:, :, ::-1]
    v1_array = np.ascontiguousarray(array)

    return v1_array


def _from_v1(array: np.ndarray) -> torch.Tensor:
    array = array.reshape(array.shape[0], array.shape[1], -1)
    if array.shape[2] == 3:
        array = array[:, :, ::-1]

    pixels = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    return pixels


def _v1_enhance(pixels: torch.Tensor, **settings: Any) -> torch.Tensor:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="v1"),
        numpy_image=_to_v1(pixels),
    )
    enhanced = _from_v1(enhance_contrast(image=image, **settings).numpy_image)

    return enhanced


def _morphology(array: np.ndarray, operation: str, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if operation == "Opening then Closing":
        opened = cv2.morphologyEx(array, cv2.MORPH_OPEN, kernel)
        transformed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    else:
        transformed = cv2.morphologyEx(array, MORPHOLOGY_CODES[operation], kernel)

    return transformed


def _same_lineage(result: ImageData, source: ImageData) -> None:
    assert result.image_id == source.image_id
    assert result.parent == source.parent
    assert result.root == source.root


CLEANUP_STEPS = [
    {"type": "v2/convert_grayscale@v1", "name": "gray", "image": "$inputs.images"},
    {
        "type": "v2/image_blur@v1",
        "name": "blur",
        "image": "$steps.gray.image",
        "blur_type": "gaussian",
        "kernel_size": 5,
    },
    {
        "type": "v2/threshold@v1",
        "name": "threshold",
        "image": "$steps.blur.image",
        "threshold_type": "otsu",
        "thresh_value": 0,
    },
    {
        "type": "v2/morphological_transformation@v2",
        "name": "morphology",
        "image": "$steps.threshold.image",
        "kernel_size": 3,
    },
]
CLEANUP_OUTPUTS = {
    "gray": "$steps.gray.image",
    "blur": "$steps.blur.image",
    "threshold": "$steps.threshold.image",
    "morphology": "$steps.morphology.image",
}


@pytest.mark.parametrize("mode", MODES)
def test_cleanup_chain_matches_opencv_and_keeps_identity(mode: str) -> None:
    # given
    plan = _compile(_definition(CLEANUP_STEPS, CLEANUP_OUTPUTS), mode)
    scenes = [_scene(1), _scene(2, height=33, width=40)]
    images = [
        ImageData.from_tensor(scenes[0], image_id="camera-a"),
        ImageData.from_tensor(scenes[1], image_id="camera-b"),
    ]

    # when
    rows = plan.create_session().run({"images": images}).rows()

    # then
    for row, image, scene in zip(rows, images, scenes):
        gray = cv2.cvtColor(_to_v1(scene), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned = _morphology(binary, "Opening then Closing", 3)
        expected = {"gray": gray, "blur": blurred, "threshold": binary}
        expected["morphology"] = cleaned
        for name, array in expected.items():
            assert torch.equal(row[name].tensor_image, _from_v1(array)), name
            _same_lineage(row[name], image)
        assert row["threshold"].channels == 1
        assert row["morphology"].channels == 3
        assert set(np.unique(cleaned)) == {0, 255}


@pytest.mark.parametrize("mode", MODES)
def test_contrast_chain_matches_v1_and_scikit_image(mode: str) -> None:
    # given
    steps = [
        {
            "type": "v2/contrast_enhancement@v1",
            "name": "enhance",
            "image": "$inputs.images",
            "clip_limit": 2,
            "contrast_multiplier": 1.5,
            "normalize_brightness": True,
        },
        {
            "type": "v2/contrast_equalization@v1",
            "name": "equalize",
            "image": "$steps.enhance.image",
        },
    ]
    outputs = {"enhanced": "$steps.enhance.image", "equalized": "$steps.equalize.image"}
    plan = _compile(_definition(steps, outputs), mode)
    scene = _scene(3) // 2 + 40
    image = ImageData.from_tensor(scene, image_id="camera-a")

    # when
    (row,) = plan.create_session().run({"images": [image]}).rows()

    # then
    enhanced = _v1_enhance(
        scene, clip_limit=2, contrast_multiplier=1.5, normalize_brightness=True
    )
    scaled = _to_v1(enhanced).astype(np.float32) / 255
    equalized = (exposure.equalize_hist(scaled) * 255).astype(np.uint8)
    assert torch.equal(row["enhanced"].tensor_image, enhanced)
    assert torch.equal(row["equalized"].tensor_image, _from_v1(equalized))
    _same_lineage(row["equalized"], image)
    assert int(scene.min()) > 0 and int(enhanced.min()) == 0


def test_selector_knobs_change_results_between_runs_of_one_plan() -> None:
    # given
    steps = [
        {
            "type": "v2/image_blur@v1",
            "name": "blur",
            "image": "$inputs.images",
            "blur_type": "$inputs.blur_type",
            "kernel_size": "$inputs.blur_size",
        },
        {
            "type": "v2/threshold@v1",
            "name": "threshold",
            "image": "$steps.blur.image",
            "threshold_type": "$inputs.threshold_type",
            "thresh_value": "$inputs.thresh_value",
        },
        {
            "type": "v2/morphological_transformation@v2",
            "name": "morphology",
            "image": "$steps.threshold.image",
            "kernel_size": "$inputs.morphology_size",
            "operation": "$inputs.operation",
        },
        {
            "type": "v2/contrast_equalization@v1",
            "name": "equalize",
            "image": "$inputs.images",
            "equalization_type": "$inputs.equalization_type",
        },
    ]
    inputs = [
        _parameter("blur_type", "string"),
        _parameter("blur_size", "integer"),
        _parameter("threshold_type", "string"),
        _parameter("thresh_value", "integer"),
        _parameter("morphology_size", "integer"),
        _parameter("operation", "string"),
        _parameter("equalization_type", "string"),
    ]
    outputs = {
        "morphology": "$steps.morphology.image",
        "equalized": "$steps.equalize.image",
    }
    session = _compile(_definition(steps, outputs, inputs)).create_session()
    scene = _scene(4)
    first = {
        "blur_type": "average",
        "blur_size": 3,
        "threshold_type": "binary",
        "thresh_value": 100,
        "morphology_size": 3,
        "operation": "Dilation",
        "equalization_type": "Contrast Stretching",
    }
    second = {
        "blur_type": "median",
        "blur_size": 4,
        "threshold_type": "tozero_inv",
        "thresh_value": 150,
        "morphology_size": 2,
        "operation": "Erosion",
        "equalization_type": "Adaptive Equalization",
    }

    # when
    rows = [
        session.run({"images": [scene], **knobs}).rows()[0] for knobs in (first, second)
    ]

    # then
    array = _to_v1(scene)
    blurred = cv2.blur(array, (3, 3))
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    low, high = np.percentile(array, 2), np.percentile(array, 98)
    stretched = exposure.rescale_intensity(array, in_range=(low, high))
    assert torch.equal(
        rows[0]["morphology"].tensor_image, _from_v1(_morphology(binary, "Dilation", 3))
    )
    assert torch.equal(rows[0]["equalized"].tensor_image, _from_v1(stretched))
    blurred = cv2.medianBlur(array, 5)
    _, kept = cv2.threshold(blurred, 150, 255, cv2.THRESH_TOZERO_INV)
    adaptive = exposure.equalize_adapthist(
        array.astype(np.float32) / 255, clip_limit=0.03
    )
    assert torch.equal(
        rows[1]["morphology"].tensor_image, _from_v1(_morphology(kept, "Erosion", 2))
    )
    assert torch.equal(
        rows[1]["equalized"].tensor_image,
        _from_v1((adaptive * 255).astype(np.uint8)),
    )


@pytest.mark.parametrize(
    "step, value, cause, message",
    [
        (
            {"type": "v2/image_blur@v1", "blur_type": "$inputs.value"},
            "box",
            ValueError,
            "blur_type",
        ),
        (
            {"type": "v2/threshold@v1", "threshold_type": "$inputs.value"},
            "otsu",
            ValueError,
            "v2/convert_grayscale@v1",
        ),
        (
            {"type": "v2/threshold@v1", "thresh_value": "$inputs.value"},
            "127",
            ResolvedParameterError,
            "thresh_value",
        ),
        (
            {
                "type": "v2/morphological_transformation@v2",
                "kernel_size": "$inputs.value",
            },
            0,
            ResolvedParameterError,
            "kernel_size",
        ),
        (
            {
                "type": "v2/morphological_transformation@v2",
                "operation": "$inputs.value",
            },
            "Thinning",
            ValueError,
            "operation",
        ),
        (
            {
                "type": "v2/contrast_equalization@v1",
                "equalization_type": "$inputs.value",
            },
            "Histogram",
            ValueError,
            "equalization_type",
        ),
    ],
)
def test_invalid_selected_values_fail_the_step(
    step: dict, value: Any, cause: type, message: str
) -> None:
    # given
    step = {"name": "step", "image": "$inputs.images", "thresh_value": 127, **step}
    if not step["type"].startswith("v2/threshold"):
        del step["thresh_value"]
    definition = _definition(
        [step], {"image": "$steps.step.image"}, [_parameter("value")]
    )
    session = _compile(definition).create_session()

    # when
    with pytest.raises(StepExecutionError) as error:
        session.run({"images": [_scene(5)], "value": value})

    # then
    assert isinstance(error.value.__cause__, cause)
    assert message in str(error.value.__cause__)


@pytest.mark.parametrize(
    "step",
    [
        {"type": "v2/image_blur@v1", "blur_type": "box"},
        {"type": "v2/image_blur@v1", "kernel_size": 5.0},
        {"type": "v2/threshold@v1"},
        {"type": "v2/threshold@v1", "thresh_value": 127, "threshold_type": "otsu_x"},
        {"type": "v2/morphological_transformation@v2", "kernel_size": 0},
        {"type": "v2/morphological_transformation@v2", "operation": "Thinning"},
        {"type": "v2/contrast_enhancement@v1", "clip_limit": 51},
        {"type": "v2/contrast_enhancement@v1", "contrast_multiplier": 0.05},
        {"type": "v2/contrast_enhancement@v1", "clip_limit": "$inputs.value"},
        {"type": "v2/contrast_equalization@v1", "equalization_type": "CLAHE"},
    ],
)
def test_invalid_literal_parameters_fail_compilation(step: dict) -> None:
    # given
    step = {"name": "step", "image": "$inputs.images", **step}
    definition = _definition(
        [step], {"image": "$steps.step.image"}, [_parameter("value")]
    )

    # when / then
    with pytest.raises(ParamsValidationError):
        _compile(definition)


@pytest.mark.parametrize("mode", MODES)
def test_crops_keep_their_geometry_through_the_chain(mode: str) -> None:
    # given
    steps = [
        {
            "type": "v2/crop",
            "name": "crop",
            "image": "$inputs.images",
            "regions": [[4, 6, 36, 30], [20, 10, 60, 44]],
        },
        {**CLEANUP_STEPS[0], "image": "$steps.crop.crops"},
        *CLEANUP_STEPS[1:],
    ]
    outputs = {"crops": "$steps.crop.crops", **CLEANUP_OUTPUTS}
    plan = _compile(_definition(steps, outputs), mode)
    scene = _scene(6)

    # when
    (row,) = plan.create_session().run({"images": [scene]}).rows()

    # then
    assert len(row["crops"]) == len(row["morphology"]) == 2
    for crop, cleaned in zip(row["crops"], row["morphology"]):
        assert crop.root.frame_id == row["crops"][0].root.frame_id
        assert crop.parent.offset_xy != (0.0, 0.0)
        _same_lineage(cleaned, crop)
        gray = cv2.cvtColor(_to_v1(crop.tensor_image), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        expected = _morphology(binary, "Opening then Closing", 3)
        assert torch.equal(cleaned.tensor_image, _from_v1(expected))


@pytest.mark.parametrize("mode", MODES)
def test_missing_static_crop_stays_missing(mode: str) -> None:
    # given
    steps = [
        {
            "type": "v2/static_crop",
            "name": "crop",
            "image": "$inputs.images",
            "regions": [[0, 0, 16, 16], [500, 500, 520, 520]],
            "outside": "none",
        },
        {"type": "v2/image_blur@v1", "name": "blur", "image": "$steps.crop.crops"},
        {
            "type": "v2/contrast_equalization@v1",
            "name": "equalize",
            "image": "$steps.blur.image",
        },
    ]
    outputs = {"crops": "$steps.crop.crops", "equalized": "$steps.equalize.image"}
    plan = _compile(_definition(steps, outputs), mode)

    # when
    (row,) = plan.create_session().run({"images": [_scene(7)]}).rows()

    # then
    assert row["crops"][1] is None and row["equalized"][1] is None
    _same_lineage(row["equalized"][0], row["crops"][0])


@pytest.mark.parametrize("mode", MODES)
def test_requested_outputs_drop_undemanded_steps(mode: str) -> None:
    # given
    steps = [
        *CLEANUP_STEPS[:3],
        {
            "type": "v2/contrast_enhancement@v1",
            "name": "enhance",
            "image": "$inputs.images",
        },
    ]
    outputs = {
        "threshold": "$steps.threshold.image",
        "enhanced": "$steps.enhance.image",
    }
    definition = _definition(steps, outputs)

    # when
    full = _compile(definition, mode)
    demanded = _compile(definition, mode, requested_outputs=("threshold",))
    (row,) = demanded.create_session().run({"images": [_scene(8)]}).rows()

    # then
    assert [step.path for step in full.steps] == [
        ("gray",),
        ("blur",),
        ("threshold",),
        ("enhance",),
    ]
    assert [step.path for step in demanded.steps] == [
        ("gray",),
        ("blur",),
        ("threshold",),
    ]
    assert set(row) == {"threshold"}


@pytest.mark.parametrize("mode", MODES)
def test_empty_batch_runs_no_block(mode: str) -> None:
    # given
    plan = _compile(_definition(CLEANUP_STEPS, CLEANUP_OUTPUTS), mode)

    # when
    rows = plan.create_session().run({"images": []}).rows()

    # then
    assert rows == []
