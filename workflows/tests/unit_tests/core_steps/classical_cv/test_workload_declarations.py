"""Per-setting workload declarations of the classical-CV blocks.

The registry census (``core_steps/test_workload_declarations_coverage.py``)
proves every block declares something. This module proves the declarations of
this category are the RIGHT ones - in particular that a block whose work is
selected by a literal manifest setting declares the selected work, not a union.
"""

from typing import List

import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.classical_cv.contours.v1 import (
    ImageContoursDetectionManifest,
)
from roboflow_workflows.core_steps.classical_cv.image_blur.v1 import ImageBlurManifest
from roboflow_workflows.core_steps.classical_cv.image_preprocessing.v1 import (
    ImagePreprocessingManifest,
)
from roboflow_workflows.core_steps.classical_cv.motion_detection.v1 import (
    MotionDetectionManifest,
)
from roboflow_workflows.core_steps.classical_cv.sift_comparison.v2 import (
    SIFTComparisonBlockManifest,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    WorkOperation,
)
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
)


def _preprocessing(task_type: str) -> ImagePreprocessingManifest:
    return ImagePreprocessingManifest(
        type="roboflow_core/image_preprocessing@v1",
        name="preprocessing",
        image="$inputs.image",
        task_type=task_type,
    )


def test_image_preprocessing_resize_declares_a_resize() -> None:
    assert _preprocessing("resize").discover_work_operations() == [
        WorkOperation.IMAGE_RESIZE
    ]


@pytest.mark.parametrize("task_type", ["rotate", "flip"])
def test_image_preprocessing_geometry_declares_a_transform(task_type: str) -> None:
    assert _preprocessing(task_type).discover_work_operations() == [
        WorkOperation.IMAGE_TRANSFORM
    ]


def test_image_preprocessing_declares_no_caveat() -> None:
    assert _preprocessing("resize").discover_portable_restrictions() == []


def _sift_comparison(visualize) -> SIFTComparisonBlockManifest:
    return SIFTComparisonBlockManifest(
        type="roboflow_core/sift_comparison@v2",
        name="comparison",
        input_1="$inputs.first",
        input_2="$inputs.second",
        visualize=visualize,
    )


def test_sift_comparison_adds_visualization_only_when_it_is_switched_on() -> None:
    assert _sift_comparison(True).discover_work_operations() == [
        WorkOperation.IMAGE_ANALYSIS,
        WorkOperation.VISUALIZATION,
    ]
    assert _sift_comparison(False).discover_work_operations() == [
        WorkOperation.IMAGE_ANALYSIS
    ]


def test_sift_comparison_reports_a_selector_as_unknown_not_as_absence() -> None:
    """A runtime value must not be answered with a complete declaration.

    Codex round-001 F002: returning a complete ``[IMAGE_ANALYSIS]`` here claims
    the block certainly does NOT visualise, which the unresolved selector does
    not license. The analysis holds either way and stays a known item; the rest
    is an explicit unknown with a reason.
    """
    declared = _sift_comparison("$inputs.visualize").discover_work_operations()
    assert isinstance(declared, Discovery)
    assert declared.complete is False
    assert declared.items == [WorkOperation.IMAGE_ANALYSIS]
    assert declared.unknown_reasons == [
        "visualize_selector_unresolved:$steps.comparison"
    ]


def test_sift_comparison_never_substitutes_the_inputs_default() -> None:
    """The referenced input's default is a runtime value, not a literal."""
    declared = _sift_comparison("$inputs.visualize").discover_work_operations()
    assert WorkOperation.VISUALIZATION not in declared.items
    assert declared.complete is False


@pytest.mark.parametrize("visualize", [True, False])
def test_sift_comparison_stays_complete_for_a_literal(visualize: bool) -> None:
    declared = _sift_comparison(visualize).discover_work_operations()
    assert isinstance(declared, list)
    assert (WorkOperation.VISUALIZATION in declared) is visualize


def test_image_preprocessing_task_type_cannot_hold_a_selector() -> None:
    """Why `image_preprocessing` needs no selector branch.

    `task_type` is a plain `Literal`, so a selector cannot reach the hook. If
    that annotation is ever widened, this test fails and the branch has to be
    added - the unknown case cannot slip in unnoticed.
    """
    with pytest.raises(ValidationError):
        _preprocessing("$inputs.task_type")


def test_a_filter_declares_filtering_and_an_analyser_declares_analysis() -> None:
    blur = ImageBlurManifest(
        type="roboflow_core/image_blur@v1", name="blur", image="$inputs.image"
    )
    contours = ImageContoursDetectionManifest(
        type="roboflow_core/contours_detection@v1",
        name="contours",
        image="$inputs.image",
    )
    assert blur.discover_work_operations() == [WorkOperation.IMAGE_FILTERING]
    assert contours.discover_work_operations() == [WorkOperation.IMAGE_ANALYSIS]


def test_motion_detection_declares_its_cross_frame_history() -> None:
    manifest = MotionDetectionManifest(
        type="roboflow_core/motion_detection@v1", name="motion", image="$inputs.image"
    )
    assert manifest.discover_work_operations() == [
        WorkOperation.IMAGE_ANALYSIS,
        WorkOperation.TEMPORAL_BUFFERING,
    ]
    restrictions: List = manifest.discover_portable_restrictions()
    assert restrictions == [
        STATEFUL_VIDEO_HTTP_SOFT_PORTABLE_RESTRICTION,
        STILL_IMAGE_INPUT_SOFT_PORTABLE_RESTRICTION,
    ]


def test_the_shared_preset_list_cannot_be_mutated_through_a_block() -> None:
    """Every block returns its own list; a caller cannot poison the preset."""
    manifest = MotionDetectionManifest(
        type="roboflow_core/motion_detection@v1", name="motion", image="$inputs.image"
    )
    first = manifest.discover_portable_restrictions()
    first.clear()
    assert len(manifest.discover_portable_restrictions()) == 2


# Adopted from the Codex round-001 reviewer reproducer: the same expectation,
# but through the public introspection API, so a regression in the hook OR in
# the way the builder normalises it is caught here permanently.
@pytest.mark.parametrize("visualize", [True, False, "$inputs.visualize"])
def test_selector_controlled_visualization_through_the_public_api(visualize) -> None:
    definition = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowParameter",
                "name": "visualize",
                "default_value": True,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/sift_comparison@v2",
                "name": "compare",
                "input_1": "$inputs.image",
                "input_2": "$inputs.image",
                "visualize": visualize,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "out",
                "selector": "$steps.compare.images_match",
            }
        ],
    }
    operations = describe_workflow_workload(definition).steps[0].operations
    if isinstance(visualize, bool):
        assert operations.complete
        assert (WorkOperation.VISUALIZATION in operations.items) is visualize
    else:
        # the input's default_value is True; the declaration must NOT adopt it
        assert not operations.complete
        assert operations.unknown_reasons == [
            "visualize_selector_unresolved:$steps.compare"
        ]
        assert operations.items == [WorkOperation.IMAGE_ANALYSIS]
