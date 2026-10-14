"""A selected keypoint ``annotator_type`` compiles and draws like its literal.

QA-ANN-SELECTOR-1 (decision 033): ``annotator_type: $inputs.style`` failed at
compile time. One compiled plan now draws each mode chosen at run time::

    keypoint model ─▶ keypoint_visualization(annotator_type=$inputs.style)
"""

from typing import Any, Dict, List

import numpy as np
import pytest
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError

from tests.v2.integration.blocks.visualization.annotations.scene import (
    SceneLoader,
    alley_image,
    annotation_catalogue,
    pixels_of,
    street_image,
)

MODES = ("edge", "vertex", "vertex_label")


def plan_for(annotator_type: Any):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "style"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_keypoint_detection_model@v3",
                "name": "keypoints",
                "image": "$inputs.image",
                "model_id": "shapes-keypoints",
            },
            {
                "type": "roboflow_core/keypoint_visualization@v1",
                "name": "draw",
                "image": "$inputs.image",
                "predictions": "$steps.keypoints.predictions",
                "annotator_type": annotator_type,
                # V1 has no default skeleton for two keypoints
                "edges": [[1, 2]],
                "text_scale": 0.35,
            },
        ],
        "outputs": [{"type": "JsonField", "name": "image", "selector": "$steps.draw.image"}],
    }  # fmt: skip
    plan = compile_workflow(definition, catalogue=annotation_catalogue())

    return plan


def drawn(plan, style: Any) -> List[np.ndarray]:
    session = plan.create_session({"models": ModelCache(SceneLoader())})
    images = [street_image(), alley_image()]
    rows = session.run({"image": images, "style": style}).rows()

    return [pixels_of(row["image"]) for row in rows]


def test_each_selected_mode_draws_like_the_same_literal() -> None:
    # given: one plan with a selected mode
    selected = plan_for("$inputs.style")

    pictures: Dict[str, List[np.ndarray]] = {}
    for mode in MODES:
        # when
        by_selector = drawn(selected, mode)
        by_literal = drawn(plan_for(mode), "unused")

        # then
        for selector_pixels, literal_pixels in zip(by_selector, by_literal):
            assert np.array_equal(selector_pixels, literal_pixels)
        pictures[mode] = by_selector

    # the three modes draw three different pictures
    edge, vertex, vertex_label = (pictures[mode][0] for mode in MODES)
    assert not np.array_equal(edge, vertex)
    assert not np.array_equal(vertex, vertex_label)


def test_an_invalid_selected_mode_names_the_allowed_modes() -> None:
    # given: a string; the engine's kind check rejects non-strings first
    selected = plan_for("$inputs.style")

    # when
    with pytest.raises(StepExecutionError) as error:
        drawn(selected, "dots")

    # then
    expected = (
        "annotator_type must be one of ['edge', 'vertex', 'vertex_label'], "
        "got 'dots'"
    )
    assert expected in str(error.value)


def test_an_invalid_literal_mode_still_fails_at_compile_time() -> None:
    with pytest.raises(Exception, match="annotator_type"):
        plan_for("dots")
