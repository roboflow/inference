import math
from typing import Any

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.classical_cv.auto_rotate_on_edges.v1 import (
    AutoRotateOnEdgesBlockV1,
    AutoRotateOnEdgesManifest,
    build_auto_rotate_matrix,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _make_vertical_stripes(
    width: int = 800, height: int = 600, spacing: int = 40, thickness: int = 3
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(0, width, spacing):
        cv2.rectangle(image, (x, 0), (x + thickness, height), (255, 255, 255), -1)
    return image


def _make_horizontal_stripes(
    width: int = 800, height: int = 600, spacing: int = 40, thickness: int = 3
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(0, height, spacing):
        cv2.rectangle(image, (0, y), (width, y + thickness), (255, 255, 255), -1)
    return image


def _make_grid(
    width: int = 800, height: int = 600, spacing: int = 40, thickness: int = 3
) -> np.ndarray:
    image = _make_vertical_stripes(width, height, spacing, thickness)
    for y in range(0, height, spacing):
        cv2.rectangle(image, (0, y), (width, y + thickness), (255, 255, 255), -1)
    return image


def _pre_rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    Rotate `image` about its center, with canvas expansion, using the exact same
    transform contract (`build_auto_rotate_matrix`) that the block itself uses
    to apply its final rotation. This makes "pre-rotate then deskew" a well-defined
    round trip: whatever angle the block reports should cancel out this rotation.
    """
    height, width = image.shape[:2]
    rotation_matrix = build_auto_rotate_matrix(width, height, angle_degrees)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _to_workflow_image(numpy_image: np.ndarray) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=numpy_image,
    )


class TestAutoRotateOnEdgesManifest:
    def test_manifest_parsing_when_minimal_valid_data_provided(self) -> None:
        # given
        data = {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "deskew",
            "image": "$inputs.image",
        }

        # when
        result = AutoRotateOnEdgesManifest.model_validate(data)

        # then
        assert result == AutoRotateOnEdgesManifest(
            type="roboflow_core/auto_rotate_on_edges@v1",
            name="deskew",
            image="$inputs.image",
            target_orientation="vertical",
            skip_below_degrees=0.4,
        )

    def test_manifest_parsing_when_all_fields_provided(self) -> None:
        # given
        data = {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "deskew",
            "image": "$inputs.image",
            "target_orientation": "horizontal",
            "skip_below_degrees": 1.5,
        }

        # when
        result = AutoRotateOnEdgesManifest.model_validate(data)

        # then
        assert result == AutoRotateOnEdgesManifest(
            type="roboflow_core/auto_rotate_on_edges@v1",
            name="deskew",
            image="$inputs.image",
            target_orientation="horizontal",
            skip_below_degrees=1.5,
        )

    @pytest.mark.parametrize("images_field_alias", ["image", "images"])
    def test_manifest_parsing_when_images_alias_used(
        self, images_field_alias: str
    ) -> None:
        # given
        data = {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "deskew",
            images_field_alias: "$inputs.image",
        }

        # when
        result = AutoRotateOnEdgesManifest.model_validate(data)

        # then
        assert result.image == "$inputs.image"

    @pytest.mark.parametrize(
        "field_name, value",
        [
            ("image", "invalid"),
            ("target_orientation", "diagonal"),
        ],
    )
    def test_manifest_parsing_when_invalid_data_provided(
        self, field_name: str, value: Any
    ) -> None:
        # given
        data = {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "deskew",
            "image": "$inputs.image",
        }
        data[field_name] = value

        # when
        with pytest.raises(ValidationError):
            _ = AutoRotateOnEdgesManifest.model_validate(data)

    def test_describe_outputs(self) -> None:
        # when
        outputs = AutoRotateOnEdgesManifest.describe_outputs()

        # then
        assert len(outputs) == 2
        names = {output.name for output in outputs}
        assert names == {"image", "angle"}

    def test_get_execution_engine_compatibility(self) -> None:
        assert (
            AutoRotateOnEdgesManifest.get_execution_engine_compatibility()
            == ">=1.3.0,<2.0.0"
        )


class TestAutoRotateOnEdgesBlock:
    def test_recovers_positive_pre_rotation_within_tolerance(self) -> None:
        # given
        block = AutoRotateOnEdgesBlockV1()
        base = _make_vertical_stripes()
        skewed = _pre_rotate(base, 5.0)

        # when
        output = block.run(
            image=_to_workflow_image(skewed),
            target_orientation="vertical",
            skip_below_degrees=0.4,
        )

        # then - recovered angle should CANCEL the applied rotation (sign pinned
        # empirically: recovered angle ~= -applied angle)
        assert abs(output["angle"] - (-5.0)) < 0.5

    def test_recovers_negative_pre_rotation_within_tolerance(self) -> None:
        # given
        block = AutoRotateOnEdgesBlockV1()
        base = _make_vertical_stripes()
        skewed = _pre_rotate(base, -12.0)

        # when
        output = block.run(
            image=_to_workflow_image(skewed),
            target_orientation="vertical",
            skip_below_degrees=0.4,
        )

        # then
        assert abs(output["angle"] - 12.0) < 0.5

    def test_90_degree_rotated_stripes_recovered_as_vertical(self) -> None:
        # given - stripes that were vertical are now horizontal after a hard
        # 90 degree rotation
        block = AutoRotateOnEdgesBlockV1()
        base = _make_vertical_stripes()
        rotated_90 = cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE)

        # when
        output = block.run(
            image=_to_workflow_image(rotated_90),
            target_orientation="vertical",
            skip_below_degrees=0.4,
        )

        # then - angle should be a 90-degree (mod-180) correction, either sign
        # is acceptable since +90 and -90 are equivalent corrections here
        angle = output["angle"]
        assert min(abs(angle - 90.0), abs(angle + 90.0)) < 0.5

    def test_horizontal_target_recovers_pre_rotation_within_tolerance(self) -> None:
        # given
        block = AutoRotateOnEdgesBlockV1()
        base = _make_horizontal_stripes()
        skewed = _pre_rotate(base, 7.0)

        # when
        output = block.run(
            image=_to_workflow_image(skewed),
            target_orientation="horizontal",
            skip_below_degrees=0.4,
        )

        # then
        assert abs(output["angle"] - (-7.0)) < 0.5

    def test_either_target_recovers_pre_rotation_on_grid_image(self) -> None:
        # given
        block = AutoRotateOnEdgesBlockV1()
        base = _make_grid()
        skewed = _pre_rotate(base, 8.0)

        # when
        output = block.run(
            image=_to_workflow_image(skewed),
            target_orientation="either",
            skip_below_degrees=0.4,
        )

        # then
        angle = output["angle"]
        assert abs(angle - (-8.0)) < 0.5
        assert abs(angle) <= 45.0

    def test_flat_image_returns_identity(self) -> None:
        # given
        block = AutoRotateOnEdgesBlockV1()
        flat = np.full((300, 300, 3), 128, dtype=np.uint8)
        wfi = _to_workflow_image(flat)

        # when
        output = block.run(
            image=wfi,
            target_orientation="vertical",
            skip_below_degrees=0.4,
        )

        # then
        assert output["angle"] == 0.0
        assert output["image"] is wfi

    def test_straight_stripes_below_skip_threshold_returns_identity(self) -> None:
        # given - already-straight stripes with a large skip threshold should
        # always be treated as "close enough" to straight
        block = AutoRotateOnEdgesBlockV1()
        base = _make_vertical_stripes()
        wfi = _to_workflow_image(base)

        # when
        output = block.run(
            image=wfi,
            target_orientation="vertical",
            skip_below_degrees=2.0,
        )

        # then
        assert output["angle"] == 0.0
        assert output["image"] is wfi

    def test_canvas_expansion_matches_expected_dimensions(self) -> None:
        # given
        block = AutoRotateOnEdgesBlockV1()
        base = _make_vertical_stripes()
        skewed = _pre_rotate(base, 5.0)
        wfi = _to_workflow_image(skewed)

        # when
        output = block.run(
            image=wfi,
            target_orientation="vertical",
            skip_below_degrees=0.4,
        )

        # then
        angle = output["angle"]
        height, width = skewed.shape[:2]
        radians = math.radians(angle)
        expected_width = int(
            (height * abs(math.sin(radians))) + (width * abs(math.cos(radians)))
        )
        expected_height = int(
            (height * abs(math.cos(radians))) + (width * abs(math.sin(radians)))
        )
        out_height, out_width = output["image"].numpy_image.shape[:2]
        assert out_width == expected_width
        assert out_height == expected_height

    def test_run_raises_value_error_for_invalid_target_orientation(self) -> None:
        # given
        block = AutoRotateOnEdgesBlockV1()
        base = _make_vertical_stripes()
        wfi = _to_workflow_image(base)

        # when / then
        with pytest.raises(ValueError):
            block.run(
                image=wfi,
                target_orientation="diagonal",
                skip_below_degrees=0.4,
            )


class TestBuildDeskewRotationMatrix:
    def test_identity_when_angle_is_zero(self) -> None:
        # when
        matrix = build_auto_rotate_matrix(width=800, height=600, angle_degrees=0.0)

        # then - a zero-degree rotation should leave points where they were
        # (canvas is not expanded when there is no rotation)
        assert matrix.shape == (2, 3)
        assert np.allclose(matrix[:, :2], np.eye(2), atol=1e-6)

    def test_matrix_shape_and_type(self) -> None:
        # when
        matrix = build_auto_rotate_matrix(width=800, height=600, angle_degrees=12.5)

        # then
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (2, 3)


class TestMaxCorrectionDegrees:
    def test_run_declines_correction_beyond_cap(self) -> None:
        # given - stripes pre-rotated well beyond the allowed correction range
        base = _make_vertical_stripes()
        rotated = _pre_rotate(base, 20.0)
        wfi = _to_workflow_image(rotated)
        block = AutoRotateOnEdgesBlockV1()

        # when
        output = block.run(
            image=wfi,
            target_orientation="vertical",
            skip_below_degrees=0.4,
            max_correction_degrees=10.0,
        )

        # then - identity passthrough, same object, no rotation applied
        assert output["angle"] == 0.0
        assert output["image"] is wfi

    def test_run_applies_correction_within_cap(self) -> None:
        # given - the same skew, but with a cap that permits it
        base = _make_vertical_stripes()
        rotated = _pre_rotate(base, 20.0)
        wfi = _to_workflow_image(rotated)
        block = AutoRotateOnEdgesBlockV1()

        # when
        output = block.run(
            image=wfi,
            target_orientation="vertical",
            skip_below_degrees=0.4,
            max_correction_degrees=45.0,
        )

        # then - the recovered angle cancels the applied pre-rotation
        assert abs(output["angle"] + 20.0) <= 0.5

    def test_manifest_accepts_max_correction_degrees(self) -> None:
        # given
        data = {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "deskew",
            "image": "$inputs.image",
            "max_correction_degrees": 45.0,
        }

        # when
        result = AutoRotateOnEdgesManifest.model_validate(data)

        # then
        assert result.max_correction_degrees == 45.0


class TestDownscaleMaxDimension:
    def test_manifest_accepts_internal_resolution(self) -> None:
        # given
        data = {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "rotate",
            "image": "$inputs.image",
            "internal_resolution": 500,
        }

        # when
        result = AutoRotateOnEdgesManifest.model_validate(data)

        # then
        assert result.internal_resolution == 500

    def test_manifest_defaults_internal_resolution_to_1000(self) -> None:
        # given
        data = {
            "type": "roboflow_core/auto_rotate_on_edges@v1",
            "name": "rotate",
            "image": "$inputs.image",
        }

        # when
        result = AutoRotateOnEdgesManifest.model_validate(data)

        # then
        assert result.internal_resolution == 1000

    def test_run_recovers_angle_at_reduced_downscale_resolution(self) -> None:
        # given - a known pre-rotation, searched on a smaller working copy
        base = _make_vertical_stripes()
        rotated = _pre_rotate(base, 5.0)
        wfi = _to_workflow_image(rotated)
        block = AutoRotateOnEdgesBlockV1()

        # when
        output = block.run(
            image=wfi,
            target_orientation="vertical",
            skip_below_degrees=0.4,
            internal_resolution=500,
        )

        # then - the recovered angle still cancels the applied rotation
        assert abs(output["angle"] + 5.0) <= 0.5
