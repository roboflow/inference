from inference.enterprise.workflows.complier.steps_executors.utils import (
    get_image,
    resolve_parameter,
)
from inference.enterprise.workflows.entities.steps import ObjectDetectionModel


def test_get_image_when_image_to_be_found_in_input() -> None:
    # given
    step = ObjectDetectionModel(
        type="ObjectDetectionModel",
        name="detect",
        image="$inputs.image",
        model_id="$inputs.model_id",
        confidence="$inputs.confidence",
    )
    runtime_parameters = {
        "image": {"type": "url", "value": "", "parent_id": "$inputs.image"},
    }

    # when
    result = get_image(
        step=step, runtime_parameters=runtime_parameters, outputs_lookup={}
    )

    # then
    assert result == {"type": "url", "value": "", "parent_id": "$inputs.image"}


def test_get_image_when_image_to_be_found_in_steps_output() -> None:
    # given
    step = ObjectDetectionModel(
        type="ObjectDetectionModel",
        name="detect",
        image="$steps.crop.crops",
        model_id="$inputs.model_id",
        confidence="$inputs.confidence",
    )
    outputs_lookup = {
        "$steps.crop": {
            "crops": [
                {"type": "url", "value": "", "parent_id": "detection_1"},
                {"type": "url", "value": "", "parent_id": "detection_2"},
            ]
        },
    }

    # when
    result = get_image(step=step, runtime_parameters={}, outputs_lookup=outputs_lookup)

    # then
    assert result == [
        {"type": "url", "value": "", "parent_id": "detection_1"},
        {"type": "url", "value": "", "parent_id": "detection_2"},
    ]


def test_resolve_parameter_when_not_a_selector_given() -> None:
    # when
    result = resolve_parameter(
        selector_or_value=3,
        runtime_parameters={},
        outputs_lookup={},
    )

    # then
    assert result == 3


def test_resolve_parameter_when_step_output_selector_given_and_field_is_compound() -> (
    None
):
    # give
    outputs_lookup = {"$steps.three": {"param": 39}}

    # when
    result = resolve_parameter(
        selector_or_value="$steps.three.param",
        runtime_parameters={},
        outputs_lookup=outputs_lookup,
    )

    # then
    assert result == 39


def test_resolve_parameter_when_step_output_selector_given_and_field_is_simple() -> (
    None
):
    # give
    outputs_lookup = {"$steps.three": [{"param": 39}, {"param": 47}]}

    # when
    result = resolve_parameter(
        selector_or_value="$steps.three.param",
        runtime_parameters={},
        outputs_lookup=outputs_lookup,
    )

    # then
    assert result == [39, 47]


def test_resolve_parameter_when_input_selector_given() -> None:
    # give
    runtime_parameters = {"some": 39}

    # when
    result = resolve_parameter(
        selector_or_value="$inputs.some",
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert result == 39
