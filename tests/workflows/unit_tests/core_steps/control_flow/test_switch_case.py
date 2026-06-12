import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.flow_control.switch_case.v1 import (
    BlockManifest,
    SwitchCaseBlockV1,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl


@pytest.mark.parametrize(
    "value", ["$inputs.mode", "$steps.classifier.top", "red", 5, 1.5, True]
)
def test_switch_case_manifest_parsing_when_input_is_valid(value) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/switch_case@v1",
        "name": "switch",
        "value": value,
        "cases": {"red": "$steps.on_red", "blue": "$steps.on_blue"},
        "default_next_steps": ["$steps.fallback"],
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/switch_case@v1",
        name="switch",
        value=value,
        cases={"red": "$steps.on_red", "blue": "$steps.on_blue"},
        default_next_steps=["$steps.fallback"],
    )


def test_switch_case_manifest_parsing_when_cases_and_default_are_omitted() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/switch_case@v1",
        "name": "switch",
        "value": "$inputs.mode",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.cases == {}
    assert result.default_next_steps == []


def test_switch_case_manifest_parsing_when_target_duplicated_across_cases() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/switch_case@v1",
        "name": "switch",
        "value": "$inputs.mode",
        "cases": {"red": "$steps.on_red", "crimson": "$steps.on_red"},
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_switch_case_manifest_parsing_when_target_duplicated_between_cases_and_default() -> (
    None
):
    # given
    raw_manifest = {
        "type": "roboflow_core/switch_case@v1",
        "name": "switch",
        "value": "$inputs.mode",
        "cases": {"red": "$steps.on_red"},
        "default_next_steps": ["$steps.on_red"],
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_switch_case_manifest_parsing_when_keys_collide_case_insensitively() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/switch_case@v1",
        "name": "switch",
        "value": "$inputs.mode",
        "cases": {"Red": "$steps.on_red", "red": "$steps.on_other"},
        "case_insensitive": True,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_switch_case_manifest_parsing_when_colliding_keys_allowed_case_sensitively() -> (
    None
):
    # given
    raw_manifest = {
        "type": "roboflow_core/switch_case@v1",
        "name": "switch",
        "value": "$inputs.mode",
        "cases": {"Red": "$steps.on_red", "red": "$steps.on_other"},
        "case_insensitive": False,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.cases == {"Red": "$steps.on_red", "red": "$steps.on_other"}


def test_switch_case_manifest_parsing_when_case_target_is_not_step_selector() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/switch_case@v1",
        "name": "switch",
        "value": "$inputs.mode",
        "cases": {"red": "$inputs.on_red"},
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_switch_case_run_when_case_matches() -> None:
    # given
    block = SwitchCaseBlockV1()

    # when
    result = block.run(
        value="red",
        cases={"red": "$steps.on_red", "blue": "$steps.on_blue"},
        case_insensitive=False,
        default_next_steps=["$steps.fallback"],
    )

    # then
    assert result == FlowControl(mode="select_step", context=["$steps.on_red"])


def test_switch_case_run_when_no_case_matches_and_default_defined() -> None:
    # given
    block = SwitchCaseBlockV1()

    # when
    result = block.run(
        value="green",
        cases={"red": "$steps.on_red", "blue": "$steps.on_blue"},
        case_insensitive=False,
        default_next_steps=["$steps.fallback"],
    )

    # then
    assert result == FlowControl(mode="select_step", context=["$steps.fallback"])


def test_switch_case_run_when_no_case_matches_and_default_empty() -> None:
    # given
    block = SwitchCaseBlockV1()

    # when
    result = block.run(
        value="green",
        cases={"red": "$steps.on_red"},
        case_insensitive=False,
        default_next_steps=[],
    )

    # then
    assert result == FlowControl(mode="terminate_branch")


def test_switch_case_run_when_case_insensitive_match() -> None:
    # given
    block = SwitchCaseBlockV1()

    # when
    result = block.run(
        value="RED",
        cases={"red": "$steps.on_red"},
        case_insensitive=True,
        default_next_steps=["$steps.fallback"],
    )

    # then
    assert result == FlowControl(mode="select_step", context=["$steps.on_red"])


def test_switch_case_run_when_case_sensitive_mismatch() -> None:
    # given
    block = SwitchCaseBlockV1()

    # when
    result = block.run(
        value="RED",
        cases={"red": "$steps.on_red"},
        case_insensitive=False,
        default_next_steps=["$steps.fallback"],
    )

    # then
    assert result == FlowControl(mode="select_step", context=["$steps.fallback"])


@pytest.mark.parametrize(
    "value,matching_key",
    [
        (None, "None"),
        (5, "5"),
        (True, "True"),
        (1.0, "1.0"),
    ],
)
def test_switch_case_run_coerces_value_to_string(value, matching_key) -> None:
    # given
    block = SwitchCaseBlockV1()

    # when
    result = block.run(
        value=value,
        cases={matching_key: "$steps.on_match"},
        case_insensitive=False,
        default_next_steps=[],
    )

    # then
    assert result == FlowControl(mode="select_step", context=["$steps.on_match"])


def test_switch_case_run_when_no_cases_defined() -> None:
    # given
    block = SwitchCaseBlockV1()

    # when
    result = block.run(
        value="anything",
        cases={},
        case_insensitive=False,
        default_next_steps=["$steps.fallback"],
    )

    # then
    assert result == FlowControl(mode="select_step", context=["$steps.fallback"])
