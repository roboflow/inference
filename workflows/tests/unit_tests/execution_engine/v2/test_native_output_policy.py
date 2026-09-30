"""Compiler wildcard policies and validated output-union fallbacks."""

import pytest
import torch
from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
)
from roboflow_workflows.execution_engine.v2.errors import (
    KindMismatchError,
    WorkflowExecutionError,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    BOOLEAN_KIND,
    WILDCARD_KIND,
    Kind,
)

from inference_models.models.base.object_detection import Detections


def _source(kinds, *, payload=7, identity="test/output-source", configured=False):
    if configured:

        class Source(Block):
            type = identity
            output_fields = ("label",)

            class Params(BlockParams):
                label: str = Field(default="value", description="Name of the output.")

            @classmethod
            def describe_outputs(cls, params):
                return {params.label: Output(*kinds)}

            def run(self, *, label):
                return {label: payload}

    else:

        class Source(Block):
            type = identity
            outputs = {"value": Output(*kinds)}

            def run(self):
                return {"value": payload}

    return Source


def _plan(*blocks, kinds=(), coordinates_system=None):
    catalogue = Catalogue(blocks, kinds=kinds)
    outputs = [
        {
            "type": "JsonField",
            "name": f"value{index}",
            "selector": f"$steps.s{index}.value",
        }
        for index in range(len(blocks))
    ]
    if coordinates_system is not None:
        for output in outputs:
            output["coordinates_system"] = coordinates_system
    definition = {
        "version": "2.0",
        "inputs": [],
        "steps": [
            {"type": block.type, "name": f"s{index}"}
            for index, block in enumerate(blocks)
        ],
        "outputs": outputs,
    }
    plan = compile_workflow(definition, catalogue=catalogue)

    return catalogue, plan


def _prediction(*, composite=False, empty=False):
    image = ImageData.from_tensor(
        torch.zeros(3, 20, 30, dtype=torch.uint8), image_id="root"
    ).crop((5, 7, 11, 11), image_id="crop")
    if composite:
        image = ImageData.composite(torch.zeros(3, 4, 6, dtype=torch.uint8), sources=())

    prediction = Detections(
        xyxy=torch.tensor([] if empty else [[0.0, 0.0, 4.0, 2.0]]).reshape(-1, 4),
        class_id=torch.tensor([] if empty else [0], dtype=torch.int64),
        confidence=torch.tensor([] if empty else [0.75]),
        image_metadata=image.prediction_metadata(),
    )

    return prediction


@pytest.mark.parametrize("empty", [False, True])
def test_boolean_union_member_cannot_hide_composite_root_conversion_failure(empty):
    prediction = _prediction(composite=True, empty=empty)
    source = _source(
        (OBJECT_DETECTION_PREDICTION_KIND, BOOLEAN_KIND), payload=prediction
    )
    _, plan = _plan(source)
    result = plan.create_session().run({})

    with pytest.raises(WorkflowExecutionError, match="composite|Composite"):
        result.rows()


def test_boolean_union_member_cannot_hide_native_serialization_failure():
    prediction = _prediction()
    prediction.image_metadata["unserializable"] = object()
    source = _source(
        (OBJECT_DETECTION_PREDICTION_KIND, BOOLEAN_KIND), payload=prediction
    )
    _, plan = _plan(source, coordinates_system="own")
    result = plan.create_session().run({})

    assert result.rows()[0]["value0"] is prediction
    with pytest.raises(WorkflowExecutionError, match="unsupported type object"):
        result.rows(serialize=True)


def test_missing_native_coordinate_option_means_root_but_raw_result_stays_local():
    prediction = _prediction()
    source = _source((OBJECT_DETECTION_PREDICTION_KIND,), payload=prediction)
    _, plan = _plan(source)
    result = plan.create_session().run({})

    key = next(iter(result.selections["value0"].values()))
    assert result.outputs.data[key] is prediction
    assert torch.equal(prediction.xyxy, torch.tensor([[0.0, 0.0, 4.0, 2.0]]))
    output = result.rows()[0]["value0"]
    assert output is not prediction
    assert torch.equal(output.xyxy, torch.tensor([[5.0, 7.0, 9.0, 9.0]]))


@pytest.mark.parametrize("configured", [False, True])
def test_neutral_output_wildcard_uses_the_catalogue_policy(configured):
    policy = Kind(name="*", serialize=lambda value: {"custom": value})
    source = _source((WILDCARD_KIND,), configured=configured)

    catalogue, plan = _plan(source, kinds=[policy])

    assert plan.catalogue.kind("*") is policy
    assert catalogue.kind("*") is policy
    assert plan.create_session().run({}).rows(serialize=True) == [
        {"value0": {"custom": 7}}
    ]


@pytest.mark.parametrize("explicit_first", [False, True])
def test_configured_wildcard_policy_is_discovered_in_either_order(explicit_first):
    policy = Kind(name="*", serialize=lambda value: {"custom": value})
    neutral = _source((WILDCARD_KIND,), identity="test/neutral", configured=True)
    explicit = _source((policy,), identity="test/explicit", configured=True)
    blocks = (explicit, neutral) if explicit_first else (neutral, explicit)

    catalogue, plan = _plan(*blocks)

    assert catalogue.kind("*") is WILDCARD_KIND
    assert plan.catalogue.kind("*") is policy
    assert plan.create_session().run({}).rows(serialize=True) == [
        {"value0": {"custom": 7}, "value1": {"custom": 7}}
    ]


@pytest.mark.parametrize("already_registered", [False, True])
def test_conflicting_explicit_configured_wildcards_remain_errors(already_registered):
    policy = Kind(name="*", serialize=str)
    other = Kind(name="*", serialize=lambda value: {"other": value})
    first = _source((policy,), identity="test/first", configured=True)
    second = _source((other,), identity="test/second", configured=True)

    with pytest.raises(KindMismatchError, match="different Kind"):
        _plan(first, second, kinds=[policy] if already_registered else [])


def _fail_serialization(value):
    raise ValueError("deliberate output failure")


def _fail_conversion(value, options):
    raise ValueError("deliberate output failure")


def _failing_kind(hook):
    hooks = {hook: _fail_serialization if hook == "serialize" else _fail_conversion}
    kind = Kind(name="failing", validate=lambda value: True, **hooks)

    return kind


@pytest.mark.parametrize("hook", ["serialize", "convert_output"])
def test_hook_free_union_alternative_must_validate_the_payload(hook):
    source = _source((_failing_kind(hook), BOOLEAN_KIND), payload=object())
    _, plan = _plan(source)

    with pytest.raises(
        WorkflowExecutionError, match="deliberate output failure"
    ) as error:
        plan.create_session().run({}).rows(serialize=hook == "serialize")

    assert "boolean" in str(error.value)


@pytest.mark.parametrize("hook", ["serialize", "convert_output"])
@pytest.mark.parametrize(
    "fallback", [BOOLEAN_KIND, WILDCARD_KIND], ids=["validated", "neutral"]
)
def test_valid_or_unvalidated_neutral_fallback_keeps_payload(hook, fallback):
    source = _source((_failing_kind(hook), fallback), payload=True)
    _, plan = _plan(source)

    rows = plan.create_session().run({}).rows(serialize=hook == "serialize")

    assert rows == [{"value0": True}]


@pytest.mark.parametrize("hook", ["serialize", "convert_output"])
def test_first_successful_hook_precedes_hook_free_fallback(hook):
    hooks = {
        hook: (
            (lambda value: "hooked")
            if hook == "serialize"
            else (lambda value, options: "hooked")
        )
    }
    successful = Kind(name="successful", validate=lambda value: True, **hooks)
    source = _source((BOOLEAN_KIND, _failing_kind(hook), successful), payload=True)
    _, plan = _plan(source)

    rows = plan.create_session().run({}).rows(serialize=hook == "serialize")

    assert rows == [{"value0": "hooked"}]


@pytest.mark.parametrize("hook", ["serialize", "convert_output"])
def test_rejected_hook_free_member_does_not_prevent_a_later_valid_fallback(hook):
    string = Kind(name="only-string", validate=lambda value: isinstance(value, str))
    source = _source((_failing_kind(hook), string, BOOLEAN_KIND), payload=True)
    _, plan = _plan(source)

    rows = plan.create_session().run({}).rows(serialize=hook == "serialize")

    assert rows == [{"value0": True}]
