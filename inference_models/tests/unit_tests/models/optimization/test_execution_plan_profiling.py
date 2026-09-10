import pytest

from inference_models.models.optimization.execution_plan import InferenceExecutionPlan
from inference_models.models.optimization.runtime_metadata import SelectionSnapshot


def _payload():
    return {
        "preprocessor": "threaded-exact-v1",
        "buffer_strategy": "base",
        "scheduler": "base",
        "postprocessor": "base",
        "engine_plugin": "base",
        "allow_compatibility_fallback": False,
        "allow_runtime_failure_fallback": False,
    }


def test_execution_plan_round_trips_and_validates_for_profiling() -> None:
    payload = _payload()

    plan = InferenceExecutionPlan.from_dict(payload).validate_for_profiling()

    assert plan.to_dict() == payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("scheduler"), "exactly"),
        (lambda value: value.update(extra="base"), "exactly"),
        (lambda value: value.update(preprocessor="auto"), "must not"),
        (lambda value: value.update(preprocessor=None), "strings"),
        (
            lambda value: value.update(allow_compatibility_fallback=True),
            "disable both fallback",
        ),
    ],
)
def test_profiling_execution_plan_rejects_invalid_payloads(mutation, message) -> None:
    payload = _payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        InferenceExecutionPlan.from_dict(payload).validate_for_profiling()


def test_selection_snapshot_serializes_fallback_only_when_present() -> None:
    selected = SelectionSnapshot(requested_id="base", effective_id="base")
    fallback = SelectionSnapshot(
        requested_id="candidate",
        effective_id="base",
        fallback_reason="incompatible input",
    )

    assert selected.to_dict() == {
        "requested_id": "base",
        "effective_id": "base",
        "fallback_occurred": False,
    }
    assert fallback.to_dict()["fallback_reason"] == "incompatible input"
    assert fallback.to_dict()["fallback_occurred"]
