import pytest

from inference.core.entities.responses.action_recognition import (
    ActionRecognitionPrediction,
)
from inference.core.exceptions import PayloadTooLargeError
from inference.core.utils.action_recognition_results import (
    ActionRecognitionResultBudget,
)


def prediction(label="walk"):
    return ActionRecognitionPrediction(
        start_frame_idx=0,
        end_frame_idx=1,
        class_name=label,
        class_id=0,
        confidence=0.9,
    )


def test_candidate_budget_applies_across_windows_without_truncation():
    budget = ActionRecognitionResultBudget(max_candidates=2, max_bytes=10000)
    budget.add_candidates([prediction()])
    budget.add_candidates([prediction()])
    with pytest.raises(PayloadTooLargeError, match="No partial results"):
        budget.check_candidate_count(1)
    assert budget.candidate_count == 2


@pytest.mark.parametrize("location", ["candidate", "timeline", "metadata"])
def test_byte_budget_covers_both_lists_and_metadata(location):
    budget = ActionRecognitionResultBudget(max_candidates=10, max_bytes=512)
    large = prediction("é" * 1000)
    with pytest.raises(PayloadTooLargeError, match="response exceeds"):
        if location == "candidate":
            budget.add_candidates([large])
        elif location == "timeline":
            budget.check_response([large], {})
        else:
            budget.check_response(
                [], {"per_class_confidence_thresholds": {"é" * 1000: 0.9}}
            )
