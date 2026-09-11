import ast
from pathlib import Path

# tests/workflows/unit_tests/core_steps/models/roboflow/action_recognition/<file>
# -> parents[7] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[7]
BASE = REPO_ROOT / "inference/core/workflows/execution_engine/entities/base.py"


def test_base_does_not_import_server_action_recognition_entities() -> None:
    assert BASE.is_file(), BASE
    tree = ast.parse(BASE.read_text(encoding="utf-8"))
    offenders = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("inference.core.entities")
    ]
    assert offenders == []


def test_there_is_exactly_one_action_recognition_prediction_class() -> None:
    from inference.core.entities.responses.action_recognition import (
        ActionRecognitionPrediction as ServerName,
    )
    from inference.core.workflows.core_steps.models.roboflow.action_recognition.entities import (
        ActionRecognitionPrediction,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        ActionRecognitionPrediction as ReExported,
    )

    assert ServerName is ActionRecognitionPrediction
    assert ReExported is ActionRecognitionPrediction


def test_isinstance_holds_for_objects_the_server_helper_builds() -> None:
    """`serialize_wildcard_kind` dispatches on isinstance; the timeline entries
    are built by `inference.core.models.action_recognition.merge_window_segments`,
    which stays server-side (Phase 9's Task 9.9 copies it into workflows; the
    server function keeps existing)."""
    from inference.core.models.action_recognition import merge_window_segments
    from inference.core.workflows.core_steps.common.serializers import (
        serialize_wildcard_kind,
    )
    from inference_models.models.base.action_recognition import (
        ActionRecognitionPrediction as ModelSegment,
    )

    timeline = []
    merge_window_segments(
        timeline=timeline,
        frame_numbers=[10, 20, 30],
        segments=[ModelSegment(start_frame_idx=0, end_frame_idx=1, class_name="wave")],
        id_vocabulary=["wave"],
        stride=1.0,
    )
    assert len(timeline) == 1
    assert serialize_wildcard_kind(value=timeline[0]) == {
        "start_frame_idx": 10,
        "end_frame_idx": 20,
        "class": "wave",
        "class_id": 0,
    }


def test_schema_description_survives_the_move() -> None:
    from inference.core.workflows.core_steps.models.roboflow.action_recognition.entities import (
        ActionRecognitionPrediction,
    )

    schema = ActionRecognitionPrediction.model_json_schema()
    assert "One classified frame range of a video." in schema["description"]
    assert schema["properties"]["class"]["title"] == "Class"
