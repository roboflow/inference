"""Real compiler/session with a test-local producer and monitoring fake port."""

from datetime import datetime, timedelta

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring import (
    ModelMonitoringInferenceAggregatorBlock,
)
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import StepExecutionError

from inference_models.models.base.object_detection import Detections


class Producer(Block):
    type = "test/monitoring_producer@v1"
    outputs = {
        "predictions": Output(OBJECT_DETECTION_PREDICTION_KIND),
        "inference_id": Output(STRING_KIND),
    }

    class Params(BlockParams):
        score: Ref(FLOAT_KIND) = Field(description="Fixture confidence.")
        inference_id: Ref(STRING_KIND) = Field(description="Fixture V2 inference id.")

    def run(self, *, score, inference_id):
        predictions = Detections(
            xyxy=torch.zeros((1, 4)),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([score], dtype=torch.float64),
            image_metadata={"class_names": {0: "cat"}, "inference_id": "old-metadata"},
        )
        return {"predictions": predictions, "inference_id": inference_id}


class Port:
    def __init__(self, *, fail=False):
        self.reports = []
        self.fail = fail

    def get_workspace(self, *, api_key):
        assert isinstance(api_key, Secret)
        return "test-workspace"

    def send_report(self, *, api_key, workspace_id, predictions):
        self.reports.append(predictions)
        if self.fail:
            raise RuntimeError("invented-secret-leak-probe")


def plan(*, outputs=True, **monitor):
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "score"},
            {"type": "WorkflowParameter", "name": "inference_id"},
            {"type": "WorkflowParameter", "name": "frequency", "default_value": 5},
        ],
        "steps": [
            {
                "type": Producer.type,
                "name": "producer",
                "score": "$inputs.score",
                "inference_id": "$inputs.inference_id",
            },
            {
                "type": ModelMonitoringInferenceAggregatorBlock.type,
                "name": "monitor",
                "predictions": "$steps.producer.predictions",
                "inference_id": "$steps.producer.inference_id",
                "model_id": "project/3",
                "unique_aggregator_key": "compiled-monitor",
                **monitor,
            },
        ],
        "outputs": (
            [
                {
                    "type": "JsonField",
                    "name": name,
                    "selector": f"$steps.monitor.{name}",
                }
                for name in ("error_status", "message")
            ]
            if outputs
            else [
                {
                    "type": "JsonField",
                    "name": "id",
                    "selector": "$steps.producer.inference_id",
                }
            ]
        ),
    }
    compiled = compile_workflow(
        definition,
        catalogue=Catalogue(blocks=[Producer, ModelMonitoringInferenceAggregatorBlock]),
    )
    return compiled


@pytest.mark.parametrize("outputs", [True, False])
def test_invocations_collect_and_side_effect_is_kept_without_demanded_outputs(outputs):
    clock, port = [0], Port()
    session = plan(outputs=outputs).create_session(
        resources={
            "api_key": "invented-secret-leak-probe",
            "model_monitoring_port": port,
            "model_monitoring_clock": lambda: datetime(2026, 1, 1)
            + timedelta(seconds=clock[0]),
        }
    )
    for time, score, inference_id in [
        (0, 0.2, "source-a"),
        (4, 0.9, "source-b"),
        (5, 0.4, "source-c"),
    ]:
        clock[0] = time
        session.run({"score": score, "inference_id": inference_id})
    assert port.reports == [
        [
            {
                "model_id": "project/3",
                "class_name": "cat",
                "confidence": 0.9,
                "inference_id": "source-b",
                "model_type": "",
            }
        ]
    ]


@pytest.mark.parametrize("disabled", [True, False])
def test_compiled_failure_and_disabled_outcomes_serialize_safely(disabled):
    clock, port = [0], Port(fail=True)
    session = plan().create_session(
        resources={
            "api_key": "invented-secret-leak-probe",
            "disable_sinks": disabled,
            "model_monitoring_port": port,
            "model_monitoring_clock": lambda: datetime(2026, 1, 1)
            + timedelta(seconds=clock[0]),
        }
    )
    session.run({"score": 0.9, "inference_id": "first"})
    clock[0] = 5
    (row,) = session.run({"score": 0.8, "inference_id": "second"}).rows(serialize=True)
    assert row["error_status"] is (not disabled)
    assert "invented-secret-leak-probe" not in repr(row)
    assert len(port.reports) == (0 if disabled else 1)


def test_compiled_replayed_secret_fails_before_collection():
    port = Port()
    session = plan().create_session(
        resources={
            "api_key": Secret.unavailable("RF_KEY"),
            "model_monitoring_port": port,
        }
    )
    with pytest.raises(StepExecutionError):
        session.run({"score": 0.9, "inference_id": "id"})
    assert port.reports == []


def test_compiled_runtime_frequency_selector_and_validation():
    clock, port = [0], Port()
    session = plan(frequency="$inputs.frequency").create_session(
        resources={
            "api_key": "invented-secret-leak-probe",
            "model_monitoring_port": port,
            "model_monitoring_clock": lambda: datetime(2026, 1, 1)
            + timedelta(seconds=clock[0]),
        }
    )
    session.run({"score": 0.9, "inference_id": "first", "frequency": 1})
    clock[0] = 1
    session.run({"score": 0.8, "inference_id": "second", "frequency": 1})
    assert len(port.reports) == 1
    with pytest.raises(StepExecutionError):
        session.run({"score": 0.9, "inference_id": "third", "frequency": 0})
    assert len(port.reports) == 1


def test_two_sessions_share_runner_and_time_but_keep_pending_windows_separate():
    from threading import Event

    from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring import (
        PENDING_MESSAGE,
        SKIPPED_MESSAGE,
    )
    from roboflow_workflows.v2.blocks.sinks.runtime import (
        BACKGROUND_MESSAGE,
        BackgroundEffects,
    )

    entered, release = Event(), Event()
    clock, port, cache = [0], Port(), {}

    class Cache:
        def get(self, key):
            return cache.get(key)

        def set(self, key, value):
            cache[key] = value

    send = port.send_report

    def delayed(**kwargs):
        entered.set()
        assert release.wait(5)
        send(**kwargs)

    port.send_report = delayed
    with BackgroundEffects() as effects:
        resources = {
            "api_key": "invented-secret-leak-probe",
            "model_monitoring_port": port,
            "model_monitoring_cache": Cache(),
            "model_monitoring_clock": lambda: datetime(2026, 1, 1)
            + timedelta(seconds=clock[0]),
            "sink_effects": effects,
        }
        with plan().create_session(resources=resources) as first, plan().create_session(
            resources=resources
        ) as second:
            first.run({"score": 0.9, "inference_id": "a-old"})
            second.run({"score": 0.8, "inference_id": "b-old"})
            clock[0] = 5
            try:
                (row,) = first.run({"score": 0.2, "inference_id": "a-due"}).rows(
                    serialize=True
                )
                assert row["message"] == BACKGROUND_MESSAGE
                assert entered.wait(5)
                (row,) = first.run({"score": 0.7, "inference_id": "a-new"}).rows(
                    serialize=True
                )
                assert row["message"] == PENDING_MESSAGE
                clock[0] = 20
            finally:
                release.set()
            assert effects.drain(5)
            (row,) = second.run({"score": 0.3, "inference_id": "b-suppressed"}).rows(
                serialize=True
            )
            assert row["message"] == SKIPPED_MESSAGE
            clock[0] = 25
            second.run({"score": 0.1, "inference_id": "b-due"})
            assert effects.drain(5)
        assert [report[0]["inference_id"] for report in port.reports] == [
            "a-old",
            "b-old",
        ]
        # Session exit leaves caller-owned effects open; there is no final flush.
        assert effects.stats().completed == 2
        assert (
            effects.deliver(lambda: (False, "host still owns runner"), label="host")[0]
            is False
        )
