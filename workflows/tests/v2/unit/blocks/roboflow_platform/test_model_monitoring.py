"""Monitoring conversion, ownership and failure timing against fake host ports."""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest
import torch
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring_inference_aggregator.v1 import DISABLED_MESSAGE, FAILURE_MESSAGE, SKIPPED_MESSAGE, SUCCESS_MESSAGE, ModelMonitoringInferenceAggregatorBlock, format_predictions_for_model_monitoring
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.object_detection import Detections

KEY = "invented-monitoring-A/b+c?= private"


class Clock:
    def __init__(self):
        self.seconds = 0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return datetime(2026, 1, 1) + timedelta(seconds=self.seconds)


class Cache:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value


class Port:
    def __init__(self):
        self.reports = []
        self.workspace_error = None
        self.report_error = None

    def get_workspace(self, *, api_key):
        assert isinstance(api_key, Secret)
        if self.workspace_error:
            raise self.workspace_error
        return "invented-workspace"

    def send_report(self, *, api_key, workspace_id, predictions):
        assert isinstance(api_key, Secret)
        assert workspace_id == "invented-workspace"
        self.reports.append(predictions)
        if self.report_error:
            raise self.report_error


def detection(score=0.7, *, name="cat", inference_id="inference-1"):
    return Detections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([score], dtype=torch.float64),
        image_metadata={
            "class_names": {0: name},
            "inference_id": inference_id,
            "prediction_type": "object-detection",
        },
    )


def invoke(block, predictions=None, **overrides):
    return block.run(
        **{
            "predictions": predictions,
            "model_id": "model/1",
            "frequency": 5,
            "unique_aggregator_key": "monitor-1",
            **overrides,
        }
    )


def setup_block(**resources):
    clock, port = Clock(), Port()
    block = ModelMonitoringInferenceAggregatorBlock(
        model_monitoring_clock=clock,
        model_monitoring_port=port,
        api_key=KEY,
        **resources,
    )
    return block, clock, port


@pytest.mark.parametrize("threshold", [None, 0.2, 0.8])
def test_single_label_thresholds_raw_scores_then_rounds_and_sorts(threshold):
    metadata = {
        "class_names": {0: "low", 1: "high", 2: "middle"},
        "inference_id": "native-id",
        "prediction_type": "classification",
        "classification_confidence_threshold": threshold,
    }
    prediction = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.19999, 0.60006, 0.19996]], dtype=torch.float64),
        images_metadata=[metadata],
    )
    rows = format_predictions_for_model_monitoring(prediction, model_id="model/7")
    expected = [("high", 0.6001), ("low", 0.2), ("middle", 0.2)]
    if threshold == 0.2:
        expected = expected[:1]
    elif threshold == 0.8:
        expected = []
    assert [(r["class_name"], r["confidence"]) for r in rows] == expected
    assert all(r["model_id"] == "model/7" for r in rows)
    assert all(r["inference_id"] == "native-id" for r in rows)
    assert all(r["model_type"] == "classification" for r in rows)
    assert prediction.images_metadata == [metadata]
    assert float(prediction.confidence[0, 0]) == 0.19999


@pytest.mark.parametrize("threshold", [None, 0.2])
def test_multilabel_keeps_raw_scores_and_class_id_order(threshold):
    prediction = MultiLabelClassificationPrediction(
        class_ids=torch.tensor([1]),
        confidence=torch.tensor([0.123456789, 0.876543211], dtype=torch.float64),
        image_metadata={
            "class_names": {1: "high", 0: "low"},
            "classification_confidence_threshold": threshold,
        },
    )
    rows = format_predictions_for_model_monitoring(
        prediction, model_id="model/2", inference_id="explicit-v2-id"
    )
    expected = [0.123456789, 0.876543211] if threshold is None else [0.876543211]
    assert [r["confidence"] for r in rows] == expected
    assert all(r["inference_id"] == "explicit-v2-id" for r in rows)


def test_detection_and_keypoint_tuple_preserve_identifiers_and_missing_confidence():
    prediction = detection()
    expected = {
        "model_id": "project/3",
        "class_name": "cat",
        "confidence": 0.7,
        "inference_id": "inference-1",
        "model_type": "object-detection",
    }
    for carrier in [prediction, (object(), prediction)]:
        assert format_predictions_for_model_monitoring(
            carrier, model_id="project/3"
        ) == [expected]
    prediction.confidence = None
    prediction.image_metadata = None
    rows = format_predictions_for_model_monitoring(prediction, model_id="project/3")
    assert rows[0] == {
        **expected,
        "confidence": 0.0,
        "class_name": "",
        "inference_id": "",
        "model_type": "",
    }


def test_classification_batch_metadata_and_empty_inputs():
    prediction = ClassificationPrediction(
        class_id=torch.tensor([0, 0]),
        confidence=torch.tensor([[0.5], [0.9]]),
        images_metadata=[{"class_names": {0: "cat"}, "inference_id": "image-1"}],
    )
    rows = format_predictions_for_model_monitoring(prediction, model_id="model/1")
    assert len(rows) == 1
    assert rows[0]["inference_id"] == "image-1"
    assert format_predictions_for_model_monitoring(None, model_id="m") == []
    assert (
        format_predictions_for_model_monitoring(
            Detections(
                xyxy=torch.empty((0, 4)),
                class_id=torch.empty((0,), dtype=torch.long),
                confidence=torch.empty((0,)),
            ),
            model_id="m",
        )
        == []
    )
    with pytest.raises(ValueError, match="requires detections"):
        format_predictions_for_model_monitoring((object(), None), model_id="m")


def test_collect_then_flush_on_exact_integer_second_boundary_and_empty_report():
    block, clock, port = setup_block()
    assert invoke(block, detection())["message"] == SKIPPED_MESSAGE
    clock.seconds = 4.999999
    assert invoke(block, detection(0.9))["message"] == SKIPPED_MESSAGE
    assert port.reports == []
    clock.seconds = 5
    assert invoke(block)["message"] == SUCCESS_MESSAGE
    assert len(port.reports) == 1
    assert port.reports[0][0]["confidence"] == 0.9
    clock.seconds = 10
    assert invoke(block)["message"] == SUCCESS_MESSAGE
    assert port.reports[1] == []


def test_sources_share_buffer_models_compete_by_class_and_ties_use_model_order():
    block, clock, port = setup_block()
    invoke(block, detection(0.4, inference_id="source-a"), model_id="first/1")
    invoke(block, detection(0.8, inference_id="source-b"), model_id="second/2")
    invoke(block, detection(0.8, inference_id="source-c"), model_id="first/1")
    invoke(block, detection(0.3, name="dog"), model_id="second/2")
    clock.seconds = 5
    invoke(block)
    cat, dog = port.reports[0]
    assert cat["model_id"] == "first/1"
    assert cat["inference_id"] == "source-c"
    assert dog["class_name"] == "dog"


def test_two_instances_share_only_cache_time_not_predictions():
    cache, clock, port = Cache(), Clock(), Port()
    blocks = [
        ModelMonitoringInferenceAggregatorBlock(
            model_monitoring_cache=cache,
            model_monitoring_clock=clock,
            model_monitoring_port=port,
            api_key=KEY,
        )
        for _ in range(2)
    ]
    invoke(blocks[0], detection(0.5, name="a"))
    invoke(blocks[1], detection(0.6, name="b"))
    clock.seconds = 5
    invoke(blocks[0])
    assert invoke(blocks[1])["message"] == SKIPPED_MESSAGE
    clock.seconds = 10
    invoke(blocks[1])
    assert [[r["class_name"] for r in report] for report in port.reports] == [
        ["a"],
        ["b"],
    ]
    assert list(cache.values) == [
        "workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:"
        "monitor-1:last_report_time"
    ]


def test_different_keys_and_default_caches_do_not_share_time():
    for shared in [False, True]:
        clock, port, cache = Clock(), Port(), Cache()
        blocks = [
            ModelMonitoringInferenceAggregatorBlock(
                model_monitoring_cache=cache if shared else None,
                model_monitoring_clock=clock,
                model_monitoring_port=port,
                api_key=KEY,
            )
            for _ in range(2)
        ]
        for index, block in enumerate(blocks):
            invoke(block, detection(), unique_aggregator_key=str(index))
        clock.seconds = 5
        for index, block in enumerate(blocks):
            invoke(block, unique_aggregator_key=str(index))
        assert len(port.reports) == 2


def test_failed_upload_flushes_loses_window_advances_time_and_retries_next_window(
    caplog,
):
    block, clock, port = setup_block()
    invoke(block, detection(name="lost"))
    clock.seconds = 5
    port.report_error = RuntimeError(KEY)
    result = invoke(block)
    assert result == {"error_status": True, "message": FAILURE_MESSAGE}
    assert KEY not in repr(result) + caplog.text
    clock.seconds = 6
    assert invoke(block, detection(name="new"))["message"] == SKIPPED_MESSAGE
    port.report_error = None
    clock.seconds = 10
    invoke(block)
    assert [r["class_name"] for r in port.reports[1]] == ["new"]


def test_workspace_failure_flushes_without_advancing_time():
    block, clock, port = setup_block()
    invoke(block, detection(name="lost"))
    clock.seconds = 5
    error = PlatformRequestError(
        "workspace lookup failed", error_type="builtins.ValueError"
    )
    port.workspace_error = error
    with pytest.raises(PlatformRequestError):
        invoke(block)
    port.workspace_error = None
    invoke(block, detection(name="new"))
    assert [r["class_name"] for r in port.reports[0]] == ["new"]


def test_concurrent_calls_are_serialized_and_snapshot_inputs():
    block, clock, port = setup_block()
    original = detection(0.8)
    invoke(block, original)
    original.confidence.fill_(0.0)
    clock.seconds = 5
    send = port.send_report

    def slow_report(**arguments):
        # Simulate I/O releasing the GIL while other calls reach the due gate.
        # Removing serialization produces duplicate reports for the same window.
        time.sleep(0.02)
        send(**arguments)

    port.send_report = slow_report
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: invoke(block), range(24)))
    assert sum(r["message"] == SUCCESS_MESSAGE for r in results) == 1
    assert len(port.reports) == 1
    assert port.reports[0][0]["confidence"] == 0.8


@pytest.mark.parametrize("fire_and_forget", [False, True])
def test_both_compatibility_modes_wait_and_return_upload_failure(fire_and_forget):
    block, clock, port = setup_block()
    invoke(block)
    clock.seconds = 5
    port.report_error = TimeoutError(KEY)
    assert invoke(block, fire_and_forget=fire_and_forget)["error_status"] is True
    assert len(port.reports) == 1


def test_disabled_skips_missing_credentials_validation_and_collection():
    block = ModelMonitoringInferenceAggregatorBlock(disable_sinks=True)
    assert invoke(block, object(), frequency=-1) == {
        "error_status": False,
        "message": DISABLED_MESSAGE,
    }
    assert block._predictions == {}


@pytest.mark.parametrize("key", [None, Secret.unavailable("REPLAYED")])
def test_missing_or_unavailable_credential_fails_before_collecting(key):
    block = ModelMonitoringInferenceAggregatorBlock(api_key=key)
    with pytest.raises((ValueError, SecretUnavailableError)):
        invoke(block, detection())
    assert block._predictions == {}


@pytest.mark.parametrize("with_runner", [False, True])
@pytest.mark.parametrize("fire_and_forget", [False, True])
def test_offline_default_refuses_first_call_without_state_or_admission(
    with_runner, fire_and_forget
):
    # A missing port fails on the first call, not when a report becomes due.
    from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects

    clock, cache = Clock(), Cache()
    with BackgroundEffects() as effects:
        block = ModelMonitoringInferenceAggregatorBlock(
            api_key=KEY,
            model_monitoring_cache=cache,
            model_monitoring_clock=clock,
            sink_effects=effects if with_runner else None,
        )
        for predictions in (detection(), None):
            with pytest.raises(PlatformUnavailableError, match="model_monitoring_port"):
                invoke(block, predictions, fire_and_forget=fire_and_forget)
        assert effects.drain(5)
    assert block._predictions == {} and block._entries == 0
    assert not block._pending
    assert cache.values == {} and clock.calls == 0
    assert effects.stats().submitted == 0 and effects.failures() == ()


def test_offline_default_keeps_disabled_and_validation_outcomes():
    disabled = ModelMonitoringInferenceAggregatorBlock(
        disable_sinks=True, model_monitoring_clock=Clock()
    )
    assert invoke(disabled, detection())["message"] == DISABLED_MESSAGE
    offline = ModelMonitoringInferenceAggregatorBlock(api_key=KEY)
    with pytest.raises(ValueError, match="frequency"):
        invoke(offline, detection(), frequency=0)
    with pytest.raises(ValueError, match="must not be empty"):
        invoke(offline, detection(), model_id="")


@pytest.mark.parametrize("frequency", [0, -1, True, 1.5, "5"])
def test_invalid_runtime_frequency_fails_before_collecting(frequency):
    block, _, port = setup_block()
    with pytest.raises(ValueError, match="frequency"):
        invoke(block, detection(), frequency=frequency)
    assert port.reports == []
    assert block._predictions == {}


@pytest.mark.parametrize("identifier", ["model_id", "unique_aggregator_key"])
def test_empty_identifiers_fail_before_collection(identifier):
    block, _, _ = setup_block()
    with pytest.raises(ValueError, match="must not be empty"):
        invoke(block, detection(), **{identifier: ""})
    assert block._predictions == {}


def test_empty_classification_preserves_model_order_for_later_tied_representatives():
    block, clock, port = setup_block()
    empty = ClassificationPrediction(
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[1.0]]),
        images_metadata=[{}],
    )
    invoke(block, empty, model_id="first/1")
    invoke(block, detection(0.8), model_id="second/2")
    invoke(block, detection(0.8), model_id="first/1")
    clock.seconds = 5
    invoke(block)
    assert port.reports[0][0]["model_id"] == "first/1"


def test_unavailable_upload_raises_without_advancing_the_interval():
    cache = Cache()
    block, clock, port = setup_block(model_monitoring_cache=cache)
    invoke(block, detection())
    initial_time = next(iter(cache.values.values()))
    clock.seconds = 5
    port.report_error = PlatformUnavailableError("missing test host upload")
    with pytest.raises(PlatformUnavailableError):
        invoke(block)
    assert next(iter(cache.values.values())) == initial_time
    assert block._predictions == {}


def test_report_interval_restarts_at_completion_not_at_start():
    block, clock, port = setup_block()
    invoke(block, detection())
    clock.seconds = 5
    send = port.send_report

    def delayed_report(**arguments):
        send(**arguments)
        clock.seconds = 20

    port.send_report = delayed_report
    invoke(block)
    clock.seconds = 24.999999
    assert invoke(block)["message"] == SKIPPED_MESSAGE
    clock.seconds = 25
    assert invoke(block)["message"] == SUCCESS_MESSAGE


def test_repeated_shared_key_suppression_retains_only_vocabulary():
    cache, clock, port = Cache(), Clock(), Port()
    a, b = [
        ModelMonitoringInferenceAggregatorBlock(
            model_monitoring_cache=cache,
            model_monitoring_clock=clock,
            model_monitoring_port=port,
            api_key=KEY,
        )
        for _ in range(2)
    ]
    invoke(a)
    for index in range(200):
        clock.seconds = (index + 1) * 5
        invoke(a)
        invoke(b, detection(0.8, inference_id="first"), model_id="a")
        invoke(b, detection(0.8, inference_id="second"), model_id="b")
        invoke(b, detection(0.2, name="dog"), model_id="a")
        assert b._entries == 5  # two models and three model/class slots
        assert sum(map(len, b._predictions.values())) == 3
    clock.seconds += 5
    invoke(b)
    assert [(r["class_name"], r["inference_id"]) for r in port.reports[-1]] == [
        ("cat", "first"),
        ("dog", "inference-1"),
    ]
    assert b._entries == 0


def test_nan_rejected_atomically_and_infinities_keep_stable_ties():
    block, clock, port = setup_block()
    invoke(block, detection(float("-inf")), model_id="first")
    before = repr(block._predictions)
    with pytest.raises(ValueError, match="NaN"):
        invoke(block, detection(float("nan")), model_id="new")
    assert repr(block._predictions) == before
    invoke(block, detection(float("inf"), inference_id="winner"), model_id="second")
    invoke(block, detection(float("inf"), inference_id="tie"), model_id="second")
    clock.seconds = 5
    invoke(block)
    assert port.reports[0][0]["confidence"] == float("inf")
    assert port.reports[0][0]["inference_id"] == "winner"


def test_namespace_budget_rejects_whole_invocation_without_eviction():
    block, clock, port = setup_block(model_monitoring_max_entries=2)
    invoke(block, detection())
    before = repr(block._predictions)
    with pytest.raises(ValueError, match="namespace budget"):
        invoke(block, detection(0.9, name="new"))
    assert repr(block._predictions) == before
    assert block._entries == 2
    clock.seconds = 5
    invoke(block)
    invoke(block, detection(name="new"), model_id="new")
    assert block._entries == 2


def test_background_collects_while_workspace_is_pending_and_dates_completion():
    from threading import Event

    from roboflow_workflows.v2.blocks.roboflow_platform.model_monitoring_inference_aggregator.v1 import PENDING_MESSAGE
    from roboflow_workflows.v2.blocks.sinks.runtime import (
        BACKGROUND_MESSAGE,
        BackgroundEffects,
    )

    entered, release = Event(), Event()
    with BackgroundEffects() as effects:
        block, clock, port = setup_block(sink_effects=effects)
        invoke(block, detection(0.8, name="old"))
        workspace = port.get_workspace

        def delayed_workspace(**kwargs):
            entered.set()
            assert release.wait(5)
            return workspace(**kwargs)

        port.get_workspace = delayed_workspace
        clock.seconds = 5
        try:
            assert invoke(block)["message"] == BACKGROUND_MESSAGE
            assert entered.wait(5)
            prediction = detection(0.9, name="new")
            assert invoke(block, prediction)["message"] == PENDING_MESSAGE
            prediction.confidence.fill_(0)
            clock.seconds = 20
        finally:
            release.set()
        assert effects.drain(5)
        assert [r["class_name"] for r in port.reports[0]] == ["old"]
        clock.seconds = 24
        assert invoke(block)["message"] == SKIPPED_MESSAGE
        clock.seconds = 25
        invoke(block)
        assert effects.drain(5)
        assert port.reports[1][0]["confidence"] == 0.9


@pytest.mark.parametrize(
    "failure,advances", [("workspace", False), ("upload", True), ("unavailable", False)]
)
def test_background_failure_outcomes_and_cache_order(failure, advances, caplog):
    from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects

    cache = Cache()
    with BackgroundEffects() as effects:
        block, clock, port = setup_block(
            sink_effects=effects, model_monitoring_cache=cache
        )
        invoke(block, detection())
        initial = dict(cache.values)
        clock.seconds = 5
        if failure == "workspace":
            port.workspace_error = RuntimeError(KEY)
        else:
            port.report_error = (
                PlatformUnavailableError(KEY)
                if failure == "unavailable"
                else RuntimeError(KEY)
            )
        invoke(block)
        assert effects.drain(5)
        assert (cache.values != initial) is advances
        assert not block._pending
        assert effects.stats().failed == 1
        assert KEY not in repr(effects.failures()) + caplog.text


def test_rejected_admission_discards_window_without_advancing_and_closed_runner_raises():
    from threading import Event

    from roboflow_workflows.v2.blocks.sinks.runtime import (
        FULL_MESSAGE,
        BackgroundEffects,
    )

    release = Event()
    effects = BackgroundEffects(max_pending=1, when_full="reject")
    try:
        effects.deliver(lambda: (release.wait(5) and False, "fixture"), label="fixture")
        cache = Cache()
        block, clock, port = setup_block(
            sink_effects=effects, model_monitoring_cache=cache
        )
        invoke(block, detection(name="lost"))
        initial = dict(cache.values)
        clock.seconds = 5
        assert invoke(block) == {"error_status": True, "message": FULL_MESSAGE}
        assert not block._pending and block._predictions == {}
        assert cache.values == initial and port.reports == []
    finally:
        release.set()
        effects.close()
    with pytest.raises(RuntimeError, match="closed"):
        invoke(block, detection(name="also-lost"))
    assert not block._pending and block._predictions == {}
    assert cache.values == initial


def test_timed_close_cancels_queued_report_without_late_report_or_cache_update():
    from threading import Event

    from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects

    started, release = Event(), Event()
    effects = BackgroundEffects(max_workers=1, max_pending=2)

    def occupy():
        started.set()
        release.wait(5)
        return False, "fixture"

    effects.deliver(occupy, label="fixture")
    assert started.wait(5)
    cache = Cache()
    block, clock, port = setup_block(sink_effects=effects, model_monitoring_cache=cache)
    invoke(block, detection())
    initial = dict(cache.values)
    clock.seconds = 5
    invoke(block)
    try:
        effects.close(timeout=0)
    finally:
        release.set()
    assert effects.drain(5)
    assert effects.stats().abandoned == 1
    assert port.reports == [] and cache.values == initial
    # The host retired the old session before close; replace, never reuse it.
    assert block._pending
    replacement, _, _ = setup_block(model_monitoring_cache=cache)
    assert not replacement._pending and replacement._predictions == {}


def test_offline_message_names_actual_adapter_and_resource():
    block = ModelMonitoringInferenceAggregatorBlock(
        api_key=KEY, model_monitoring_clock=Clock()
    )
    with pytest.raises(PlatformUnavailableError) as caught:
        invoke(block, detection())
    assert '"model_monitoring_port": RoboflowModelMonitoringAdapter()' in str(
        caught.value
    )
    assert "inference.roboflow_workflows_plugin.v2.platform_monitoring_adapter" in str(
        caught.value
    )
    assert "RoboflowPlatformAdapter" not in str(caught.value)


def test_inline_backpressure_and_explicit_sync_use_same_completion_rules():
    from threading import Event, get_ident

    from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects

    release = Event()
    with BackgroundEffects(max_pending=1, when_full="run_inline") as effects:
        effects.deliver(lambda: (release.wait(5) and False, "fixture"), label="fixture")
        block, clock, port = setup_block(sink_effects=effects)
        invoke(block)
        seen = []
        original = port.send_report

        def send(**kwargs):
            seen.append(get_ident())
            original(**kwargs)

        port.send_report = send
        try:
            clock.seconds = 5
            assert invoke(block)["message"] == SUCCESS_MESSAGE
            clock.seconds = 10
            assert invoke(block, fire_and_forget=False)["message"] == SUCCESS_MESSAGE
        finally:
            release.set()
        assert seen == [get_ident(), get_ident()]
        assert effects.stats().ran_inline == 1


def test_replacement_has_empty_state_while_retired_admitted_report_finishes():
    from threading import Event

    from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects

    entered, release = Event(), Event()
    cache = Cache()
    with BackgroundEffects() as effects:
        old, clock, port = setup_block(
            sink_effects=effects, model_monitoring_cache=cache
        )
        invoke(old, detection(name="old"))
        send = port.send_report

        def delayed(**kwargs):
            entered.set()
            assert release.wait(5)
            send(**kwargs)

        port.send_report = delayed
        clock.seconds = 5
        try:
            invoke(old)
            assert entered.wait(5)
            replacement = ModelMonitoringInferenceAggregatorBlock(
                model_monitoring_cache=cache,
                model_monitoring_clock=clock,
                model_monitoring_port=port,
                api_key=KEY,
                sink_effects=effects,
            )
            assert replacement._predictions == {} and not replacement._pending
            clock.seconds = 20
        finally:
            release.set()
        assert effects.drain(5)
        assert (
            invoke(replacement, detection(name="replacement"))["message"]
            == SKIPPED_MESSAGE
        )
        assert port.reports[0][0]["class_name"] == "old"
        assert list(replacement._predictions["model/1"]) == ["replacement"]


def test_background_cache_failure_clears_pending_without_leaking(caplog):
    from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects

    class BrokenCache(Cache):
        fail = False

        def set(self, key, value):
            if self.fail:
                raise RuntimeError(KEY)
            super().set(key, value)

    cache = BrokenCache()
    with BackgroundEffects() as effects:
        block, clock, _ = setup_block(
            sink_effects=effects, model_monitoring_cache=cache
        )
        invoke(block)
        clock.seconds = 5
        cache.fail = True
        invoke(block)
        assert effects.drain(5)
        assert effects.stats().failed == 1 and not block._pending
        assert KEY not in caplog.text + repr(effects.failures())
