"""Vision event blocks with fake ports: skips, cooldown, detachment, outcomes."""

import gc
import weakref
from contextlib import nullcontext
from threading import Event

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.roboflow_platform import vision_events as events
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.blocks.roboflow_platform.vision_events import (
    EventBundle,
    RoboflowVisionEventsBlock,
    VisionEventBundleBlock,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    FULL_MESSAGE,
    BackgroundEffects,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

ADAPTER_MODULE = "inference.roboflow_workflows_plugin.v2.platform_events_adapter"


def arguments(block=RoboflowVisionEventsBlock, **changes):
    required = {"event_type": "custom", "solution": "use-case"}
    if block is VisionEventBundleBlock:
        required["target_directory"] = "/bundles"
    values = block.Params(**required).model_dump()
    values.update(fire_and_forget=False, cooldown_seconds=0, custom_value="alert")
    values.update(changes)
    return values


def image(value=50):
    return ImageData.from_tensor(torch.full((3, 4, 5), value, dtype=torch.uint8))


class Port:
    """Records calls; ``result`` is returned, or raised when an exception."""

    def __init__(self, result="event-1"):
        self.result = result
        self.calls = []

    def _answer(self, call):
        self.calls.append(call)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def submit(self, request, **kwargs):
        return self._answer(("submit", request, kwargs))

    def prepare_destination(self, **kwargs):
        self.calls.append(("prepare", kwargs))

    def create_bundle(self, request, **kwargs):
        return self._answer(("bundle", request, kwargs))


def bundle_port():
    return Port(EventBundle(event_id="event-2", bundle_path="/bundles/b.tar.gz"))


def held_effects(**options):
    """BackgroundEffects whose only worker is busy until ``release`` is set."""
    effects = BackgroundEffects(max_workers=1, **options)
    started, release = Event(), Event()

    def hold():
        started.set()
        assert release.wait(5)
        return False, "held"

    effects.deliver(hold, label="hold")
    assert started.wait(5)
    return effects, release


@pytest.mark.parametrize("block", [RoboflowVisionEventsBlock, VisionEventBundleBlock])
@pytest.mark.parametrize("policy,step", [(True, False), (False, True)])
def test_disabled_needs_no_key_and_touches_no_port(block, policy, step):
    port = Port(AssertionError("port called"))
    resource = (
        {"vision_events_port": port}
        if block is RoboflowVisionEventsBlock
        else {"vision_event_bundle_port": port}
    )

    result = block(disable_sinks=policy, **resource).run(
        **arguments(block, disable_sink=step)
    )

    expected = "execution policy" if policy else "parameter `disable_sink`"
    assert expected in result["message"]
    assert result["error_status"] is False and result["event_id"] == ""
    assert result.get("bundle_path", "") == "" and not port.calls


def test_cloud_needs_a_key_and_local_store_does_not():
    with pytest.raises(ValueError, match="API key"):
        RoboflowVisionEventsBlock(vision_events_port=Port()).run(**arguments())
    with pytest.raises(SecretUnavailableError):
        RoboflowVisionEventsBlock(
            vision_events_port=Port(), api_key=Secret.unavailable("key")
        ).run(**arguments())

    port = Port("local-1")
    result = RoboflowVisionEventsBlock(vision_events_port=port).run(
        **arguments(write_to_event_store=True, event_store_url="http://edge:8001")
    )

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "event_id": "local-1",
        "message": "Event written to local event store successfully",
    }
    assert port.calls[0][2] == {
        "api_key": None,
        "write_to_event_store": True,
        "event_store_url": "http://edge:8001",
    }


def test_cloud_key_reaches_the_port_as_a_secret():
    port = Port()

    result = RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented").run(
        **arguments()
    )

    assert result["message"] == "Vision event sent successfully"
    api_key = port.calls[0][2]["api_key"]
    assert isinstance(api_key, Secret) and api_key.reveal() == "invented"


def test_offline_defaults_name_this_family_adapter():
    with pytest.raises(PlatformUnavailableError) as event_error:
        RoboflowVisionEventsBlock(api_key="invented").run(**arguments())

    message = str(event_error.value)
    assert "'vision_events_port'" in message
    assert "RoboflowVisionEventsAdapter()" in message
    assert ADAPTER_MODULE in message
    with pytest.raises(PlatformUnavailableError) as caught:
        VisionEventBundleBlock().run(**arguments(VisionEventBundleBlock))
    assert "'vision_event_bundle_port'" in str(caught.value)
    assert "VisionEventBundleAdapter(allow_access_to_file_system=True)" in str(
        caught.value
    )
    assert "RoboflowPlatformAdapter" not in str(caught.value) + message


@pytest.mark.parametrize(
    "event_type,changes,expected",
    [
        (
            "custom",
            {"custom_value": "", "external_id": "ext"},
            {"value": "", "externalId": "ext"},
        ),
        ("quality_check", {"qc_result": "pass"}, {"result": "pass"}),
        (
            "inventory_count",
            {"item_count": 0, "location": "A", "item_type": "part"},
            {"itemCount": 0, "location": "A", "itemType": "part"},
        ),
        (
            "safety_alert",
            {"alert_type": "spill", "severity": "high", "alert_description": "floor"},
            {"alertType": "spill", "severity": "high", "description": "floor"},
        ),
        (
            "operator_feedback",
            {"related_event_id": "e", "feedback": "correct", "external_id": "ignored"},
            {"relatedEventId": "e", "feedback": "correct"},
        ),
    ],
)
def test_event_data_matches_v1(event_type, changes, expected):
    port = Port()

    RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented").run(
        **arguments(event_type=event_type, **changes)
    )

    request = port.calls[0][1]
    assert request.event_type == event_type and request.event_data == expected


def test_request_carries_images_predictions_metadata_and_use_case():
    port = Port()
    predictions = object()

    RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented").run(
        **arguments(
            input_image=image(10),
            output_image=image(20),
            predictions=predictions,
            custom_metadata={"camera": "north", "count": 2},
        )
    )

    request = port.calls[0][1]
    assert request.input_image.tensor_image.max().item() == 10
    assert request.output_image.tensor_image.max().item() == 20
    assert request.solution == "use-case"
    assert request.custom_metadata == {"camera": "north", "count": 2}
    # Deep-copied: an opaque object is copied, not shared.
    assert (
        type(request.predictions) is object and request.predictions is not predictions
    )


def test_optional_images_and_predictions_are_none():
    port = Port()

    RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented").run(
        **arguments()
    )

    request = port.calls[0][1]
    assert (request.input_image, request.output_image, request.predictions) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    "change", [{"event_type": "unknown"}, {"cooldown_seconds": float("nan")}]
)
def test_invalid_selector_values_fail_before_any_call(change):
    port = Port()

    with pytest.raises(ValueError):
        RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented").run(
            **arguments(**change)
        )

    assert not port.calls


def test_use_case_is_required_for_events_and_optional_for_bundles():
    with pytest.raises(ValidationError):
        RoboflowVisionEventsBlock.Params(event_type="custom")
    params = VisionEventBundleBlock.Params(event_type="custom", target_directory="/b")
    assert params.solution is None


def test_background_outcome_is_v1_admission_and_inputs_are_detached():
    port = Port()
    source, metadata = image(50), {"nested": [1]}
    effects, release = held_effects(max_pending=2)
    with effects:
        try:
            result = RoboflowVisionEventsBlock(
                vision_events_port=port, api_key="invented", sink_effects=effects
            ).run(
                **arguments(
                    input_image=source, custom_metadata=metadata, fire_and_forget=True
                )
            )
            source.tensor_image.zero_()
            metadata["nested"].append(2)
        finally:
            release.set()

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "event_id": "",
        "message": "Vision event sent in background task",
    }
    request = port.calls[0][1]
    assert request.input_image.tensor_image.max().item() == 50
    assert request.input_image.device.type == "cpu"
    assert request.custom_metadata == {"nested": [1]}


@pytest.mark.parametrize("local", [False, True])
def test_queued_event_releases_original_payloads_before_delivery(local):
    class Payload(dict):
        """A weak-referenceable input container with normal deepcopy behavior."""

    class DelayedEffects:
        """Keep the effect queued until the test explicitly delivers it."""

        def __init__(self):
            self.pending = []

        def deliver(self, effect, *, label):
            assert label == RoboflowVisionEventsBlock.type
            self.pending.append(effect)
            return False, BACKGROUND_MESSAGE

    port, effects = Port(), DelayedEffects()
    input_image, output_image = image(10), image(20)
    predictions = Payload(scores=torch.tensor([0.75]))
    metadata = Payload(nested=[1])
    originals = {
        "input image": weakref.ref(input_image),
        "input pixels": weakref.ref(input_image.tensor_image),
        "output image": weakref.ref(output_image),
        "output pixels": weakref.ref(output_image.tensor_image),
        "predictions": weakref.ref(predictions),
        "prediction scores": weakref.ref(predictions["scores"]),
        "metadata": weakref.ref(metadata),
    }
    values = arguments(
        input_image=input_image,
        output_image=output_image,
        predictions=predictions,
        custom_metadata=metadata,
        fire_and_forget=True,
        write_to_event_store=local,
        event_store_url="http://edge:8001/events",
    )
    block = RoboflowVisionEventsBlock(
        vision_events_port=port, api_key="invented", sink_effects=effects
    )

    result = block.run(**values)

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "event_id": "",
        "message": "Vision event sent in background task",
    }
    assert len(effects.pending) == 1 and not port.calls
    input_image.tensor_image.zero_()
    output_image.tensor_image.zero_()
    predictions["scores"].zero_()
    metadata["nested"].append(2)
    values["event_store_url"] = "http://changed:9000"
    del values, input_image, output_image, predictions, metadata
    gc.collect()
    retained = [name for name, reference in originals.items() if reference() is not None]
    assert retained == [], f"queued effect retained original payloads: {retained}"

    outcome = effects.pending.pop()()

    expected_message = (
        "Event written to local event store successfully"
        if local else "Vision event sent successfully"
    )
    assert outcome == (False, expected_message)
    (call,) = port.calls
    operation, request, delivery = call
    assert operation == "submit"
    assert request.input_image.tensor_image.unique().tolist() == [10]
    assert request.output_image.tensor_image.unique().tolist() == [20]
    assert request.input_image.device.type == request.output_image.device.type == "cpu"
    assert request.predictions["scores"].tolist() == [0.75]
    assert request.custom_metadata == {"nested": [1]}
    assert request.event_data == {"value": "alert"} and request.solution == "use-case"
    assert delivery["event_store_url"] == "http://edge:8001/events"
    assert delivery["write_to_event_store"] is local
    if local:
        assert delivery["api_key"] is None
    else:
        assert isinstance(delivery["api_key"], Secret)
        assert delivery["api_key"].reveal() == "invented"


def test_background_failure_reaches_effects_not_the_step():
    port = Port(PlatformRequestError("vision event failed", error_type="x"))
    with BackgroundEffects() as effects:
        result = RoboflowVisionEventsBlock(
            vision_events_port=port, api_key="invented", sink_effects=effects
        ).run(**arguments(fire_and_forget=True))

    assert result["error_status"] is False and result["event_id"] == ""
    (failure,) = effects.failures()
    assert failure.label == RoboflowVisionEventsBlock.type
    assert failure.message == "PlatformRequestError: vision event failed"


def test_fire_and_forget_without_sink_effects_runs_now():
    port = Port()

    result = RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented").run(
        **arguments(fire_and_forget=True)
    )

    assert result["event_id"] == "event-1" and len(port.calls) == 1


def test_inline_failure_is_an_error_outcome_without_an_id():
    port = Port(PlatformRequestError("vision event submission failed", error_type="x"))

    result = RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented").run(
        **arguments()
    )

    assert result == {
        "error_status": True,
        "throttling_status": False,
        "event_id": "",
        "message": "PlatformRequestError: vision event submission failed",
    }


@pytest.mark.parametrize("returned", [None, 3, {"id": "x"}])
def test_non_string_event_id_is_an_error_outcome(returned):
    result = RoboflowVisionEventsBlock(
        vision_events_port=Port(returned), api_key="invented"
    ).run(**arguments())

    assert result["error_status"] is True and result["event_id"] == ""
    assert "non-str event id" in result["message"]


def test_local_store_without_id_is_still_created_as_in_v1():
    result = RoboflowVisionEventsBlock(vision_events_port=Port("")).run(
        **arguments(write_to_event_store=True)
    )

    assert result["error_status"] is False and result["event_id"] == ""


def test_cooldown_throttles_without_preparing_and_ignores_failures():
    port = Port(PlatformRequestError("failed", error_type="x"))
    block = RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented")

    first = block.run(**arguments(cooldown_seconds=100))
    second = block.run(**arguments(cooldown_seconds=100, event_type="unknown"))

    # As in V1 a failed delivery still starts the window; the throttled call
    # is not prepared, so its invalid event_type is never seen.
    assert first["error_status"] is True and len(port.calls) == 1
    assert second == {
        "error_status": False,
        "throttling_status": True,
        "event_id": "",
        "message": "Sink cooldown applies",
    }


def test_negative_cooldown_from_a_selector_disables_throttling():
    port = Port()
    block = RoboflowVisionEventsBlock(vision_events_port=port, api_key="invented")

    for _ in range(2):
        block.run(**arguments(cooldown_seconds=-5))

    assert len(port.calls) == 2


def test_full_effects_reject_starts_the_window_as_v1_dispatch_did():
    port = Port()
    effects, release = held_effects(max_pending=1, when_full="reject")
    block = RoboflowVisionEventsBlock(
        vision_events_port=port, api_key="invented", sink_effects=effects
    )
    with effects:
        try:
            refused = block.run(**arguments(fire_and_forget=True, cooldown_seconds=100))
            throttled = block.run(**arguments(cooldown_seconds=100))
        finally:
            release.set()

    assert refused["error_status"] is True and refused["message"] == FULL_MESSAGE
    assert throttled["throttling_status"] is True and not port.calls


def test_full_effects_run_inline_return_the_created_id():
    port = Port()
    effects, release = held_effects(max_pending=1, when_full="run_inline")
    with effects:
        try:
            result = RoboflowVisionEventsBlock(
                vision_events_port=port, api_key="invented", sink_effects=effects
            ).run(**arguments(fire_and_forget=True))
        finally:
            release.set()

    assert result["event_id"] == "event-1"
    assert result["message"] == "Vision event sent successfully"


def test_bundle_now_returns_id_and_path():
    port = bundle_port()

    result = VisionEventBundleBlock(vision_event_bundle_port=port).run(
        **arguments(VisionEventBundleBlock, solution=None, file_name="b")
    )

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "event_id": "event-2",
        "bundle_path": "/bundles/b.tar.gz",
        "message": "Vision event bundle written successfully",
    }
    destination = {"target_directory": "/bundles", "file_name": "b"}
    assert port.calls[0] == ("prepare", destination)
    assert port.calls[1][0] == "bundle" and port.calls[1][2] == destination
    assert port.calls[1][1].solution is None


def test_bundle_background_admission_and_failure():
    port = bundle_port()
    port.result = OSError("disk full")
    with BackgroundEffects() as effects:
        result = VisionEventBundleBlock(
            vision_event_bundle_port=port, sink_effects=effects
        ).run(**arguments(VisionEventBundleBlock, fire_and_forget=True))

    assert result == {
        "error_status": False,
        "throttling_status": False,
        "event_id": "",
        "bundle_path": "",
        "message": "Vision event bundle written in background task",
    }
    assert [failure.message for failure in effects.failures()] == ["OSError: disk full"]


def test_refused_destination_fails_the_run_even_in_background():
    class Refusing(Port):
        refuse = True

        def prepare_destination(self, **kwargs):
            if self.refuse:
                raise ValueError("outside the allowed write directory")
            super().prepare_destination(**kwargs)

    port = Refusing(EventBundle(event_id="event-3", bundle_path="/bundles/c"))
    with BackgroundEffects() as effects:
        block = VisionEventBundleBlock(
            vision_event_bundle_port=port, sink_effects=effects
        )
        with pytest.raises(ValueError, match="allowed write directory"):
            block.run(
                **arguments(
                    VisionEventBundleBlock, fire_and_forget=True, cooldown_seconds=100
                )
            )
        assert not port.calls and effects.stats().submitted == 0

        # The refusal started no cooldown window: a corrected call runs.
        port.refuse = False
        result = block.run(**arguments(VisionEventBundleBlock, cooldown_seconds=100))

    assert result["bundle_path"] == "/bundles/c"


def test_bundle_port_without_event_bundle_result_is_an_error_outcome():
    result = VisionEventBundleBlock(
        vision_event_bundle_port=Port({"event_id": "x", "bundle_path": "y"})
    ).run(**arguments(VisionEventBundleBlock))

    assert result["error_status"] is True and result["bundle_path"] == ""
    assert "no EventBundle" in result["message"]


@pytest.fixture
def no_event_work(monkeypatch):
    """Fail if the block snapshots inputs or starts a delivery window."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("snapshot or cooldown start")
        raise AssertionError("missing-port/no-work path prepared a delivery")

    monkeypatch.setattr(events, "_detached_request", forbidden)
    return calls, forbidden


@pytest.mark.parametrize("runner", [False, True])
@pytest.mark.parametrize("fire", [False, True])
@pytest.mark.parametrize("local", [False, True])
def test_missing_default_event_port_fails_before_work(runner, fire, local, no_event_work):
    effects = BackgroundEffects() if runner else None
    block = RoboflowVisionEventsBlock(api_key="invented", sink_effects=effects)
    calls, forbidden = no_event_work
    block._cooldown.try_start = forbidden
    before = block._cooldown._last

    with effects if effects is not None else nullcontext():
        with pytest.raises(PlatformUnavailableError) as caught:
            block.run(**arguments(
                fire_and_forget=fire, write_to_event_store=local, cooldown_seconds=100
            ))

    assert "'vision_events_port'" in str(caught.value)
    assert "RoboflowVisionEventsAdapter()" in str(caught.value)
    assert ADAPTER_MODULE in str(caught.value)
    assert not calls and block._cooldown._last == before
    if effects is not None:
        assert effects.stats().submitted == 0 and not effects.failures()


@pytest.mark.parametrize("runner", [False, True])
@pytest.mark.parametrize("fire", [False, True])
@pytest.mark.parametrize("policy,step", [(True, False), (False, True)])
def test_disabled_default_event_port_skips_before_validation(
    runner, fire, policy, step, no_event_work
):
    effects = BackgroundEffects() if runner else None
    block = RoboflowVisionEventsBlock(disable_sinks=policy, sink_effects=effects)
    calls, forbidden = no_event_work
    block._cooldown.try_start = forbidden
    with effects if effects is not None else nullcontext():
        result = block.run(**arguments(
            fire_and_forget=fire, disable_sink=step, event_type="unknown",
            cooldown_seconds=float("nan")
        ))
    assert result["error_status"] is False and result["event_id"] == ""
    assert result["throttling_status"] is False
    assert not calls and block._cooldown._last is None
    if effects is not None:
        assert effects.stats().submitted == 0 and not effects.failures()


@pytest.mark.parametrize("runner", [False, True])
@pytest.mark.parametrize("fire", [False, True])
def test_cooling_default_event_port_skips_validation_and_work(runner, fire, no_event_work):
    effects = BackgroundEffects() if runner else None
    block = RoboflowVisionEventsBlock(sink_effects=effects)
    block._cooldown._clock = lambda: 10.0
    assert block._cooldown.try_start(100)
    before = block._cooldown._last
    calls, forbidden = no_event_work
    block._cooldown.try_start = forbidden
    with effects if effects is not None else nullcontext():
        result = block.run(**arguments(
            fire_and_forget=fire, event_type="unknown", cooldown_seconds=100
        ))
    assert result == {
        "error_status": False, "throttling_status": True,
        "event_id": "", "message": "Sink cooldown applies",
    }
    assert not calls and block._cooldown._last == before
    if effects is not None:
        assert effects.stats().submitted == 0 and not effects.failures()


@pytest.mark.parametrize("local,key,changes,error,match", [
    (False, "invented", {"event_type": "unknown"}, ValueError, "Unknown event_type"),
    (True, None, {"event_type": "unknown"}, ValueError, "Unknown event_type"),
    (False, "invented", {"cooldown_seconds": float("nan")}, ValueError, "NaN"),
    (False, None, {}, ValueError, "API key"),
    (False, None, {"event_type": "unknown"}, ValueError, "API key"),
    (False, Secret.unavailable("key"), {}, SecretUnavailableError, "key"),
])
def test_missing_default_event_port_preserves_validation_order(
    local, key, changes, error, match, no_event_work
):
    with BackgroundEffects() as effects:
        block = RoboflowVisionEventsBlock(api_key=key, sink_effects=effects)
        calls, forbidden = no_event_work
        block._cooldown.try_start = forbidden
        with pytest.raises(error, match=match):
            block.run(**arguments(
                fire_and_forget=True, write_to_event_store=local, **changes
            ))
        assert not calls and block._cooldown._last is None
        assert effects.stats().submitted == 0 and not effects.failures()


def test_missing_default_event_port_refusal_does_not_start_cooldown():
    block = RoboflowVisionEventsBlock(api_key="invented")
    with pytest.raises(PlatformUnavailableError):
        block.run(**arguments(cooldown_seconds=100))
    port = Port()
    block._port = port
    result = block.run(**arguments(cooldown_seconds=100))
    assert result["event_id"] == "event-1" and len(port.calls) == 1


@pytest.mark.parametrize("runner", [False, True])
@pytest.mark.parametrize("fire", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("failure", [
    PlatformUnavailableError("configured port is unavailable"),
    PlatformRequestError("configured delivery failed", error_type="x"),
])
def test_configured_event_port_keeps_late_failure_semantics(runner, fire, local, failure):
    class ConfiguredPort(Port):
        def ensure_available(self):
            raise AssertionError("custom ports must not be availability-probed")

    port = ConfiguredPort(failure)
    effects = BackgroundEffects() if runner else None
    block = RoboflowVisionEventsBlock(
        vision_events_port=port, api_key="invented", sink_effects=effects
    )
    with effects if effects is not None else nullcontext():
        result = block.run(**arguments(
            fire_and_forget=fire, write_to_event_store=local
        ))
    expected = f"{type(failure).__name__}: {failure}"
    assert len(port.calls) == 1 and result["event_id"] == ""
    if runner and fire:
        assert result["error_status"] is False
        assert result["message"] == "Vision event sent in background task"
        assert [item.message for item in effects.failures()] == [expected]
    else:
        assert result["error_status"] is True and result["message"] == expected
        if effects is not None:
            assert not effects.failures()
