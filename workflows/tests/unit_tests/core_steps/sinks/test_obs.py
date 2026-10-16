import json
import threading
import time
from pathlib import Path as pathlib_Path

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.sinks.obs import client as obs_client
from roboflow_workflows.core_steps.sinks.obs import discovery as obs_discovery
from roboflow_workflows.core_steps.sinks.obs.action.v1 import (
    BlockManifest as ActionManifest,
)
from roboflow_workflows.core_steps.sinks.obs.action.v1 import OBSActionBlockV1
from roboflow_workflows.core_steps.sinks.obs.connection import (
    v1 as obs_connection_module,
)
from roboflow_workflows.core_steps.sinks.obs.connection.v1 import (
    BlockManifest as ConnectionManifest,
)
from roboflow_workflows.core_steps.sinks.obs.connection.v1 import OBSConnectionBlockV1
from roboflow_workflows.core_steps.sinks.obs.websocket_client import OBSRequestError
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.prototypes.block import COOLDOWN_HTTP_SOFT_RESTRICTION, Severity


class FakeOBSClient:
    """Stands in for OBSWebSocketClient, recording each request it receives."""

    RESPONSES = {
        "GetVersion": {"obsVersion": "32.2.2"},
        "GetSceneItemId": {"sceneItemId": 7},
        "GetVideoSettings": {"baseWidth": 1920, "baseHeight": 1080},
    }

    def __init__(self, fail_times: int = 0):
        self.calls = []
        self._fail_times = fail_times
        self.closed = False

    def call(self, request_type, *, data=None):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ConnectionError("websocket closed")
        self.calls.append((request_type, data or {}))
        return dict(self.RESPONSES.get(request_type, {}))

    def close(self):
        self.closed = True


@pytest.fixture
def fake_obs(monkeypatch):
    created = []

    def fake_connect(host, port, password, timeout):
        client = FakeOBSClient()
        created.append(client)
        return client

    obs_client.reset_clients()
    monkeypatch.setattr(obs_client, "_connect", fake_connect)
    yield created
    obs_client.reset_clients()


CONNECTION = {"host": "127.0.0.1", "port": 4455, "password": "secret", "timeout": 3}


def _action_block() -> OBSActionBlockV1:
    return OBSActionBlockV1(
        background_tasks=None, thread_pool_executor=None, disable_sinks=False
    )


def test_connection_block_reports_obs_version_when_verification_succeeds(fake_obs):
    block = OBSConnectionBlockV1()

    result = block.run(
        host="127.0.0.1",
        port=4455,
        password="secret",
        timeout=3,
        discover_password=False,
        verify_connection=True,
    )

    assert result["error_status"] is False
    assert result["obs_version"] == "32.2.2"
    assert result["connection"]["port"] == 4455
    assert fake_obs[0].calls == [("GetVersion", {})]


def test_connection_block_skips_contacting_obs_when_verification_disabled(fake_obs):
    block = OBSConnectionBlockV1()

    result = block.run(
        host="127.0.0.1",
        port=4455,
        password=None,
        timeout=3,
        discover_password=False,
        verify_connection=False,
    )

    assert result["error_status"] is False
    assert result["message"].startswith("Connection not verified")
    assert fake_obs == []


def test_connection_block_reports_error_instead_of_raising_when_obs_unreachable(
    monkeypatch,
):
    obs_client.reset_clients()

    def refuse(host, port, password, timeout):
        raise ConnectionRefusedError("connection refused")

    monkeypatch.setattr(obs_client, "_connect", refuse)
    block = OBSConnectionBlockV1()

    result = block.run(
        host="127.0.0.1",
        port=4455,
        password="secret",
        timeout=3,
        discover_password=False,
        verify_connection=True,
    )

    assert result["error_status"] is True
    assert "Could not connect to OBS" in result["message"]
    assert result["connection"]["host"] == "127.0.0.1"


def test_set_scene_action_switches_program_scene(fake_obs):
    result = _action_block().run(
        connection=CONNECTION,
        action="set_scene",
        scene_name="Detected",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert result["error_status"] is False
    assert fake_obs[0].calls == [("SetCurrentProgramScene", {"sceneName": "Detected"})]


def test_set_text_action_overlays_input_settings(fake_obs):
    result = _action_block().run(
        connection=CONNECTION,
        action="set_text",
        scene_name=None,
        source_name="Counter",
        filter_name=None,
        text="3 people",
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert result["error_status"] is False
    assert fake_obs[0].calls == [
        (
            "SetInputSettings",
            {
                "inputName": "Counter",
                "inputSettings": {"text": "3 people"},
                "overlay": True,
            },
        )
    ]


def test_toggle_filter_action_enables_named_filter(fake_obs):
    _action_block().run(
        connection=CONNECTION,
        action="toggle_filter",
        scene_name=None,
        source_name="Webcam",
        filter_name="Blur",
        text=None,
        enabled=True,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert fake_obs[0].calls == [
        (
            "SetSourceFilterEnabled",
            {"sourceName": "Webcam", "filterName": "Blur", "filterEnabled": True},
        )
    ]


def test_source_visibility_action_resolves_scene_item_id(fake_obs):
    _action_block().run(
        connection=CONNECTION,
        action="set_source_visibility",
        scene_name="Main",
        source_name="Overlay",
        filter_name=None,
        text=None,
        enabled=False,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert fake_obs[0].calls == [
        ("GetSceneItemId", {"sceneName": "Main", "sourceName": "Overlay"}),
        (
            "SetSceneItemEnabled",
            {"sceneName": "Main", "sceneItemId": 7, "sceneItemEnabled": False},
        ),
    ]


def test_action_reconnects_once_when_pooled_socket_is_dead(monkeypatch):
    obs_client.reset_clients()
    created = []

    def fake_connect(host, port, password, timeout):
        # first client fails its single request, the replacement succeeds
        client = FakeOBSClient(fail_times=1 if not created else 0)
        created.append(client)
        return client

    monkeypatch.setattr(obs_client, "_connect", fake_connect)

    result = _action_block().run(
        connection=CONNECTION,
        action="set_scene",
        scene_name="Main",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert result["error_status"] is False
    assert len(created) == 2
    assert created[0].closed is True
    assert created[1].calls == [("SetCurrentProgramScene", {"sceneName": "Main"})]
    obs_client.reset_clients()


def test_action_reports_error_status_when_obs_request_keeps_failing(monkeypatch):
    obs_client.reset_clients()
    monkeypatch.setattr(
        obs_client,
        "_connect",
        lambda host, port, password, timeout: FakeOBSClient(fail_times=5),
    )

    result = _action_block().run(
        connection=CONNECTION,
        action="set_scene",
        scene_name="Main",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert result["error_status"] is True
    assert "failed" in result["message"]
    obs_client.reset_clients()


def test_cooldown_throttles_second_execution(fake_obs):
    block = _action_block()
    kwargs = dict(
        connection=CONNECTION,
        action="set_scene",
        scene_name="Main",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=30,
        fire_and_forget=False,
        disable_sink=False,
    )

    first = block.run(**kwargs)
    second = block.run(**kwargs)

    assert first["throttling_status"] is False
    assert second["throttling_status"] is True
    assert len(fake_obs[0].calls) == 1


def test_disabled_sink_does_not_contact_obs(fake_obs):
    result = _action_block().run(
        connection=CONNECTION,
        action="set_scene",
        scene_name="Main",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=True,
    )

    assert result["error_status"] is False
    assert fake_obs == []


def test_manifest_rejects_action_missing_required_field():
    with pytest.raises(ValueError) as error:
        ActionManifest.model_validate(
            {
                "type": "roboflow_core/obs_action@v1",
                "name": "obs",
                "connection": "$steps.obs_connection.connection",
                "action": "toggle_filter",
                "source_name": "Webcam",
            }
        )

    assert "filter_name" in str(error.value)
    assert "enabled" in str(error.value)


def test_manifest_accepts_action_with_required_fields():
    manifest = ActionManifest.model_validate(
        {
            "type": "roboflow_core/obs_action@v1",
            "name": "obs",
            "connection": "$steps.obs_connection.connection",
            "action": "set_scene",
            "scene_name": "Detected",
        }
    )

    assert manifest.action == "set_scene"
    assert manifest.scene_name == "Detected"


def test_connection_manifest_defaults_to_local_obs():
    manifest = ConnectionManifest.model_validate(
        {"type": "roboflow_core/obs_connection@v1", "name": "obs_connection"}
    )

    assert manifest.host == "127.0.0.1"
    assert manifest.port == 4455


def test_application_level_obs_error_is_not_retried(monkeypatch):
    """A rejected request (bad scene name) must fail fast, not churn the connection."""
    obs_client.reset_clients()
    created = []

    class RejectingClient(FakeOBSClient):
        def call(self, request_type, *, data=None):
            raise OBSRequestError(request_type, code=600, comment="no such scene")

    def fake_connect(host, port, password, timeout):
        client = RejectingClient()
        created.append(client)
        return client

    monkeypatch.setattr(obs_client, "_connect", fake_connect)

    result = _action_block().run(
        connection=CONNECTION,
        action="set_scene",
        scene_name="NoSuchScene",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert result["error_status"] is True
    assert len(created) == 1, "application-level errors must not trigger a reconnect"
    obs_client.reset_clients()


# --- password discovery ------------------------------------------------------


def _write_obs_config(tmp_path, **overrides):
    config = {
        "alerts_enabled": False,
        "auth_required": True,
        "first_load": False,
        "server_enabled": True,
        "server_password": "from-obs-config",
        "server_port": 4455,
    }
    config.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_discover_password_reads_local_obs_config(tmp_path, monkeypatch):
    path = _write_obs_config(tmp_path)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [path])

    discovered = obs_discovery.discover_password()

    assert discovered.password == "from-obs-config"
    assert discovered.source == path
    assert discovered.auth_required is True
    assert discovered.server_enabled is True


def test_discover_password_returns_none_when_no_config_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(
        obs_discovery, "candidate_config_paths", lambda: [tmp_path / "missing.json"]
    )

    assert obs_discovery.discover_password() is None


def test_discover_password_skips_malformed_config(tmp_path, monkeypatch):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    good = _write_obs_config(tmp_path)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [broken, good])

    assert obs_discovery.discover_password().password == "from-obs-config"


def test_is_local_host_recognises_loopback_forms():
    assert obs_discovery.is_local_host("127.0.0.1")
    assert obs_discovery.is_local_host("LOCALHOST")
    assert not obs_discovery.is_local_host("192.168.1.50")


def _resolve(host="127.0.0.1", password=None, allow_discovery=True):
    return OBSConnectionBlockV1._resolve_password(
        host=host, password=password, allow_discovery=allow_discovery
    )


def test_supplied_password_is_never_replaced_by_discovery(monkeypatch):
    monkeypatch.setattr(
        obs_discovery,
        "discover_password",
        lambda: (_ for _ in ()).throw(AssertionError("discovery must not run")),
    )

    resolved, note = _resolve(password="explicit")

    assert resolved == "explicit"
    assert note == "password supplied by Workflow"


def test_password_discovered_when_absent_and_host_is_local(tmp_path, monkeypatch):
    path = _write_obs_config(tmp_path)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [path])

    resolved, note = _resolve()

    assert resolved == "from-obs-config"
    assert str(path) in note
    assert "from-obs-config" not in note, "the password itself must not be in the note"


def test_remote_host_does_not_read_this_machines_config(tmp_path, monkeypatch):
    path = _write_obs_config(tmp_path)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [path])

    resolved, note = _resolve(host="192.168.1.50")

    assert resolved is None
    assert "not local" in note


def test_discovery_never_reads_disk_when_local_storage_access_is_off(
    tmp_path, monkeypatch
):
    path = _write_obs_config(tmp_path)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [path])
    monkeypatch.setattr(
        obs_connection_module, "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE", False
    )
    reads = []
    monkeypatch.setattr(
        obs_discovery, "discover_password", lambda: reads.append(1) or None
    )

    resolved, note = _resolve(host="127.0.0.1", password=None, allow_discovery=True)

    assert resolved is None
    assert "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE" in note
    assert reads == [], "the OBS config must not be read when the flag is off"


def test_discovery_can_be_switched_off(tmp_path, monkeypatch):
    path = _write_obs_config(tmp_path)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [path])

    resolved, note = _resolve(allow_discovery=False)

    assert resolved is None
    assert "discovery disabled" in note


def test_no_password_sent_when_obs_has_auth_disabled(tmp_path, monkeypatch):
    path = _write_obs_config(tmp_path, auth_required=False)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [path])

    resolved, note = _resolve()

    assert resolved is None
    assert "authentication disabled" in note


def test_note_warns_when_discovered_config_has_server_disabled(tmp_path, monkeypatch):
    path = _write_obs_config(tmp_path, server_enabled=False)
    monkeypatch.setattr(obs_discovery, "candidate_config_paths", lambda: [path])

    _, note = _resolve()

    assert "websocket server is disabled" in note


def test_execution_policy_disable_is_reported_distinctly_from_block_parameter(fake_obs):
    """`disable_sinks` from the runtime must not read as if the user set `disable_sink`."""
    policy_blocked = OBSActionBlockV1(
        background_tasks=None, thread_pool_executor=None, disable_sinks=True
    )
    kwargs = dict(
        connection=CONNECTION,
        action="set_scene",
        scene_name="Dog",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    by_policy = policy_blocked.run(**kwargs)
    by_parameter = _action_block().run(**{**kwargs, "disable_sink": True})

    assert by_policy["message"] == "Sink was disabled by workflow execution policy"
    assert by_parameter["message"] == "Sink was disabled by parameter `disable_sink`"
    assert fake_obs == []


# --- source transform actions ------------------------------------------------


@pytest.fixture
def fake_transform_obs(monkeypatch):
    created = []

    def fake_connect(host, port, password, timeout):
        client = FakeOBSClient()
        created.append(client)
        return client

    obs_client.reset_clients()
    monkeypatch.setattr(obs_client, "_connect", fake_connect)
    yield created
    obs_client.reset_clients()


def _image(width=640, height=360):
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _detections(*boxes_with_confidence):
    boxes = np.array([b[:4] for b in boxes_with_confidence], dtype=np.float64)
    confidence = np.array([b[4] for b in boxes_with_confidence], dtype=np.float64)
    return sv.Detections(xyxy=boxes, confidence=confidence)


def _move_kwargs(**overrides):
    kwargs = dict(
        connection=CONNECTION,
        action="move_source_to_detection",
        scene_name="Apple",
        source_name="Apple GIF",
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
        predictions=_detections((64, 36, 128, 108, 0.9)),
        image=_image(),
    )
    kwargs.update(overrides)
    return kwargs


def test_move_source_maps_image_coordinates_onto_obs_canvas(fake_transform_obs):
    # image 640x360 -> canvas 1920x1080 is a 3x scale in both axes
    result = _action_block().run(**_move_kwargs())

    assert result["error_status"] is False
    transform_call = [
        c for c in fake_transform_obs[0].calls if c[0] == "SetSceneItemTransform"
    ][0]
    request = transform_call[1]
    assert request["sceneName"] == "Apple" and request["sceneItemId"] == 7
    transform = request["sceneItemTransform"]
    assert transform["positionX"] == pytest.approx(192.0)
    assert transform["positionY"] == pytest.approx(108.0)
    assert transform["boundsWidth"] == pytest.approx(192.0)
    assert transform["boundsHeight"] == pytest.approx(216.0)
    assert transform["boundsType"] == "OBS_BOUNDS_SCALE_INNER"
    assert (
        "SetSceneItemEnabled",
        {"sceneName": "Apple", "sceneItemId": 7, "sceneItemEnabled": True},
    ) in fake_transform_obs[0].calls


def test_move_source_follows_highest_confidence_detection(fake_transform_obs):
    predictions = _detections((0, 0, 10, 10, 0.3), (320, 180, 480, 270, 0.95))

    _action_block().run(**_move_kwargs(predictions=predictions))

    transform = [
        c for c in fake_transform_obs[0].calls if c[0] == "SetSceneItemTransform"
    ][0][1]["sceneItemTransform"]
    assert transform["positionX"] == pytest.approx(960.0)
    assert transform["positionY"] == pytest.approx(540.0)


def test_move_source_hides_source_when_no_detections(fake_transform_obs):
    result = _action_block().run(**_move_kwargs(predictions=sv.Detections.empty()))

    assert result["error_status"] is False
    assert "hid source" in result["message"]
    assert (
        "SetSceneItemEnabled",
        {"sceneName": "Apple", "sceneItemId": 7, "sceneItemEnabled": False},
    ) in fake_transform_obs[0].calls
    assert not any(c[0] == "SetSceneItemTransform" for c in fake_transform_obs[0].calls)


def test_move_source_leaves_source_alone_when_hide_disabled(fake_transform_obs):
    result = _action_block().run(
        **_move_kwargs(predictions=sv.Detections.empty(), hide_when_empty=False)
    )

    assert result["error_status"] is False
    assert "left unchanged" in result["message"]
    assert not any(
        c[0] in ("SetSceneItemEnabled", "SetSceneItemTransform")
        for c in fake_transform_obs[0].calls
    )


def test_transform_lookups_are_cached_across_calls(fake_transform_obs):
    block = _action_block()

    block.run(**_move_kwargs())
    block.run(**_move_kwargs())

    calls = fake_transform_obs[0].calls
    assert sum(1 for c in calls if c[0] == "GetSceneItemId") == 1
    assert sum(1 for c in calls if c[0] == "GetVideoSettings") == 1
    assert sum(1 for c in calls if c[0] == "SetSceneItemTransform") == 2


def test_stretch_fit_uses_stretch_bounds(fake_transform_obs):
    _action_block().run(**_move_kwargs(fit="stretch"))

    transform = [
        c for c in fake_transform_obs[0].calls if c[0] == "SetSceneItemTransform"
    ][0][1]["sceneItemTransform"]
    assert transform["boundsType"] == "OBS_BOUNDS_STRETCH"


def test_set_source_transform_places_source_at_explicit_coordinates(fake_transform_obs):
    result = _action_block().run(
        **_move_kwargs(
            action="set_source_transform",
            predictions=None,
            image=None,
            position_x=100.0,
            position_y=200.0,
            width=300.0,
            height=400.0,
        )
    )

    assert result["error_status"] is False
    transform = [
        c for c in fake_transform_obs[0].calls if c[0] == "SetSceneItemTransform"
    ][0][1]["sceneItemTransform"]
    assert transform["positionX"] == pytest.approx(100.0)
    assert transform["boundsWidth"] == pytest.approx(300.0)


def test_manifest_rejects_move_action_without_predictions_and_image():
    with pytest.raises(ValueError) as error:
        ActionManifest.model_validate(
            {
                "type": "roboflow_core/obs_action@v1",
                "name": "obs",
                "connection": "$steps.obs.connection",
                "action": "move_source_to_detection",
                "scene_name": "Apple",
                "source_name": "Apple GIF",
            }
        )

    assert "predictions" in str(error.value)
    assert "image" in str(error.value)


def test_move_source_offset_places_source_beside_detection(fake_transform_obs):
    # detection maps to canvas x=192 w=192; offset_x=-1 puts the source fully left of it
    _action_block().run(**_move_kwargs(offset_x=-1.0))

    transform = [
        c for c in fake_transform_obs[0].calls if c[0] == "SetSceneItemTransform"
    ][0][1]["sceneItemTransform"]
    assert transform["positionX"] == pytest.approx(0.0)
    assert transform["boundsWidth"] == pytest.approx(192.0)


def test_move_source_offset_right_and_vertical(fake_transform_obs):
    _action_block().run(**_move_kwargs(offset_x=1.0, offset_y=-0.5))

    transform = [
        c for c in fake_transform_obs[0].calls if c[0] == "SetSceneItemTransform"
    ][0][1]["sceneItemTransform"]
    assert transform["positionX"] == pytest.approx(384.0)  # 192 + 192
    assert transform["positionY"] == pytest.approx(0.0)  # 108 - 0.5*216


# --- review fixes: credential registry, locking, per-instance caching ---------


def test_connection_descriptor_never_carries_the_password(fake_obs):
    result = OBSConnectionBlockV1().run(
        host="127.0.0.1",
        port=4455,
        password="secret",
        timeout=3,
        discover_password=False,
        verify_connection=False,
    )

    assert "password" not in result["connection"]
    assert set(result["connection"]) == {"host", "port", "timeout"}


def test_action_authenticates_from_registry_when_descriptor_has_no_password(
    monkeypatch,
):
    obs_client.reset_clients()
    seen = {}

    def fake_connect(host, port, password, timeout):
        seen["password"] = password
        seen["timeout"] = timeout
        return FakeOBSClient()

    monkeypatch.setattr(obs_client, "_connect", fake_connect)
    OBSConnectionBlockV1().run(
        host="127.0.0.1",
        port=4455,
        password="secret",
        timeout=7,
        discover_password=False,
        verify_connection=False,
    )

    result = _action_block().run(
        connection={"host": "127.0.0.1", "port": 4455, "timeout": 7},
        action="set_scene",
        scene_name="Dog",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )

    assert result["error_status"] is False
    assert seen == {"password": "secret", "timeout": 7}
    obs_client.reset_clients()


def test_reregistering_a_new_password_drops_the_pooled_client(fake_obs):
    OBSConnectionBlockV1().run(
        host="127.0.0.1",
        port=4455,
        password="old",
        timeout=3,
        discover_password=False,
        verify_connection=True,
    )
    OBSConnectionBlockV1().run(
        host="127.0.0.1",
        port=4455,
        password="new",
        timeout=3,
        discover_password=False,
        verify_connection=True,
    )

    # first client closed, second created with the new credential
    assert fake_obs[0].closed is True
    assert len(fake_obs) == 2


def test_requests_on_one_connection_are_serialised(monkeypatch):
    """The websocket client is not thread-safe; concurrent callers must not overlap."""
    obs_client.reset_clients()
    monkeypatch.setattr(
        obs_client, "_connect", lambda host, port, password, timeout: FakeOBSClient()
    )
    in_flight, peak, guard = [0], [0], threading.Lock()

    def slow_operation(client):
        with guard:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.02)
        with guard:
            in_flight[0] -= 1

    threads = [
        threading.Thread(
            target=obs_client.call_with_reconnect,
            kwargs=dict(
                host="127.0.0.1",
                port=4455,
                operation=slow_operation,
                password="x",
                timeout=1,
            ),
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak[0] == 1
    obs_client.reset_clients()


def test_connection_verifies_once_per_instance_not_per_frame(fake_obs):
    block = OBSConnectionBlockV1()
    kwargs = dict(
        host="127.0.0.1",
        port=4455,
        password="secret",
        timeout=3,
        discover_password=False,
        verify_connection=True,
    )

    first = block.run(**kwargs)
    second = block.run(**kwargs)

    assert first["obs_version"] == second["obs_version"] == "32.2.2"
    assert sum(1 for c in fake_obs[0].calls if c[0] == "GetVersion") == 1
    assert "verified earlier" in second["message"]


def test_password_discovery_runs_once_per_instance(fake_obs, monkeypatch):
    calls = [0]

    def counting_discovery():
        calls[0] += 1
        return obs_discovery.DiscoveredPassword(
            password="from-config",
            source=pathlib_Path("/x/config.json"),
            auth_required=True,
            server_enabled=True,
        )

    monkeypatch.setattr(
        "roboflow_workflows.core_steps.sinks.obs.connection.v1.discover_password",
        counting_discovery,
    )
    block = OBSConnectionBlockV1()
    kwargs = dict(
        host="127.0.0.1",
        port=4455,
        password=None,
        timeout=3,
        discover_password=True,
        verify_connection=False,
    )

    block.run(**kwargs)
    block.run(**kwargs)

    assert calls[0] == 1


def test_connection_block_honours_disable_sinks(fake_obs):
    block = OBSConnectionBlockV1(disable_sinks=True)

    result = block.run(
        host="127.0.0.1",
        port=4455,
        password="secret",
        timeout=3,
        discover_password=False,
        verify_connection=True,
    )

    assert result["error_status"] is False
    assert "execution policy" in result["message"]
    assert fake_obs == []  # never opened a socket


class ImmediateExecutor:
    """Runs submitted work inline so a background failure surfaces in the test."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


def test_fire_and_forget_failure_is_logged_not_raised(monkeypatch, caplog):
    obs_client.reset_clients()

    def refuse(host, port, password, timeout):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(obs_client, "_connect", refuse)
    block = OBSActionBlockV1(
        background_tasks=None,
        thread_pool_executor=ImmediateExecutor(),
        disable_sinks=False,
    )

    with caplog.at_level("WARNING"):
        result = block.run(
            connection=CONNECTION,
            action="set_scene",
            scene_name="Dog",
            source_name=None,
            filter_name=None,
            text=None,
            enabled=None,
            hotkey_name=None,
            cooldown_seconds=0,
            fire_and_forget=True,
            disable_sink=False,
        )

    assert result["error_status"] is False  # background: result is not awaited
    assert any("Background OBS action" in r.message for r in caplog.records)
    obs_client.reset_clients()


def test_action_restrictions_cover_lan_access_and_http_cooldown():
    restrictions = ActionManifest.get_restrictions()

    assert COOLDOWN_HTTP_SOFT_RESTRICTION in restrictions
    lan = [r for r in restrictions if r.code == "requires_lan_access_to_device"]
    assert lan and lan[0].severity is Severity.HARD


def test_obs_blocks_do_not_restrict_step_execution_mode():
    # OBS requests go out from the process running the Workflow whatever the step
    # execution mode, as with the ONVIF and PLC blocks; only network reach matters
    action = ActionManifest.model_validate(
        {
            "type": "roboflow_core/obs_action@v1",
            "name": "obs",
            "connection": "$steps.obs_connection.connection",
            "action": "set_scene",
            "scene_name": "Detected",
        }
    )
    connection = ConnectionManifest.model_validate(
        {"type": "roboflow_core/obs_connection@v1", "name": "obs_connection"}
    )

    for manifest in (action, connection):
        declared = manifest.get_actual_restrictions().items
        assert [r.code for r in declared if r.severity is Severity.HARD] == [
            "requires_lan_access_to_device"
        ]
        assert not [r for r in declared if r.applies_to_step_execution_modes]


# --- runtime skip, unchanged dedup, available-targets hint ---------------------


def _scene_kwargs(**overrides):
    kwargs = dict(
        connection=CONNECTION,
        action="set_scene",
        scene_name="Dog",
        source_name=None,
        filter_name=None,
        text=None,
        enabled=None,
        hotkey_name=None,
        cooldown_seconds=0,
        fire_and_forget=False,
        disable_sink=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_action_skips_when_target_resolves_to_none_at_runtime(fake_obs):
    # a router that emitted nothing this frame -> hold state, not an error
    result = _action_block().run(**_scene_kwargs(scene_name=None))

    assert result["error_status"] is False
    assert "skipped: no value for scene_name" in result["message"]
    assert fake_obs == []


def test_unchanged_scene_is_not_resent(fake_obs):
    block = _action_block()

    first = block.run(**_scene_kwargs())
    second = block.run(**_scene_kwargs())

    assert "Switched" in first["message"]
    assert "unchanged; skipped" in second["message"]
    assert fake_obs[0].calls == [("SetCurrentProgramScene", {"sceneName": "Dog"})]


def test_changed_scene_is_sent_and_switch_back_is_sent_again(fake_obs):
    block = _action_block()

    block.run(**_scene_kwargs(scene_name="Dog"))
    block.run(**_scene_kwargs(scene_name="Cat"))
    block.run(**_scene_kwargs(scene_name="Dog"))

    assert [c[1]["sceneName"] for c in fake_obs[0].calls] == ["Dog", "Cat", "Dog"]


def test_skip_if_unchanged_can_be_disabled(fake_obs):
    block = _action_block()

    block.run(**_scene_kwargs(skip_if_unchanged=False))
    block.run(**_scene_kwargs(skip_if_unchanged=False))

    assert len(fake_obs[0].calls) == 2


def test_failed_apply_is_retried_next_time_not_deduplicated(monkeypatch):
    obs_client.reset_clients()
    attempts = []

    class FlakyClient(FakeOBSClient):
        def call(self, request_type, *, data=None):
            if request_type == "GetSceneList":
                return {"scenes": [{"sceneName": "Cat"}, {"sceneName": "Dog"}]}
            attempts.append(data["sceneName"])
            if len(attempts) == 1:
                raise OBSRequestError(
                    request_type, code=600, comment="No source was found"
                )
            return super().call(request_type, data=data)

    monkeypatch.setattr(
        obs_client, "_connect", lambda host, port, password, timeout: FlakyClient()
    )
    block = _action_block()

    first = block.run(**_scene_kwargs(scene_name="Dog"))
    second = block.run(**_scene_kwargs(scene_name="Dog"))

    assert first["error_status"] is True
    assert second["error_status"] is False and "Switched" in second["message"]
    assert attempts == ["Dog", "Dog"]
    obs_client.reset_clients()


def test_unknown_scene_error_lists_available_scenes(monkeypatch):
    obs_client.reset_clients()

    class RejectingClient(FakeOBSClient):
        def call(self, request_type, *, data=None):
            if request_type == "GetSceneList":
                return {"scenes": [{"sceneName": "Cat"}, {"sceneName": "Coffee"}]}
            raise OBSRequestError(
                request_type,
                code=600,
                comment="No source was found by the name of `Coffe`",
            )

    monkeypatch.setattr(
        obs_client, "_connect", lambda host, port, password, timeout: RejectingClient()
    )

    result = _action_block().run(**_scene_kwargs(scene_name="Coffe"))

    assert result["error_status"] is True
    assert "Available scenes: Cat, Coffee" in result["message"]
    obs_client.reset_clients()


# --- Detections Class Router -----------------------------------------------------


def _tensor_detections(class_names, confidences, boxes=None):
    """inference_models.Detections as produced with ENABLE_TENSOR_DATA_REPRESENTATION."""
    torch = pytest.importorskip("torch")
    from inference_models.models.base.object_detection import Detections

    lookup = {i: n for i, n in enumerate(dict.fromkeys(class_names))}
    inverse = {n: i for i, n in lookup.items()}
    boxes = boxes or [(0, 0, 10, 10)] * len(class_names)
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor([inverse[n] for n in class_names], dtype=torch.long),
        confidence=torch.tensor(confidences, dtype=torch.float32),
        image_metadata={"class_names": lookup},
    )


def test_move_source_follows_tensor_native_detections(fake_transform_obs):
    predictions = _tensor_detections(
        ["apple", "apple"], [0.3, 0.95], boxes=[(0, 0, 10, 10), (320, 180, 480, 270)]
    )

    result = _action_block().run(**_move_kwargs(predictions=predictions))

    assert result["error_status"] is False
    transform = [
        c for c in fake_transform_obs[0].calls if c[0] == "SetSceneItemTransform"
    ][0][1]["sceneItemTransform"]
    assert transform["positionX"] == pytest.approx(960.0)
    assert transform["positionY"] == pytest.approx(540.0)


def test_unknown_hotkey_error_does_not_list_scenes(monkeypatch):
    obs_client.reset_clients()

    class RejectingClient(FakeOBSClient):
        def call(self, request_type, *, data=None):
            if request_type == "GetSceneList":
                return {"scenes": [{"sceneName": "Cat"}]}
            raise OBSRequestError(
                request_type, code=600, comment="No hotkeys were found by that name."
            )

    monkeypatch.setattr(
        obs_client, "_connect", lambda host, port, password, timeout: RejectingClient()
    )

    result = _action_block().run(
        **_scene_kwargs(action="trigger_hotkey", scene_name=None, hotkey_name="Nope")
    )

    assert result["error_status"] is True
    assert "No hotkeys were found" in result["message"]
    assert "Available scenes" not in result["message"]
    obs_client.reset_clients()


def _connection_manifest(**fields):
    return ConnectionManifest.model_validate(
        {"type": "roboflow_core/obs_connection@v1", "name": "obs", **fields}
    )


def test_connection_declares_only_the_work_it_is_configured_to_do():
    assert _connection_manifest().discover_work_operations() == [
        WorkOperation.EXTERNAL_REQUEST,
        WorkOperation.STORAGE_READ,
    ]
    assert (
        _connection_manifest(
            verify_connection=False, discover_password=False
        ).discover_work_operations()
        == []
    )


def test_obs_blocks_declare_no_dependent_resources():
    action = ActionManifest.model_validate(
        {
            "type": "roboflow_core/obs_action@v1",
            "name": "obs",
            "connection": "$steps.obs_connection.connection",
            "action": "set_scene",
            "scene_name": "Detected",
        }
    )

    assert action.discover_work_operations() == [WorkOperation.EXTERNAL_REQUEST]
    assert action.discover_dependent_resources() == []
    assert _connection_manifest().discover_dependent_resources() == []
