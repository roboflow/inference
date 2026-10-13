"""Offline author checks: real controller/PID, fake camera; SDK tests labeled below."""

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from roboflow_workflows.v2.blocks.industrial.onvif import (
    OnvifCameraPort,
    OnvifController,
    OnvifError,
    OnvifTarget,
    OnvifTrackingBlock,
    TrackingOptions,
)
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.masks import SegmentationView
from roboflow_workflows.v2.data.secrets import Secret

TARGET = OnvifTarget(
    "camera.invalid", 80, "viewer", Secret("sentinel-password", name="camera")
)
OPTIONS = TrackingOptions(dead_zone=5, pid_kp=1, pid_kd=0, camera_update_rate_limit=0)


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds=1):
        self.now += seconds


class RecordingCamera:
    supports_zoom = True

    def __init__(self, target=None):
        self.commands = []
        self.open_count = 0
        self.close_count = 0
        self.fail = set()
        self.preset_called = threading.Event()
        self.move_entered = threading.Event()
        self.move_gate = None
        self.callback = None

    async def open(self):
        self.open_count += 1
        self.move_gate = asyncio.Event()
        self.move_gate.set()
        if "open" in self.fail:
            raise RuntimeError("sentinel-password setup")

    async def move(self, x, y, z):
        self.move_entered.set()
        if self.callback:
            self.callback()
        await self.move_gate.wait()
        if "move" in self.fail:
            raise RuntimeError("sentinel-password command")
        self.commands.append(("move", x, y, z))

    async def stop(self):
        self.commands.append(("stop",))
        if "stop" in self.fail:
            raise RuntimeError("sentinel-password stop")

    async def preset(self, name):
        self.commands.append(("preset", name))
        self.preset_called.set()
        if "preset" in self.fail:
            raise RuntimeError("sentinel-password preset")

    async def close(self):
        self.close_count += 1
        self.commands.append(("close",))
        if "close" in self.fail:
            raise RuntimeError("sentinel-password close")


class ManualSleep:
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.waiters = []

    async def __call__(self, delay):
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((delay, future))
        self.started.set()
        try:
            await future
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    def fire(self, owner):
        future = self.waiters[-1][1]
        owner._loop.call_soon_threadsafe(future.set_result, None)


def predictions(
    boxes=((70, 40, 90, 60),),
    *,
    confidence=None,
    trackers=None,
    segment=False,
    crop=False,
):
    count = len(boxes)
    xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(count, 4)
    mapping = FrameMapping(
        "root",
        (100, 100),
        scale_xy=(2, 2) if crop else (1, 1),
        offset_xy=(50, 0) if crop else (0, 0),
    )
    frame = FrameRef(
        "crop" if crop else "root", (50, 25) if crop else (100, 100), mapping, mapping
    )
    fields = dict(
        xyxy=xyxy,
        confidence=torch.tensor(confidence or [0.8] * count),
        class_id=torch.zeros(count, dtype=torch.long),
        image_metadata=frame.prediction_metadata(),
        bboxes_metadata=[
            {"tracker_id": tracker, "label": f"row-{i}"}
            for i, tracker in enumerate(trackers or [None] * count)
        ],
    )
    if segment:
        result = InstanceDetections(
            **fields,
            mask=torch.arange(count * 100 * 100).reshape(count, 100, 100) % 2 == 0,
        )
    else:
        result = Detections(**fields)
    return result


@pytest.fixture
def rig():
    camera, clock = RecordingCamera(), Clock()
    owner = OnvifController(TARGET, port_factory=lambda target: camera, clock=clock)
    yield owner, camera, clock
    camera.fail.clear()
    owner.close()
    assert not owner._thread.is_alive()


def process(owner, value=None, **options):
    return owner.process(
        predictions() if value is None else value,
        target=TARGET,
        options=replace(OPTIONS, **options),
    )


def test_max_confidence_tracker_zero_continuity_and_disappearance(rig):
    owner, camera, clock = rig
    a = predictions(
        ((70, 40, 90, 60), (10, 40, 30, 60)), confidence=[0.9, 0.8], trackers=[0, 1]
    )
    assert process(owner, a)["predictions"].bboxes_metadata[0]["tracker_id"] == 0
    clock.advance()
    b = predictions(
        ((70, 40, 90, 60), (10, 40, 30, 60)), confidence=[0.1, 0.99], trackers=[0, 1]
    )
    assert process(owner, b)["predictions"].bboxes_metadata[0]["tracker_id"] == 0
    clock.advance()
    assert (
        process(owner, b, follow_tracker=False)["predictions"].bboxes_metadata[0][
            "tracker_id"
        ]
        == 1
    )
    clock.advance()
    assert process(owner, a)["predictions"].bboxes_metadata[0]["tracker_id"] == 0
    clock.advance()
    c = predictions(trackers=[2])
    assert process(owner, c)["predictions"].bboxes_metadata[0]["tracker_id"] == 2
    assert camera.open_count == 1


@pytest.mark.parametrize("segment", [False, True])
def test_empty_stops_preserves_family_metadata_and_clears_tracker(rig, segment):
    owner, camera, clock = rig
    process(owner, predictions(trackers=[0]))
    empty = predictions((), segment=segment)
    result = process(owner, empty)
    assert result == {"predictions": empty, "seeking": False}
    assert owner._tracked is None
    assert camera.commands[-1] == ("stop",)


def test_selection_keeps_segmentation_masks_and_metadata(rig):
    owner, _, _ = rig
    value = predictions(
        ((5, 5, 20, 20), (70, 40, 90, 60)),
        confidence=[0.2, 0.9],
        segment=True,
        trackers=[3, 4],
    )
    result = process(owner, value)["predictions"]
    assert isinstance(result, InstanceDetections)
    assert torch.equal(result.mask, value.mask[1:2])
    assert result.bboxes_metadata == value.bboxes_metadata[1:2]
    assert result.image_metadata == value.image_metadata
    assert result.xyxy.device == value.xyxy.device


def test_lazy_segmentation_view_never_materializes_masks(rig):
    owner, _, _ = rig
    value = SegmentationView.from_dense(predictions(segment=True))
    result = process(owner, value)["predictions"]
    assert isinstance(result, SegmentationView)
    assert value.materialization_count == 0
    assert result.materialization_count == 0


def test_crop_root_mapping_controls_camera_without_rewriting_output(rig):
    owner, camera, _ = rig
    crop = predictions(((10, 20, 20, 30),), crop=True)
    result = process(owner, crop)
    assert camera.commands[-1] == pytest.approx(("move", -0.3, 0.0, 0.0))
    assert torch.equal(result["predictions"].xyxy, crop.xyxy)
    assert result["predictions"].image_metadata == crop.image_metadata


@pytest.mark.parametrize(
    "flip_x,flip_y,expected",
    [(False, True, (-0.3, 0.0, 0.0)), (True, True, (0.3, 0.0, 0.0))],
)
def test_pan_and_flip_trace(rig, flip_x, flip_y, expected):
    owner, camera, _ = rig
    process(owner, flip_x_movement=flip_x, flip_y_movement=flip_y)
    assert camera.commands[-1][1:] == pytest.approx(expected)


@pytest.mark.parametrize("flip_y,expected", [(True, 0.3), (False, -0.3)])
def test_tilt_flip_and_dead_zone(rig, flip_y, expected):
    owner, camera, _ = rig
    process(owner, predictions(((40, 70, 60, 90),)), flip_y_movement=flip_y)
    assert camera.commands[-1][1:] == pytest.approx((0.0, expected, 0.0))
    result = process(owner, predictions(((45, 45, 55, 55),)))
    assert not result["seeking"] and camera.commands[-1] == ("stop",)


def test_minimum_speed_pid_and_relative_axis_scaling(rig):
    owner, camera, _ = rig
    process(
        owner, predictions(((70, 60, 90, 80),)), pid_kp=0.01, minimum_camera_speed=0.2
    )
    assert camera.commands[-1][1:] == pytest.approx((-0.2, 0.2 * 2 / 3, 0))


def test_integral_gain_accumulates_with_monotonic_clock(rig):
    owner, camera, clock = rig
    process(owner, pid_kp=0, pid_ki=1, minimum_camera_speed=0)
    clock.advance()
    process(owner, pid_kp=0, pid_ki=1, minimum_camera_speed=0)
    assert camera.commands[-1][1:] == pytest.approx((-0.3, 0, 0), abs=1e-5)


def test_stop_bypasses_rate_limit_and_axis_stop_is_immediate(rig):
    owner, camera, clock = rig
    process(owner, predictions(((70, 70, 90, 90),)), camera_update_rate_limit=1000)
    process(owner, predictions(((70, 70, 90, 90),)), camera_update_rate_limit=1000)
    assert len(camera.commands) == 1
    process(owner, predictions(), camera_update_rate_limit=1000)
    assert len(camera.commands) == 2 and camera.commands[-1][2] == 0
    process(owner, predictions(()), camera_update_rate_limit=1000)
    assert camera.commands[-1] == ("stop",)


def test_zoom_expiry_edge_backoff_and_post_zoom_speed(rig):
    owner, camera, clock = rig
    process(owner, predictions(((40, 40, 60, 60),)), zoom_if_able=True)
    assert camera.commands[-1][1:] == pytest.approx((0, 0, 0.375))
    clock.advance(0.5)
    process(owner, predictions(((0, 0, 100, 100),)), zoom_if_able=True)
    assert camera.commands[-1][1:] == pytest.approx((0, 0, -0.05))
    clock.advance(3)
    process(owner)
    assert camera.commands[-1][1:] == pytest.approx((-0.15, 0, 0))


def test_variable_speed_negative_axis_emits_fractional_signed_pulses(rig):
    owner, camera, clock = rig
    value = predictions(((65, 40, 85, 60),))
    for _ in range(8):
        process(owner, value, simulate_variable_speed=True)
        clock.advance()
    assert camera.commands == [("stop",)] * 3 + [("move", -1.0, 0.0, 0.0)] + [
        ("stop",)
    ] * 3 + [("move", -1.0, 0.0, 0.0)]


def test_preset_outputs_aligned_empty_and_is_rate_limited(rig):
    owner, camera, clock = rig
    value = predictions(segment=True)
    options = dict(
        movement_type="Go To Preset",
        default_position_preset="home",
        camera_update_rate_limit=1000,
    )
    result = process(owner, value, **options)
    process(owner, value, **options)
    assert isinstance(result["predictions"], InstanceDetections)
    assert len(result["predictions"].xyxy) == 0
    assert result["predictions"].image_metadata == value.image_metadata
    assert camera.commands == [("stop",), ("preset", "home"), ("stop",)]
    clock.advance()
    process(owner, value, **options)
    assert camera.commands[-1] == ("preset", "home")


def test_idle_timer_cancelled_by_single_axis_motion_and_rearmed_once():
    timer, camera = ManualSleep(), RecordingCamera()
    with OnvifController(TARGET, port_factory=lambda t: camera, sleep=timer) as owner:
        options = dict(
            default_position_preset="home", move_to_position_after_idle_seconds=3
        )
        process(owner, predictions(()), **options)
        assert timer.started.wait(2)
        process(owner, predictions(()), **options)
        assert len(timer.waiters) == 1
        process(owner, **options)
        assert timer.cancelled.wait(2)
        timer.started.clear()
        process(owner, predictions(()), **options)
        assert timer.started.wait(2) and len(timer.waiters) == 2
        timer.fire(owner)
        assert camera.preset_called.wait(2)
        process(owner, predictions(()), **options)
        assert len(timer.waiters) == 2


def test_host_reset_cancels_old_generation_timer_then_rearms():
    timer, camera = ManualSleep(), RecordingCamera()
    with OnvifController(TARGET, port_factory=lambda t: camera, sleep=timer) as owner:
        opts = dict(
            default_position_preset="home", move_to_position_after_idle_seconds=3
        )
        process(owner, predictions(()), **opts)
        assert timer.started.wait(2)
        old = timer.waiters[0][1]
        owner.reset_tracking()
        assert old.cancelled() and owner._tracked is None
        with pytest.raises(OnvifError, match="quiesced"):
            process(owner)
        owner.resume()
        timer.started.clear()
        process(owner, predictions(()), **opts)
        assert timer.started.wait(2) and len(timer.waiters) == 2
    assert all(f.done() for _, f in timer.waiters)
    assert not camera.preset_called.is_set()


def test_delayed_failure_surfaces_on_next_command_and_is_retained():
    timer, camera = ManualSleep(), RecordingCamera()
    camera.fail.add("preset")
    with OnvifController(TARGET, port_factory=lambda t: camera, sleep=timer) as owner:
        process(
            owner,
            predictions(()),
            default_position_preset="home",
            move_to_position_after_idle_seconds=3,
        )
        assert timer.started.wait(2)
        timer.fire(owner)
        assert camera.preset_called.wait(2)
        with pytest.raises(OnvifError, match="delayed preset") as caught:
            process(owner)
        assert "sentinel-password" not in str(caught.value)
        assert owner.failures


@pytest.mark.parametrize("failure", ["open", "move"])
def test_setup_command_failure_scrubbed_and_setup_unwinds(failure):
    camera = RecordingCamera()
    camera.fail.add(failure)
    owner = OnvifController(TARGET, port_factory=lambda t: camera)
    try:
        with pytest.raises(OnvifError) as caught:
            process(owner)
        assert "sentinel-password" not in str(caught.value)
        assert caught.value.__context__ is None
        assert bool(owner.failures)
        if failure == "open":
            assert camera.close_count == 1
    finally:
        owner.close()
    assert camera.close_count == 1


def test_stop_and_close_failure_still_join_and_close_once():
    camera = RecordingCamera()
    owner = OnvifController(TARGET, port_factory=lambda t: camera)
    process(owner)
    camera.fail.update(("stop", "close"))
    with pytest.raises(OnvifError, match="Stop"):
        owner.close()
    owner.close()
    assert owner.closed and not owner._thread.is_alive()
    assert camera.close_count == 1
    assert len(owner.failures) == 2


def test_close_waits_for_admitted_command_and_rejects_new_work(rig):
    owner, camera, clock = rig
    process(owner)
    camera.move_entered.clear()
    owner._loop.call_soon_threadsafe(camera.move_gate.clear)
    closing = threading.Event()
    original_shutdown = owner._shutdown

    async def observed_shutdown():
        closing.set()
        return await original_shutdown()

    owner._shutdown = observed_shutdown
    with ThreadPoolExecutor(2) as pool:
        command = pool.submit(process, owner)
        assert camera.move_entered.wait(2)
        close = pool.submit(owner.close)
        assert closing.wait(2)
        assert not close.done() and camera.close_count == 0
        with pytest.raises(OnvifError, match="closed"):
            process(owner)
        owner._loop.call_soon_threadsafe(camera.move_gate.set)
        assert command.result(2)["seeking"]
        close.result(2)
    assert camera.commands[-2:] == [("stop",), ("close",)]


def test_owner_loop_callback_cannot_join_itself(rig):
    owner, camera, _ = rig
    errors = []

    def callback():
        try:
            owner.close()
        except OnvifError as e:
            errors.append(str(e))

    camera.callback = callback
    process(owner)
    assert "owner loop" in errors[0]
    assert not owner.closed


def test_identity_change_rejected_without_closing_borrowed_work(rig):
    owner, camera, _ = rig
    process(owner)
    with pytest.raises(OnvifError, match="identity changed"):
        owner.process(
            predictions(), target=replace(TARGET, password="different"), options=OPTIONS
        )
    assert camera.close_count == 0
    process(owner)


def test_disabled_no_owner_no_reveal_and_missing_owner_actionable():
    block = OnvifTrackingBlock()
    value = predictions()
    args = dict(
        predictions=value,
        camera_ip="camera.invalid",
        camera_username="viewer",
        camera_password=Secret.unavailable("camera"),
    )
    assert block.run(**args, disable_sink=True) == {
        "predictions": value,
        "seeking": False,
    }
    with pytest.raises(OnvifError, match="resources"):
        block.run(**args)
    assert OnvifTrackingBlock(disable_sinks=True).run(**args)["predictions"] is value


@pytest.mark.parametrize(
    "options",
    [
        {"minimum_camera_speed": 2},
        {"pid_kp": float("nan")},
        {"dead_zone": -1},
        {"movement_type": "bad"},
        {"move_to_position_after_idle_seconds": 2},
    ],
)
def test_invalid_settings_rejected_before_commands(options):
    with pytest.raises(ValueError):
        replace(OPTIONS, **options)


def test_no_resource_acquisition_on_construct_and_repeated_host_close():
    for _ in range(3):
        calls = []
        owner = OnvifController(TARGET, port_factory=lambda t: calls.append(t))
        owner.close()
        owner.close()
        assert not calls and not owner._thread.is_alive()


# Installed adapter evidence: real onvif-zeep-async 2.0.0 WSDLs, real Zeep
# parsing/serialization and HTTPX clients, in-memory MockTransport. No camera,
# socket, listener or ONVIFCamera/ONVIFService Python wrapper is used.
SOAP = "http://www.w3.org/2003/05/soap-envelope"
TT = "http://www.onvif.org/ver10/schema"


def soap(body):
    return f'<s:Envelope xmlns:s="{SOAP}" xmlns:tt="{TT}"><s:Body>{body}</s:Body></s:Envelope>'


def sdk_fixture(monkeypatch, *, fail_operation=None):
    import httpx
    from lxml import etree

    requests = []
    clients = []
    device = "http://www.onvif.org/ver10/device/wsdl"
    media = "http://www.onvif.org/ver10/media/wsdl"
    ptz = "http://www.onvif.org/ver20/ptz/wsdl"

    def handler(request):
        root = etree.fromstring(request.content)
        operation = etree.QName(root.find(f"{{{SOAP}}}Body")[0]).localname
        requests.append((operation, root))
        if operation == fail_operation:
            raise httpx.ReadTimeout("sentinel-password", request=request)
        bodies = {
            "GetCapabilities": f'<d:GetCapabilitiesResponse xmlns:d="{device}"><d:Capabilities><tt:Media><tt:XAddr>http://camera.invalid/media</tt:XAddr></tt:Media><tt:PTZ><tt:XAddr>http://camera.invalid/ptz</tt:XAddr></tt:PTZ></d:Capabilities></d:GetCapabilitiesResponse>',
            "GetProfiles": f'<m:GetProfilesResponse xmlns:m="{media}"><m:Profiles token="profile"><tt:Name>main</tt:Name><tt:PTZConfiguration token="config"><tt:Name>ptz</tt:Name><tt:UseCount>1</tt:UseCount><tt:NodeToken>node</tt:NodeToken></tt:PTZConfiguration></m:Profiles></m:GetProfilesResponse>',
            "GetConfigurationOptions": f'<p:GetConfigurationOptionsResponse xmlns:p="{ptz}"><p:PTZConfigurationOptions><tt:Spaces><tt:ContinuousPanTiltVelocitySpace><tt:URI>pan</tt:URI><tt:XRange><tt:Min>-2</tt:Min><tt:Max>3</tt:Max></tt:XRange><tt:YRange><tt:Min>-4</tt:Min><tt:Max>5</tt:Max></tt:YRange></tt:ContinuousPanTiltVelocitySpace><tt:ContinuousZoomVelocitySpace><tt:URI>zoom</tt:URI><tt:XRange><tt:Min>-6</tt:Min><tt:Max>7</tt:Max></tt:XRange></tt:ContinuousZoomVelocitySpace></tt:Spaces></p:PTZConfigurationOptions></p:GetConfigurationOptionsResponse>',
            "GetPresets": f'<p:GetPresetsResponse xmlns:p="{ptz}"><p:Preset token="home-token"><tt:Name>home</tt:Name></p:Preset></p:GetPresetsResponse>',
        }
        body = bodies.get(operation, f'<p:{operation}Response xmlns:p="{ptz}"/>')
        return httpx.Response(
            200, content=soap(body), headers={"Content-Type": "application/soap+xml"}
        )

    async_cls, sync_cls = httpx.AsyncClient, httpx.Client

    def async_factory(*args, **kwargs):
        client = async_cls(*args, transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    def sync_factory(*args, **kwargs):
        client = sync_cls(*args, transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", async_factory)
    monkeypatch.setattr(httpx, "Client", sync_factory)
    return requests, clients


def test_installed_wsdl_zeep_httpx_commands_and_complete_cleanup(monkeypatch):
    requests, clients = sdk_fixture(monkeypatch)
    monkeypatch.setenv("http_proxy", "http://proxy.invalid")
    monkeypatch.setenv("https_proxy", "http://proxy.invalid")
    before = dict(os.environ)
    with OnvifController(TARGET) as owner:
        process(owner, predictions(((70, 70, 90, 90),)), zoom_if_able=False)
        process(owner, movement_type="Go To Preset", default_position_preset="home")
    assert all(client.is_closed for client in clients) and len(clients) == 2
    assert dict(os.environ) == before
    assert [name for name, _ in requests] == [
        "GetCapabilities",
        "GetProfiles",
        "GetConfigurationOptions",
        "GetPresets",
        "ContinuousMove",
        "Stop",
        "GotoPreset",
        "Stop",
    ]
    move = requests[4][1]
    axis = move.find(f".//{{{TT}}}PanTilt")
    assert float(axis.get("x")) == pytest.approx(-0.6)
    assert float(axis.get("y")) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "fail_operation",
    ["GetCapabilities", "GetProfiles", "GetConfigurationOptions", "GetPresets"],
)
def test_installed_wsdl_partial_setup_closes_both_clients(monkeypatch, fail_operation):
    requests, clients = sdk_fixture(monkeypatch, fail_operation=fail_operation)
    owner = OnvifController(TARGET)
    with pytest.raises(OnvifError) as caught:
        process(owner)
    owner.close()
    assert all(client.is_closed for client in clients)
    assert fail_operation in [name for name, _ in requests]
    assert "sentinel-password" not in str(caught.value)
    assert caught.value.__context__ is None
    if fail_operation in ("GetConfigurationOptions", "GetPresets"):
        assert requests[-1][0] == "Stop"


def test_installed_wsdl_second_client_construction_failure_closes_first(monkeypatch):
    _, clients = sdk_fixture(monkeypatch)
    import httpx

    def fail(*args, **kwargs):
        raise RuntimeError("sentinel-password")

    monkeypatch.setattr(httpx, "Client", fail)
    with OnvifController(TARGET) as owner:
        with pytest.raises(OnvifError):
            process(owner)
    assert len(clients) == 1 and clients[0].is_closed


def test_no_full_tensor_or_mask_host_roundtrip(rig, monkeypatch):
    owner, _, _ = rig
    value = predictions(
        ((5, 5, 20, 20), (70, 40, 90, 60)), confidence=[0.2, 0.9], segment=True
    )
    original = torch.Tensor.tolist

    def small_list(tensor, *args, **kwargs):
        assert tensor.numel() <= 4, "full tensor crossed host boundary"
        return original(tensor, *args, **kwargs)

    def no_cpu(*args, **kwargs):
        raise AssertionError("unexpected explicit tensor CPU transfer")

    monkeypatch.setattr(torch.Tensor, "tolist", small_list)
    monkeypatch.setattr(torch.Tensor, "cpu", no_cpu)
    result = process(owner, value)["predictions"]
    assert result.mask.shape == (1, 100, 100)


def test_rle_selection_keeps_compressed_masks_aligned(rig):
    from inference_models.models.base.types import InstancesRLEMasks

    owner, _, _ = rig
    value = predictions(
        ((5, 5, 20, 20), (70, 40, 90, 60)), confidence=[0.2, 0.9], segment=True
    )
    value.mask = InstancesRLEMasks(image_size=(100, 100), masks=[b"`h9", b"`h9"])
    result = process(owner, value)["predictions"]
    assert isinstance(result.mask, InstancesRLEMasks)
    assert result.mask.masks == [value.mask.masks[1]]


def test_derivative_gain_changes_command_after_target_motion(rig):
    owner, camera, clock = rig
    process(owner, pid_kp=0, pid_kd=0.5, minimum_camera_speed=0)
    clock.advance()
    process(
        owner,
        predictions(((75, 40, 95, 60),)),
        pid_kp=0,
        pid_kd=0.5,
        minimum_camera_speed=0,
    )
    assert camera.commands[-1][1:] == pytest.approx((-0.025, 0, 0))


def test_zoom_disabled_by_camera_capability(rig):
    owner, camera, _ = rig
    camera.supports_zoom = False
    result = process(owner, predictions(((40, 40, 60, 60),)), zoom_if_able=True)
    assert result["seeking"] is False and camera.commands == [("stop",)]


def test_open_cancellation_still_unwinds_partial_resource():
    camera = RecordingCamera()

    async def cancelled_open():
        camera.open_count += 1
        raise asyncio.CancelledError()

    camera.open = cancelled_open
    with OnvifController(TARGET, port_factory=lambda t: camera) as owner:
        with pytest.raises(OnvifError, match="CancelledError"):
            process(owner)
    assert camera.close_count == 1 and not owner._thread.is_alive()


def test_setup_failure_retains_original_when_stop_and_close_also_fail():
    camera = RecordingCamera()
    camera.fail.update(("open", "stop", "close"))
    with OnvifController(TARGET, port_factory=lambda t: camera) as owner:
        with pytest.raises(OnvifError, match="command failed"):
            process(owner)
        assert len(owner.failures) == 3
        assert "partial-setup stop" in owner.failures[0]
        assert "partial-setup close" in owner.failures[1]
        assert "command" in owner.failures[2]
    assert camera.close_count == 1


def test_missing_driver_is_actionable_and_lazy(monkeypatch):
    from importlib import metadata

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    with OnvifController(TARGET) as owner:
        assert owner._port is None
        with pytest.raises(OnvifError, match="onvif-zeep-async==2.0.0"):
            process(owner)


def test_adapter_close_attempts_both_clients_and_preserves_both_errors():
    port = OnvifCameraPort(TARGET)
    calls = []

    class AsyncClient:
        async def aclose(self):
            calls.append("async")
            raise RuntimeError("sentinel-password")

    class SyncClient:
        def close(self):
            calls.append("sync")
            raise ValueError("sentinel-password")

    port._async_client = AsyncClient()
    port._wsdl_client = SyncClient()
    with pytest.raises(OnvifError) as caught:
        asyncio.run(port.close())
    asyncio.run(port.close())
    assert calls == ["async", "sync"]
    assert "RuntimeError" in str(caught.value) and "ValueError" in str(caught.value)
    assert "sentinel-password" not in str(caught.value)
    assert caught.value.__context__ is None


def test_repeated_actual_adapter_owners_close_every_client(monkeypatch):
    requests, clients = sdk_fixture(monkeypatch)
    for _ in range(3):
        with OnvifController(TARGET) as owner:
            process(owner)
        owner.close()
    assert len(clients) == 6 and all(client.is_closed for client in clients)
    assert [operation for operation, _ in requests].count("Stop") == 3
