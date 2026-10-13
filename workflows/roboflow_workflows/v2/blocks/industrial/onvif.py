"""Tensor-native PTZ tracking with an explicit host-owned ONVIF controller.

The host constructs one ``OnvifController`` per logical camera stream, injects
it as ``onvif_controller``, and closes it after all sessions stop. Sharing an
owner intentionally shares target, PID, zoom and rate-limit state. Sessions do
not close it. Before a processing reset: stop submitting runs, drain those runs,
call ``quiesce()``, apply the engine update, then ``resume()``. An engine reset
alone does not cancel the owner's delayed preset or retire its tracking state.

Calls wait for camera commands; only the idle preset is delayed. An acknowledged
preset keeps travelling: empty, dead-zone and rate-limited preset frames send no
Stop until new motion, quiesce or close. Command errors fail the call; delayed
errors remain in ``failures`` and fail the next call or quiesce/close. Failed
local checks (geometry, boxes, preset name, profile, service address) name
their cause; earlier work in that call, such as connecting or a preparatory
Stop, may still have happened. Other errors report only their type, and their
outcome may be uncertain. No command is retried after an uncertain network
result. Closing waits for
admitted work, attempts Stop, closes acquired clients and joins the loop. HTTP
timeouts bound individual I/O, not total shutdown time.

Only a selected box's four coordinates, a confidence argmax index and optional
tracker scalar cross the device boundary. Aligned selection preserves masks,
row identity, local coordinates and metadata; camera geometry uses the root
mapping without expanding masks. V1 differences include tracker zero support,
correct per-axis velocity ranges, cancellation on any motion axis, and keeping
the segmentation family for empty/preset output. See the offline example.
"""

import asyncio
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, Union
from urllib.parse import urlsplit

import torch
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr
from roboflow_workflows.execution_engine.entities.workload import (
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkOperation,
)
from roboflow_workflows.v2.blocks.query.detections import boxes_of, select_rows
from roboflow_workflows.v2.data.frames import root_placement_from_metadata
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.prediction_ops import RowSelection
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_VIEW_KIND,
)


class OnvifError(RuntimeError):
    """A sanitized camera operation or host-lifecycle failure."""


class _DriverUnavailable(OnvifError):
    pass


class _TransportCloseError(OnvifError):
    pass


# Value-free texts for this module's own checks. Only these reach the host
# verbatim; any other error text may carry credentials and is dropped.
_LOCAL_DIAGNOSTICS = {
    "reopen": "camera port cannot be reopened",
    "remote_wsdl": "remote ONVIF WSDL loading is disabled",
    "service_address": "camera advertised an invalid ONVIF service address",
    "profile": "camera has no media profile at profile_index",
    "preset": "camera does not advertise the requested preset; "
    "check default_position_preset",
    "root_geometry": "tracking requires root frame geometry in image_metadata",
    "box": "tracking requires finite ordered bounding boxes",
}


class _LocalFailure(OnvifError):
    """A local check failed; the message comes only from ``_LOCAL_DIAGNOSTICS``."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__("ONVIF " + _LOCAL_DIAGNOSTICS[reason])


@dataclass(frozen=True, repr=False)
class OnvifTarget:
    """Immutable connection identity; representation never contains credentials.

    Args:
        host: Camera hostname or IP, without URL credentials or path.
        port: ONVIF device-service port.
        username: Camera account.
        password: Plaintext or a Secret; revealed only on enabled use.
        scheme: HTTP or HTTPS. HTTPS verifies certificates.
        timeout: Per-I/O HTTP timeout in seconds; not a shutdown deadline.
        profile_index: Media profile selected from GetProfiles.
    """

    host: str
    port: int
    username: str = field(repr=False)
    password: Union[str, Secret] = field(repr=False)
    scheme: Literal["http", "https"] = "http"
    timeout: float = 10.0
    profile_index: int = 0

    def __post_init__(self):
        if not self.host or any(c in self.host for c in "/@?#"):
            raise ValueError(
                "ONVIF host must be a hostname or IP, without URL credentials"
            )
        if self.scheme not in ("http", "https") or not 1 <= self.port <= 65535:
            raise ValueError("ONVIF requires an HTTP(S) scheme and port 1..65535")
        if (
            not math.isfinite(self.timeout)
            or self.timeout <= 0
            or self.profile_index < 0
        ):
            raise ValueError(
                "ONVIF timeout must be positive and profile_index nonnegative"
            )

    def __repr__(self):
        return "OnvifTarget(<connection identity>)"

    def _identity(self):
        password = (
            self.password.reveal()
            if isinstance(self.password, Secret)
            else self.password
        )
        return (
            self.host,
            self.port,
            self.username,
            password,
            self.scheme,
            self.timeout,
            self.profile_index,
        )


@dataclass(frozen=True)
class TrackingOptions:
    """Validated motion settings, snapshotted for one admitted frame.

    Args:
        movement_type: Follow or Go To Preset.
        default_position_preset: Camera preset name.
        zoom_if_able: Enable centering followed by zoom.
        follow_tracker: Retain a selected tracker ID until it disappears.
        dead_zone: Root-frame pixel tolerance around center.
        camera_update_rate_limit: Minimum command interval, milliseconds.
        flip_y_movement: Reverse vertical movement.
        flip_x_movement: Reverse horizontal movement.
        move_to_position_after_idle_seconds: Idle delay; zero disables it.
        pid_kp: Proportional gain.
        pid_ki: Integral gain.
        pid_kd: Derivative gain.
        minimum_camera_speed: Minimum nonzero speed before relative-axis scaling.
        simulate_variable_speed: Approximate speed with signed full-speed pulses.
    """

    movement_type: Literal["Follow", "Go To Preset"] = "Follow"
    default_position_preset: str = ""
    zoom_if_able: bool = False
    follow_tracker: bool = True
    dead_zone: int = 50
    camera_update_rate_limit: int = 250
    flip_y_movement: bool = True
    flip_x_movement: bool = False
    move_to_position_after_idle_seconds: float = 0
    pid_kp: float = 0.25
    pid_ki: float = 0
    pid_kd: float = 1
    minimum_camera_speed: float = 0.05
    simulate_variable_speed: bool = False

    def __post_init__(self):
        numbers = (
            self.dead_zone,
            self.camera_update_rate_limit,
            self.move_to_position_after_idle_seconds,
            self.pid_kp,
            self.pid_ki,
            self.pid_kd,
            self.minimum_camera_speed,
        )
        if not all(math.isfinite(v) and v >= 0 for v in numbers):
            raise ValueError("ONVIF movement settings must be finite and nonnegative")
        if self.minimum_camera_speed > 1:
            raise ValueError("minimum_camera_speed must be at most 1")
        if self.movement_type not in ("Follow", "Go To Preset"):
            raise ValueError("Unknown ONVIF movement_type")
        if (
            self.movement_type == "Go To Preset"
            or self.move_to_position_after_idle_seconds
        ) and not self.default_position_preset:
            raise ValueError("ONVIF preset movement requires default_position_preset")


class CameraPort(Protocol):
    """Async camera operations, exclusively owned and serialized by the controller.

    ``open`` may fail after partial acquisition; ``close`` must release every
    acquired client even then. ``move`` receives normalized velocities in [-1,1].
    Implementations must await their operations and not leave detached work.
    """

    supports_zoom: bool

    async def open(self) -> None:
        """Acquire and configure the camera; partial acquisition remains owned."""
        ...

    async def move(self, x: float, y: float, z: float) -> None:
        """Send normalized pan, tilt and zoom velocities."""
        ...

    async def stop(self) -> None:
        """Attempt the ONVIF Stop operation on all supported axes."""
        ...

    async def preset(self, name: str) -> None:
        """Move to a named preset, failing if it is absent."""
        ...

    async def close(self) -> None:
        """Close all acquired transports, including partially configured ones."""
        ...


class _TransportDiagnostics(logging.LoggerAdapter):
    """Replace SDK wire diagnostics with value-free events for this transport.

    Never forward SDK text, URLs, XML, headers, extra fields or exception info.
    The underlying logger stays untouched, including for concurrent clients.
    """

    def log(self, level, msg, *args, **kwargs):
        if not self.isEnabledFor(level):
            return

        if msg == "HTTP Post to %s:\n%s":
            self.logger.log(level, "ONVIF transport request")
        elif (
            msg == "HTTP Response from %s (status: %d):\n%s"
            and len(args) == 3
            and type(args[1]) is int
        ):
            self.logger.log(level, "ONVIF transport response (status: %d)", args[1])
        else:
            self.logger.log(level, "ONVIF transport diagnostic")


class OnvifCameraPort:
    """Service-level adapter using installed ONVIF WSDLs, Zeep and HTTPX.

    Unlike onvif-zeep-async 2.0.0's ONVIFCamera/ONVIFService constructors, this
    adapter owns both HTTP clients before WSDL setup, leaves proxy environment
    variables unchanged, disables environment proxy discovery, and closes both
    clients on every setup/command/shutdown outcome. WSDL loads are local-only.
    Transport diagnostics contain only events and integer response statuses,
    never wire contents, addresses or exception text. Other clients keep their
    logging configuration. Optional drivers load at ``open``; construction does no I/O.

    Args:
        target: Immutable camera connection configuration.
    """

    def __init__(self, target: OnvifTarget):
        self._target = target
        self._async_client = None
        self._wsdl_client = None
        self._transport = None
        self._ptz = None
        self._profile = None
        self._limits = None
        self._presets = {}
        self._closed = False
        self.supports_zoom = False

    async def open(self) -> None:
        """Discover services, profile, velocity ranges and named presets.

        Raises:
            OnvifError: If optional drivers are unavailable, an advertised service
                address is invalid or ``profile_index`` has no media profile.
        """
        if self._closed or self._async_client is not None:
            raise _LocalFailure("reopen")
        try:
            import httpx
            from zeep import AsyncClient, Settings
            from zeep.proxy import AsyncServiceProxy
            from zeep.transports import AsyncTransport
            from zeep.wsse.username import UsernameToken

            distribution = metadata.distribution("onvif-zeep-async")
        except (ImportError, metadata.PackageNotFoundError):
            raise _DriverUnavailable(
                "ONVIF requires onvif-zeep-async==2.0.0 and its httpx/zeep dependencies"
            ) from None

        class LocalWsdlTransport(AsyncTransport):
            def load(self, url):
                if urlsplit(str(url)).scheme not in ("", "file"):
                    raise _LocalFailure("remote_wsdl")
                return super().load(url)

        target = self._target
        self._async_client = httpx.AsyncClient(timeout=target.timeout, trust_env=False)
        self._wsdl_client = httpx.Client(timeout=target.timeout, trust_env=False)
        self._transport = LocalWsdlTransport(
            client=self._async_client, wsdl_client=self._wsdl_client
        )
        self._transport.logger = _TransportDiagnostics(self._transport.logger, {})
        password = target._identity()[3]
        auth = UsernameToken(target.username, password, use_digest=True)

        def service(filename, namespace, binding, address):
            parsed = urlsplit(address)
            if (
                parsed.scheme not in ("http", "https")
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise _LocalFailure("service_address")
            wsdl = Path(distribution.locate_file("onvif/wsdl")) / filename
            client = AsyncClient(
                str(wsdl),
                transport=self._transport,
                wsse=auth,
                settings=Settings(strict=False),
            )
            proxy = AsyncServiceProxy(
                client,
                client.wsdl.bindings[f"{{{namespace}}}{binding}"],
                address=address,
            )
            return proxy

        host = target.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        device = service(
            "devicemgmt.wsdl",
            "http://www.onvif.org/ver10/device/wsdl",
            "DeviceBinding",
            f"{target.scheme}://{host}:{target.port}/onvif/device_service",
        )
        capabilities = await device.GetCapabilities(Category="All")
        media = service(
            "media.wsdl",
            "http://www.onvif.org/ver10/media/wsdl",
            "MediaBinding",
            capabilities.Media.XAddr,
        )
        self._ptz = service(
            "ptz.wsdl",
            "http://www.onvif.org/ver20/ptz/wsdl",
            "PTZBinding",
            capabilities.PTZ.XAddr,
        )
        profiles = await media.GetProfiles()
        if target.profile_index >= len(profiles or ()):
            raise _LocalFailure("profile")
        profile = profiles[target.profile_index]
        # Set the token before options discovery: close can Stop after a partial setup.
        self._profile = profile.token
        options = await self._ptz.GetConfigurationOptions(
            ConfigurationToken=profile.PTZConfiguration.token
        )
        pan = options.Spaces.ContinuousPanTiltVelocitySpace[0]
        zoom = options.Spaces.ContinuousZoomVelocitySpace
        self._limits = (
            (pan.XRange.Min, pan.XRange.Max),
            (pan.YRange.Min, pan.YRange.Max),
            (zoom[0].XRange.Min, zoom[0].XRange.Max) if zoom else None,
        )
        self.supports_zoom = self._limits[2] is not None
        self._presets = {
            preset.Name: preset.token
            for preset in await self._ptz.GetPresets(ProfileToken=self._profile)
        }

    async def move(self, x: float, y: float, z: float) -> None:
        """Scale each normalized axis with its own sign and camera limits.

        Args:
            x: Pan velocity in [-1, 1].
            y: Tilt velocity in [-1, 1].
            z: Zoom velocity in [-1, 1]; ignored if unsupported.
        """
        velocities = []
        for value, limits in zip((x, y, z), self._limits):
            velocities.append(
                0.0 if limits is None else value * abs(limits[0 if value < 0 else 1])
            )
        velocity = {"PanTilt": {"x": velocities[0], "y": velocities[1]}}
        if self._limits[2] is not None:
            velocity["Zoom"] = {"x": velocities[2]}
        await self._ptz.ContinuousMove(ProfileToken=self._profile, Velocity=velocity)

    async def stop(self) -> None:
        """Attempt Stop when a media profile was acquired, including partial setup."""
        if self._profile is not None:
            await self._ptz.Stop(
                ProfileToken=self._profile,
                PanTilt=True,
                Zoom=self.supports_zoom if self._limits is not None else True,
            )

    async def preset(self, name: str) -> None:
        """Move to a discovered preset.

        Args:
            name: Exact preset name.

        Raises:
            OnvifError: If the camera did not advertise that preset.
        """
        if name not in self._presets:
            raise _LocalFailure("preset")
        await self._ptz.GotoPreset(
            ProfileToken=self._profile, PresetToken=self._presets[name]
        )

    async def close(self) -> None:
        """Attempt both client closes even if the first fails; never reopen."""
        if self._closed:
            return
        self._closed = True
        failures = []
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
            except BaseException as error:
                failures.append("async HTTP: " + type(error).__name__)
        if self._wsdl_client is not None:
            try:
                self._wsdl_client.close()
            except BaseException as error:
                failures.append("WSDL HTTP: " + type(error).__name__)
        if failures:
            raise _TransportCloseError(
                "ONVIF client close failed (" + "; ".join(failures) + ")"
            )


@dataclass
class _Failure:
    message: str


class OnvifController:
    """Host-owned loop, transport, tracking state and delayed preset.

    Commands are serialized across borrowers and finish before ``process``
    returns. The host must close this owner, including after a failed command.
    No destructor or engine hook performs cleanup. Callbacks running on the
    owner loop must never call its synchronous public methods.

    Args:
        target: Fixed connection identity; mismatches reject enabled block calls.
        port_factory: Optional async CameraPort factory for host adapters/tests.
        clock: Monotonic clock for control state (seconds).
        sleep: Async timer primitive; injectable for deterministic tests.
    """

    def __init__(
        self,
        target: OnvifTarget,
        *,
        port_factory: Callable = OnvifCameraPort,
        clock: Callable = time.monotonic,
        sleep: Callable = asyncio.sleep,
    ):
        self._target = target
        self._port_factory = port_factory
        self._clock = clock
        self._sleep = sleep
        self._port = None
        self._pid = None
        self._timer = None
        self._failures = []
        self._pending_failure = None
        self._executor_to_join = None
        self._admission = threading.Lock()
        self._lifecycle = threading.Lock()
        self._accepting = True
        self._closed = False
        self._serial = asyncio.Lock()
        self._clear_state()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="onvif-controller")
        try:
            self._thread.start()
            self._ready.wait()
        except BaseException:
            self._loop.close()
            raise

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _clear_state(self):
        self._tracked = None
        self._seeking = False
        self._zoom_start = None
        self._zoomed = False
        self._last_update = -math.inf
        self._speeds = (0.0, 0.0, 0.0)
        self._pulse = [0.0, 0.0, 0.0]
        self._idle_key = None
        self._idle_fired = False
        self._preset_travel = False
        if self._pid is not None:
            for pid in self._pid:
                pid.reset()

    def _check_thread(self):
        if threading.current_thread() is self._thread:
            raise OnvifError("ONVIF synchronous methods cannot run on the owner loop")

    def _record_failure(self, operation, error):
        # No third-party error text or exception chain escapes; those may hold credentials.
        # Exact types and a table lookup: injected errors cannot supply local text.
        reason = error.reason if type(error) is _LocalFailure else None
        if type(error) in (_DriverUnavailable, _TransportCloseError):
            message = str(error)
        elif isinstance(reason, str) and reason in _LOCAL_DIAGNOSTICS:
            message = f"ONVIF {operation} failed: {_LOCAL_DIAGNOSTICS[reason]}"
        else:
            message = f"ONVIF {operation} failed ({type(error).__name__}); outcome may be uncertain"
        failure = _Failure(message)
        with self._admission:
            self._failures.append(failure.message)
        return failure

    def _wait(self, future):
        result = future.result()
        if isinstance(result, _Failure):
            raise OnvifError(result.message)
        return result

    @property
    def failures(self) -> tuple:
        """Return retained sanitized errors, including delayed-command failures."""
        with self._admission:
            result = tuple(self._failures)
        return result

    @property
    def closed(self) -> bool:
        """Whether host shutdown has completed and joined the loop thread."""
        return self._closed

    def process(
        self, predictions: Any, *, target: OnvifTarget, options: TrackingOptions
    ) -> dict:
        """Select one target and await the corresponding camera command.

        Args:
            predictions: Tensor detection/segmentation carrier with frame metadata.
            target: Must match this owner's complete connection identity.
            options: Immutable movement settings for this frame.

        Returns:
            Aligned predictions and the last acknowledged seeking state.

        Raises:
            OnvifError: On a closed/quiesced owner, identity change or camera failure.
        """
        self._check_thread()
        if target._identity() != self._target._identity():
            raise OnvifError(
                "ONVIF connection identity changed; explicitly bind a new host owner"
            )
        with self._admission:
            if not self._accepting:
                raise OnvifError("ONVIF owner is quiesced or closed")
            future = asyncio.run_coroutine_threadsafe(
                self._process(predictions, options), self._loop
            )
        result = self._wait(future)
        return result

    async def _ensure_port(self):
        if self._port is not None:
            return
        try:
            from simple_pid import PID
        except ImportError:
            raise _DriverUnavailable(
                "ONVIF tracking requires simple-pid==2.0.1"
            ) from None
        if self._pid is None:
            self._pid = [
                PID(0, 0, 0, setpoint=0, output_limits=(-1, 1), time_fn=self._clock)
                for _ in range(3)
            ]
        self._port = self._port_factory(self._target)
        try:
            await self._port.open()
        except BaseException:
            # Keep the original setup failure while recording both cleanup attempts.
            for operation in ("stop", "close"):
                try:
                    await getattr(self._port, operation)()
                except BaseException as error:
                    self._record_failure(f"partial-setup {operation}", error)
            self._port = None
            raise

    async def _cancel_timer(self):
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
            try:
                await timer
            except asyncio.CancelledError:
                pass

    async def _idle_preset(self, key):
        try:
            await self._sleep(key[1])
            async with self._serial:
                with self._admission:
                    accepting = self._accepting
                if accepting and self._idle_key == key:
                    await self._port.preset(key[0])
                    self._idle_fired = True
                    self._preset_travel = True
                    self._tracked = None
                    self._zoomed = False
                    self._zoom_start = None
                    for pid in self._pid:
                        pid.reset()
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._pending_failure = self._record_failure("delayed preset", error)

    async def _idle(self, options):
        key = (
            options.default_position_preset,
            options.move_to_position_after_idle_seconds,
        )
        if key != self._idle_key:
            await self._cancel_timer()
            self._idle_key = key
            self._idle_fired = False
        if key[1] and not self._idle_fired and self._timer is None:
            self._timer = asyncio.create_task(self._idle_preset(key))

    async def _active(self):
        await self._cancel_timer()
        self._idle_key = None
        self._idle_fired = False

    async def _stop(self, options):
        # Stop is never rate-limited, even on the first empty frame. It would also
        # abort a preset the camera acknowledged and is still travelling to.
        if not self._preset_travel:
            await self._port.stop()
        self._speeds = (0.0, 0.0, 0.0)
        self._seeking = False
        self._zoom_start = None
        await self._idle(options)

    async def _move(self, speeds, options):
        await self._active()
        now = self._clock()
        stop_axis = any(old != 0 and new == 0 for old, new in zip(self._speeds, speeds))
        if (
            not stop_axis
            and now - self._last_update < options.camera_update_rate_limit / 1000
        ):
            return
        if options.simulate_variable_speed:
            emitted = []
            for axis, speed in enumerate(speeds):
                self._pulse[axis] += abs(speed)
                pulse = self._pulse[axis] >= 1 - 1e-12
                emitted.append(math.copysign(1.0, speed) if speed and pulse else 0.0)
                if pulse:
                    self._pulse[axis] -= 1
            speeds = tuple(emitted)
        # Cleared before sending: an uncertain move failure must not skip later Stop.
        self._preset_travel = False
        if speeds == (0.0, 0.0, 0.0):
            await self._port.stop()
        else:
            await self._port.move(*speeds)
        self._speeds = speeds
        self._last_update = now
        self._seeking = any(speeds)
        if speeds[2] > 0:
            self._zoomed = True

    def _velocity(self, box, mapping, options):
        x1, y1, x2, y2 = box
        sx, sy = mapping.scale_xy
        ox, oy = mapping.offset_xy
        x1, x2, y1, y2 = x1 * sx + ox, x2 * sx + ox, y1 * sy + oy, y2 * sy + oy
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 < x1 or y2 < y1:
            raise _LocalFailure("box")
        height, width = mapping.frame_size_hw
        dx, dy = width / 2 - (x1 + x2) / 2, height / 2 - (y1 + y2) / 2
        zoom_delta = min(x1, width - x2, y1, height - y2)
        edge = zoom_delta <= 1
        now = self._clock()
        if self._zoom_start is not None and (
            now - self._zoom_start >= 2 or (zoom_delta < options.dead_zone and not edge)
        ):
            self._zoom_start = None
        minimum = (
            max(options.minimum_camera_speed, 0.1)
            if options.simulate_variable_speed
            else options.minimum_camera_speed
        )
        for pid in self._pid:
            pid.tunings = (options.pid_kp, options.pid_ki, options.pid_kd)

        def boost(value):
            return math.copysign(max(abs(value), minimum), value) if value else 0.0

        if (
            abs(dx) <= options.dead_zone and abs(dy) <= options.dead_zone
        ) or self._zoom_start is not None:
            if (
                not options.zoom_if_able
                or not self._port.supports_zoom
                or (zoom_delta < options.dead_zone and not edge)
            ):
                return (0.0, 0.0, 0.0)
            if self._zoom_start is None:
                self._zoom_start = now
            z = (
                -minimum
                if edge
                else -boost(
                    self._pid[2](
                        abs(zoom_delta - options.dead_zone / 2) / max(width, height)
                    )
                )
            )
            return (0.0, 0.0, z)

        magnitude = max(abs(dx), abs(dy))
        x = (
            abs(dx) / magnitude * boost(self._pid[0](dx / width))
            if abs(dx) > options.dead_zone
            else 0.0
        )
        y = (
            abs(dy) / magnitude * boost(self._pid[1](dy / height))
            if abs(dy) > options.dead_zone
            else 0.0
        )
        x *= 1 if options.flip_x_movement else -1
        y *= 1 if options.flip_y_movement else -1
        if self._zoomed:
            x *= 0.5
            y *= 0.5
        return (x, y, 0.0)

    async def _process(self, predictions, options):
        async with self._serial:
            if self._pending_failure is not None:
                failure, self._pending_failure = self._pending_failure, None
                return failure
            try:
                boxes = boxes_of(predictions)
                count = len(boxes.xyxy)
                await self._ensure_port()
                if options.movement_type == "Go To Preset":
                    await self._active()
                    # Stop continuous motion only; GotoPreset replaces earlier travel.
                    if not self._preset_travel:
                        await self._port.stop()
                    self._seeking = False
                    self._speeds = (0.0, 0.0, 0.0)
                    if (
                        self._clock() - self._last_update
                        >= options.camera_update_rate_limit / 1000
                    ):
                        await self._port.preset(options.default_position_preset)
                        self._clear_state()
                        self._preset_travel = True
                        self._last_update = self._clock()
                    selected = select_rows(
                        predictions, RowSelection.of(count, indices=[])
                    )
                elif count == 0:
                    self._tracked = None
                    await self._stop(options)
                    selected = predictions
                else:
                    tracker_ids = [
                        row.get("tracker_id")
                        for row in (boxes.bboxes_metadata or [{}] * count)
                    ]
                    index = None
                    if options.follow_tracker and self._tracked is not None:
                        index = next(
                            (
                                i
                                for i, value in enumerate(tracker_ids)
                                if value == self._tracked
                            ),
                            None,
                        )
                    if index is None:
                        index = int(torch.argmax(boxes.confidence).item())
                    self._tracked = (
                        tracker_ids[index] if options.follow_tracker else None
                    )
                    selected = select_rows(
                        predictions, RowSelection.of(count, indices=[index])
                    )
                    placement = root_placement_from_metadata(boxes.image_metadata or {})
                    if placement is None:
                        raise _LocalFailure("root_geometry")
                    # Four scalars only: no full predictions/masks transfer or root-mask expansion.
                    box = boxes.xyxy[index].detach().tolist()
                    was_zooming = self._zoom_start is not None
                    speeds = self._velocity(box, placement.root, options)
                    if speeds[2] and not was_zooming:
                        # Match the deliberate stop-before-zoom transition.
                        self._preset_travel = False
                        await self._port.stop()
                        self._speeds = (0.0, 0.0, 0.0)
                        self._last_update = -math.inf
                    if any(speeds):
                        await self._move(speeds, options)
                    else:
                        await self._stop(options)
                return {"predictions": selected, "seeking": self._seeking}
            except BaseException as error:
                return self._record_failure("command", error)

    def quiesce(self) -> None:
        """Stop admissions, drain commands, cancel timers and reset tracking.

        Host runs must settle before an engine update. Call ``resume`` after
        that update. If Stop fails, the owner remains quiesced and raises.

        Raises:
            OnvifError: On camera failure or calls from the owner loop.
        """
        self._check_thread()
        with self._lifecycle:
            with self._admission:
                if self._closed:
                    raise OnvifError("ONVIF owner is closed")
                self._accepting = False
                future = asyncio.run_coroutine_threadsafe(self._reset(), self._loop)
            self._wait(future)

    async def _reset(self):
        async with self._serial:
            await self._cancel_timer()
            failure, self._pending_failure = self._pending_failure, None
            try:
                if self._port is not None:
                    await self._port.stop()
            except BaseException as error:
                failure = self._record_failure("reset Stop", error)
            self._clear_state()
            return failure

    def reset_tracking(self) -> None:
        """Stops admission until ``resume()``; an alias of ``quiesce()``.

        Retained for existing host callers; new code calls ``quiesce()``. Every
        ``process`` call fails until the host calls ``resume()``.

        Raises:
            OnvifError: As ``quiesce()``.
        """
        self.quiesce()

    def resume(self) -> None:
        """Admit commands after host reset and graph update.

        Raises:
            OnvifError: If the controller is already closed.
        """
        self._check_thread()
        with self._lifecycle, self._admission:
            if self._closed:
                raise OnvifError("ONVIF owner is closed")
            self._accepting = True

    async def _shutdown(self):
        failure = await self._reset()
        if self._port is not None:
            try:
                await self._port.close()
            except BaseException as error:
                close_failure = self._record_failure("transport close", error)
                failure = failure or close_failure
            self._port = None
        try:
            await self._loop.shutdown_asyncgens()
        except BaseException as error:
            shutdown_failure = self._record_failure("async generator shutdown", error)
            failure = failure or shutdown_failure
        try:
            await self._loop.shutdown_default_executor()
        except BaseException as error:
            # CPython starts a helper thread before joining its default executor.
            # If that start fails, the host must join the executor with the loop
            # still running so worker callbacks can finish. close() owns this join.
            self._executor_to_join = self._loop._default_executor
            shutdown_failure = self._record_failure("executor shutdown", error)
            failure = failure or shutdown_failure
        return failure

    def close(self) -> None:
        """Drain, cancel timers, attempt Stop/transport close and join exactly once.

        No total timeout is imposed: resources are never torn out from under an
        admitted operation. A failing Stop still permits transport cleanup. If
        executor joining fails, the loop remains available for a later close retry.

        Raises:
            OnvifError: If Stop, a delayed command or shutdown failed.
        """
        self._check_thread()
        with self._lifecycle:
            with self._admission:
                if self._closed:
                    return
                self._accepting = False
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            try:
                self._wait(future)
            finally:
                # Interruption of a host wait must not retire live transports.
                # Settle the admitted shutdown before stopping its event loop.
                failure = future.result()
                join_failure = None
                if self._executor_to_join is not None:
                    try:
                        self._executor_to_join.shutdown(wait=True)
                    except BaseException as error:
                        join_failure = self._record_failure("executor join", error)
                    if join_failure is not None:
                        # Keep the loop/executor available for a later close retry.
                        # Raise outside the handler so no driver exception is chained.
                        failure = failure or join_failure
                        raise OnvifError(failure.message) from None
                    self._executor_to_join = None
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()
                self._loop.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except OnvifError:
                # Retained in failures; never replace a foreground exception.
                pass


class OnvifTrackingBlock(Block):
    """Follow tensor predictions through a borrowed, host-owned controller.

    Args:
        onvif_controller: Explicit owner. Required only for enabled use.
        disable_sinks: Host policy; disabled calls pass through without Secrets/I/O.
    """

    type = "v2/onvif_sink@v1"
    metadata = {
        "name": "PTZ Tracking (ONVIF)",
        "section": "industrial",
        "license": "Apache-2.0",
        "local_only": True,
        "port_of": "roboflow_core/onvif_sink@v1",
    }
    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="predictions",
            description="Selected target, or aligned empty rows in preset mode.",
        ),
        "seeking": Output(
            BOOLEAN_KIND,
            description="Whether the last acknowledged command has nonzero velocity.",
        ),
    }

    class Params(BlockParams):
        predictions: Ref(*PREDICTION_KINDS) = Field(
            description="Detection or segmentation predictions to track."
        )
        camera_ip: StrictStr | Ref(STRING_KIND) = Field(
            description="Camera host, matching the injected owner."
        )
        camera_port: StrictInt | Ref(INTEGER_KIND) = Field(
            default=80, description="ONVIF service port."
        )
        camera_username: StrictStr | Ref(STRING_KIND) = Field(
            description="Camera account, matching the owner."
        )
        camera_password: StrictStr | Ref(STRING_KIND, SECRET_KIND) = Field(
            description="Camera password or Secret, matching the owner."
        )
        camera_scheme: Literal["http", "https"] = Field(
            default="http",
            description="ONVIF device-service scheme; HTTPS verifies certificates.",
        )
        request_timeout: StrictFloat | StrictInt = Field(
            default=10.0, gt=0, description="Per-I/O HTTP timeout, matching the owner."
        )
        profile_index: StrictInt = Field(
            default=0, ge=0, description="Media profile index, matching the owner."
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Pass through without camera resources or Secret reveal.",
        )
        movement_type: Literal["Follow", "Go To Preset"] = Field(
            default="Follow", description="Track detections or move to a named preset."
        )
        default_position_preset: StrictStr | Ref(STRING_KIND) = Field(
            default="", description="Named preset for direct or idle movement."
        )
        zoom_if_able: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Zoom after centering the target."
        )
        follow_tracker: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description="Retain tracker ID, including zero, until it disappears.",
        )
        dead_zone: StrictInt | Ref(INTEGER_KIND) = Field(
            default=50, description="Root-frame pixel tolerance around center."
        )
        camera_update_rate_limit: StrictInt | Ref(INTEGER_KIND) = Field(
            default=250, description="Minimum milliseconds between non-stop commands."
        )
        flip_y_movement: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True, description="Reverse vertical motion."
        )
        flip_x_movement: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Reverse horizontal motion."
        )
        move_to_position_after_idle_seconds: (
            StrictFloat | StrictInt | Ref(FLOAT_KIND, INTEGER_KIND)
        ) = Field(default=0, description="Idle preset delay; zero disables.")
        pid_kp: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.25, description="PID proportional gain."
        )
        pid_ki: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.0, description="PID integral gain."
        )
        pid_kd: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=1.0, description="PID derivative gain."
        )
        minimum_camera_speed: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.05, description="Minimum nonzero normalized speed, 0..1."
        )
        simulate_variable_speed: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Emit signed full-speed pulses to approximate speed.",
        )

    def __init__(
        self,
        *,
        onvif_controller: Optional[OnvifController] = None,
        disable_sinks: bool = False,
    ):
        self._controller = onvif_controller
        self._disabled = disable_sinks

    @classmethod
    def discover_work_operations(cls, params):
        """Declare external camera requests for workload inspection."""
        return [WorkOperation.EXTERNAL_REQUEST]

    @classmethod
    def discover_restrictions(cls, params):
        """Declare the same hosted-runtime LAN restriction as V1."""
        return [
            RuntimeRestriction(
                code="requires_lan_access_to_device",
                severity=Severity.HARD,
                note="Block requires LAN access to a PTZ camera.",
                applies_to_runtimes=[
                    Runtime.HOSTED_SERVERLESS,
                    Runtime.DEDICATED_DEPLOYMENT,
                ],
            )
        ]

    def run(
        self,
        *,
        predictions,
        camera_ip,
        camera_port=80,
        camera_username,
        camera_password,
        camera_scheme="http",
        request_timeout=10.0,
        profile_index=0,
        disable_sink=False,
        movement_type="Follow",
        default_position_preset="",
        zoom_if_able=False,
        follow_tracker=True,
        dead_zone=50,
        camera_update_rate_limit=250,
        flip_y_movement=True,
        flip_x_movement=False,
        move_to_position_after_idle_seconds=0,
        pid_kp=0.25,
        pid_ki=0.0,
        pid_kd=1.0,
        minimum_camera_speed=0.05,
        simulate_variable_speed=False,
    ):
        """Apply one frame's settings; disabled calls return the input unchanged.

        Args:
            predictions: Tensor prediction carrier.
            camera_ip: Owner's camera host.
            camera_port: Owner's service port.
            camera_username: Owner's account.
            camera_password: Owner's password or Secret.
            camera_scheme: Owner's HTTP(S) scheme.
            request_timeout: Owner's per-I/O timeout.
            profile_index: Owner's media profile index.
            disable_sink: Per-step disable policy.
            movement_type: Follow or Go To Preset.
            default_position_preset: Named camera preset.
            zoom_if_able: Enable zoom.
            follow_tracker: Retain selected tracker IDs.
            dead_zone: Root pixel tolerance.
            camera_update_rate_limit: Non-stop command interval in milliseconds.
            flip_y_movement: Reverse vertical motion.
            flip_x_movement: Reverse horizontal motion.
            move_to_position_after_idle_seconds: Idle preset delay.
            pid_kp: Proportional gain.
            pid_ki: Integral gain.
            pid_kd: Derivative gain.
            minimum_camera_speed: Minimum nonzero speed.
            simulate_variable_speed: Enable pulsed speed simulation.

        Returns:
            Selected predictions and seeking state.

        Raises:
            OnvifError: If enabled without an owner or camera commands fail.
        """
        if self._disabled or disable_sink:
            return {"predictions": predictions, "seeking": False}
        if self._controller is None:
            raise OnvifError(
                'Enabled ONVIF requires a host-owned OnvifController in resources["onvif_controller"]; the host must close it'
            )
        target = OnvifTarget(
            camera_ip,
            camera_port,
            camera_username,
            camera_password,
            scheme=camera_scheme,
            timeout=request_timeout,
            profile_index=profile_index,
        )
        options = TrackingOptions(
            movement_type=movement_type,
            default_position_preset=default_position_preset,
            zoom_if_able=zoom_if_able,
            follow_tracker=follow_tracker,
            dead_zone=dead_zone,
            camera_update_rate_limit=camera_update_rate_limit,
            flip_y_movement=flip_y_movement,
            flip_x_movement=flip_x_movement,
            move_to_position_after_idle_seconds=move_to_position_after_idle_seconds,
            pid_kp=pid_kp,
            pid_ki=pid_ki,
            pid_kd=pid_kd,
            minimum_camera_speed=minimum_camera_speed,
            simulate_variable_speed=simulate_variable_speed,
        )
        result = self._controller.process(predictions, target=target, options=options)
        return result
