"""Vision event submission and local event bundles through explicit host ports.

Ports of V1 ``roboflow_core/roboflow_vision_events@v1`` and
``roboflow_core/vision_event_bundle@v1``. Both blocks share one flow::

    disabled?     ─▶ error_status=False, V1's disabled message; no key, no port
    cooling?      ─▶ throttling_status=True (one window per step, as in V1)
    events:  cloud ─▶ api_key as Secret, or V1's missing-key error
             write_to_event_store ─▶ no key needed
    bundle:  port.prepare_destination ─▶ a refused destination fails the run now
    schema fields ─▶ event_data (camelCase, None dropped); inputs detached
    window starts ─▶ fire_and_forget and sink_effects ─▶ background, ids ""
                     else run now ─▶ event_id (and bundle_path)

    resource                  Protocol               host adapter
    vision_events_port        VisionEventsPort       RoboflowVisionEventsAdapter
    vision_event_bundle_port  VisionEventBundlePort  VisionEventBundleAdapter

Both adapters live in ``inference.roboflow_workflows_plugin.v2.
platform_events_adapter``. Only an adapter reveals a credential. The bundle
port writes a local ``.tar.gz`` and nothing else: no mover, no upload.

Images are copied to CPU memory and predictions and metadata are deep-copied
before delivery, so a background effect never reads a reused workflow buffer.

Differences from V1:

- ``fire_and_forget`` uses the host's ``sink_effects`` (V1 used FastAPI
  ``BackgroundTasks`` or a thread pool). Without it the call runs now.
- Failure messages are fixed host diagnostics; V1 echoed response bodies.
- An unknown ``event_type`` fails the run. V1 sent an event without data.
- The local event store key is an explicit adapter ``Secret``. V1 read
  ``EVENT_INGESTION_API_KEY`` from the environment on every call.
- A bundle ``file_name`` literal is checked when the step runs, not when the
  workflow compiles.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Union

from pydantic import Field, StrictBool
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import api_key_for_dispatch
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    BackgroundEffects,
    Cooldown,
    EffectOutcome,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

VISION_EVENTS_PORT_RESOURCE = "vision_events_port"
"""Constructor resource name of ``VisionEventsPort`` consumers."""

VISION_EVENT_BUNDLE_PORT_RESOURCE = "vision_event_bundle_port"
"""Constructor resource name of ``VisionEventBundlePort`` consumers."""

ADAPTER_MODULE = "inference.roboflow_workflows_plugin.v2.platform_events_adapter"

THROTTLED_MESSAGE = "Sink cooldown applies"

EventType = Literal[
    "quality_check", "inventory_count", "safety_alert", "custom", "operator_feedback"
]

# V1's eventData keys per schema: API camelCase name -> step parameter.
EVENT_DATA_FIELDS = {
    "quality_check": {"result": "qc_result", "externalId": "external_id"},
    "inventory_count": {
        "location": "location",
        "itemCount": "item_count",
        "itemType": "item_type",
        "externalId": "external_id",
    },
    "safety_alert": {
        "alertType": "alert_type",
        "severity": "severity",
        "description": "alert_description",
        "externalId": "external_id",
    },
    "custom": {"value": "custom_value", "externalId": "external_id"},
    "operator_feedback": {"relatedEventId": "related_event_id", "feedback": "feedback"},
}


@dataclass(frozen=True)
class VisionEventRequest:
    """One event's detached inputs, shared by submission and bundles.

    Args:
        input_image: Original image, or None. Annotations refer to it.
        output_image: Display image, or None. Stored before the input image.
        predictions: Native predictions attached to the one image entry.
        event_type: Vision Events schema name, e.g. ``"quality_check"``.
        solution: Use case id. Optional for a bundle only.
        event_data: Schema fields with V1's camelCase keys, None dropped.
        custom_metadata: Flat metadata attached to the event.
    """

    input_image: Optional[ImageData]
    output_image: Optional[ImageData]
    predictions: Any
    event_type: str
    solution: Optional[str]
    event_data: Dict[str, Any]
    custom_metadata: Dict[str, Any]


@dataclass(frozen=True)
class EventBundle:
    """A bundle written by ``VisionEventBundlePort.create_bundle``.

    Args:
        event_id: ``eventId`` stored in the bundle's payload.
        bundle_path: Absolute path of the published ``.tar.gz``.
    """

    event_id: str
    bundle_path: str


class VisionEventsPort(Protocol):
    """Send an event to the Roboflow Vision Events API or a local event store."""

    def submit(
        self,
        request: VisionEventRequest,
        *,
        api_key: Optional[Secret],
        write_to_event_store: bool,
        event_store_url: str,
    ) -> str:
        """Upload the images, if any, then create one event.

        Args:
            request: Detached inputs.
            api_key: Roboflow API key; None with ``write_to_event_store``.
            write_to_event_store: Post to ``<event_store_url>/v2/events``,
                with embedded images, instead of the Roboflow API.
            event_store_url: Base URL of the local Event Ingestion Service.

        Returns:
            The event id: the client-generated ``eventId`` in the cloud, the
            service's ``id`` locally (``""`` when it reports none, as in V1).

        Raises:
            PlatformRequestError: When an upload or the event request fails.
        """


class VisionEventBundlePort(Protocol):
    """Write event bundles into a host-approved local directory."""

    def prepare_destination(
        self, *, target_directory: str, file_name: Optional[str]
    ) -> None:
        """Create the directory and refuse a destination the write would refuse.

        Args:
            target_directory: Output directory.
            file_name: Custom bundle name, or None for a generated one.

        Raises:
            RuntimeError: When the host forbids local filesystem access.
            ValueError: When the name is unsafe, the directory is outside the
                host's allowed root or not writable, or the file exists.
        """

    def create_bundle(
        self,
        request: VisionEventRequest,
        *,
        target_directory: str,
        file_name: Optional[str],
    ) -> EventBundle:
        """Write one bundle without replacing an existing file.

        Args:
            request: Detached inputs.
            target_directory: Output directory, prepared before.
            file_name: Custom bundle name, or None for a generated one.

        Returns:
            The bundle's event id and path.

        Raises:
            PlatformRequestError: When encoding or publication fails.
            ValueError: When the destination was taken or became unsafe.
        """


def _offline(resource: str, *, needs: str, example: str) -> PlatformUnavailableError:
    # Names this family's own adapter; ports.platform_unavailable names the
    # R1 adapter, which implements neither port.
    error = PlatformUnavailableError(
        f"this step needs {needs}, but the host provided no '{resource}' "
        f"resource. Pass one when creating the session, e.g. "
        f"resources={{'{resource}': {example}}} from {ADAPTER_MODULE}."
    )

    return error


class _OfflineVisionEventsPort:
    def submit(self, request: VisionEventRequest, **arguments: Any) -> str:
        raise _offline(
            VISION_EVENTS_PORT_RESOURCE,
            needs="the Roboflow Vision Events API or a local event store",
            example="RoboflowVisionEventsAdapter()",
        )


class _OfflineVisionEventBundlePort:
    def prepare_destination(self, **arguments: Any) -> None:
        raise _offline(
            VISION_EVENT_BUNDLE_PORT_RESOURCE,
            needs="local filesystem access",
            example="VisionEventBundleAdapter(allow_access_to_file_system=True)",
        )

    def create_bundle(
        self, request: VisionEventRequest, **arguments: Any
    ) -> EventBundle:
        self.prepare_destination()


class _EventParams(BlockParams):
    input_image: Optional[Ref(IMAGE_KIND, allow_none=True)] = Field(
        default=None, description="Original image; annotations refer to it."
    )
    output_image: Optional[Ref(IMAGE_KIND, allow_none=True)] = Field(
        default=None,
        description="Display image, e.g. a visualization; stored first.",
    )
    predictions: Optional[
        Ref(
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
            CLASSIFICATION_PREDICTION_KIND,
            allow_none=True,
        )
    ] = Field(default=None, description="Predictions attached as annotations.")
    event_type: EventType | Ref(STRING_KIND) = Field(
        description="Event schema: quality_check, inventory_count, safety_alert, "
        "custom or operator_feedback."
    )
    custom_metadata: Dict[str, Union[str, int, float, bool, Ref()]] | Ref(
        DICTIONARY_KIND
    ) = Field(default_factory=dict, description="Flat metadata attached to the event.")
    fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
        default=True,
        description="Hand delivery to the host's sink_effects; run now without it.",
    )
    disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
        default=False, description="Skip this step without sending or writing."
    )
    cooldown_seconds: float | Ref(FLOAT_KIND, INTEGER_KIND) = Field(
        default=1.0,
        ge=0,
        description="Least seconds between two events of this step; 0 disables.",
    )
    external_id: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="Correlation id; not sent with operator_feedback."
    )
    qc_result: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="quality_check result: pass or fail."
    )
    location: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="inventory_count location."
    )
    item_count: Optional[int] | Ref(INTEGER_KIND) = Field(
        default=None, description="inventory_count number of items; 0 is sent."
    )
    item_type: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="inventory_count item type."
    )
    alert_type: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="safety_alert type, e.g. no_hardhat."
    )
    severity: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="safety_alert severity: low, medium or high."
    )
    alert_description: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="safety_alert description."
    )
    custom_value: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="custom event value."
    )
    related_event_id: Optional[str] | Ref(STRING_KIND) = Field(
        default=None, description="operator_feedback: id of the reviewed event."
    )
    feedback: Optional[str] | Ref(STRING_KIND) = Field(
        default=None,
        description="operator_feedback: correct, incorrect or inconclusive.",
    )


_OUTPUTS = {
    "error_status": Output(
        BOOLEAN_KIND,
        description="True when delivery failed or full sink_effects refused it.",
    ),
    "throttling_status": Output(
        BOOLEAN_KIND, description="True when the step's cooldown skipped it."
    ),
    "event_id": Output(
        STRING_KIND, description="Created event id; empty unless it ran now."
    ),
    "message": Output(STRING_KIND, description="Human-readable outcome."),
}


def _cooldown_seconds(value: float) -> float:
    if math.isnan(value):
        raise ValueError("cooldown_seconds must be a number, got NaN")

    # A selector bypasses the ge=0 check; V1 treated a negative value as 0.
    seconds = max(float(value), 0.0)

    return seconds


def _detached_request(arguments: Dict[str, Any]) -> VisionEventRequest:
    event_type = arguments["event_type"]
    fields = EVENT_DATA_FIELDS.get(event_type)
    if fields is None:
        raise ValueError(
            f"Unknown event_type {event_type!r}; expected one of "
            f"{', '.join(EVENT_DATA_FIELDS)}"
        )

    event_data = {
        name: arguments[parameter]
        for name, parameter in fields.items()
        if arguments[parameter] is not None
    }
    images = {
        name: None if arguments[name] is None else arguments[name].copy(device="cpu")
        for name in ("input_image", "output_image")
    }
    request = VisionEventRequest(
        **images,
        predictions=copy.deepcopy(arguments["predictions"]),
        event_type=event_type,
        solution=arguments["solution"],
        event_data=copy.deepcopy(event_data),
        custom_metadata=copy.deepcopy(arguments["custom_metadata"]),
    )

    return request


class _EventSink(Block):
    """Disabling, cooldown and delivery shared by both blocks."""

    outputs = _OUTPUTS
    background_message: str
    empty_outcome: Dict[str, str] = {"event_id": ""}

    def __init__(
        self, *, disable_sinks: bool, sink_effects: Optional[BackgroundEffects]
    ) -> None:
        self._disable_sinks = disable_sinks
        self._sink_effects = sink_effects
        self._cooldown = Cooldown()

    def _outcome(
        self,
        *,
        message: str,
        error_status: bool = False,
        throttling_status: bool = False,
        **completed: str,
    ) -> Dict[str, Any]:
        outcome = {
            "error_status": error_status,
            "throttling_status": throttling_status,
            **self.empty_outcome,
            **completed,
            "message": message,
        }

        return outcome

    def _skipped(self, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._disable_sinks or arguments["disable_sink"]:
            message = disabled_sink_message(by_policy=self._disable_sinks)
            return self._outcome(message=message)

        if self._cooldown.cooling(_cooldown_seconds(arguments["cooldown_seconds"])):
            return self._outcome(throttling_status=True, message=THROTTLED_MESSAGE)

        return None

    def _deliver(
        self,
        arguments: Dict[str, Any],
        effect: Callable[[], EffectOutcome],
        completed: Dict[str, str],
    ) -> Dict[str, Any]:
        # As in V1, the window starts once the event is ready, so a failed or
        # refused delivery also starts it. Another call may have started one
        # while this call prepared.
        if not self._cooldown.try_start(
            _cooldown_seconds(arguments["cooldown_seconds"])
        ):
            return self._outcome(throttling_status=True, message=THROTTLED_MESSAGE)

        error_status, message = deliver(
            effect,
            fire_and_forget=arguments["fire_and_forget"],
            sink_effects=self._sink_effects,
            label=self.type,
        )
        if (error_status, message) == (False, BACKGROUND_MESSAGE):
            return self._outcome(message=self.background_message)

        # The effect ran in this thread (now, or inline because sink_effects
        # was full) or was refused. Only a successful effect fills completed.
        outcome = self._outcome(error_status=error_status, message=message, **completed)

        return outcome


class RoboflowVisionEventsBlock(_EventSink):
    """Send one event with images and predictions per invocation.

    Args:
        vision_events_port: Host adapter; the default refuses with guidance.
        api_key: Roboflow API key (str or Secret); the local event store
            does not need it.
        disable_sinks: Host policy skipping every sink.
        sink_effects: Host-owned background runner for fire_and_forget.
    """

    type = "v2/roboflow_vision_events@v1"
    aliases = ("RoboflowVisionEvents",)
    metadata = {
        "name": "Roboflow Vision Events",
        "port_of": "roboflow_core/roboflow_vision_events@v1",
    }
    background_message = "Vision event sent in background task"

    class Params(_EventParams):
        solution: str | Ref(STRING_KIND) = Field(
            description="Use case the event belongs to."
        )
        write_to_event_store: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Post to a local Event Ingestion Service, with embedded "
            "images, instead of the Roboflow API; no API key needed.",
        )
        event_store_url: str | Ref(STRING_KIND) = Field(
            default="http://localhost:8001",
            description="Local Event Ingestion Service base URL.",
        )

    def __init__(
        self,
        *,
        vision_events_port: VisionEventsPort = _OfflineVisionEventsPort(),
        api_key: Optional[Union[str, Secret]] = None,
        disable_sinks: bool = False,
        sink_effects: Optional[BackgroundEffects] = None,
    ) -> None:
        super().__init__(disable_sinks=disable_sinks, sink_effects=sink_effects)
        self._port = vision_events_port
        self._api_key = api_key

    def run(self, **arguments: Any) -> Dict[str, Any]:
        """Send one event unless disabled or cooling down.

        Args:
            **arguments: Resolved ``Params`` values of one invocation.

        Returns:
            ``error_status``, ``throttling_status``, ``event_id`` and
            ``message``.

        Raises:
            ValueError: Without an API key in cloud mode, or on an unknown
                ``event_type``.
            SecretUnavailableError: When the API key is an unavailable secret.
        """
        skipped = self._skipped(arguments)
        if skipped is not None:
            return skipped

        local = arguments["write_to_event_store"]
        api_key = None
        if not local:
            api_key = api_key_for_dispatch(self._api_key, block_name="VisionEvents")
        request = _detached_request(arguments)
        completed = {}

        def submit() -> EffectOutcome:
            event_id = self._port.submit(
                request,
                api_key=api_key,
                write_to_event_store=local,
                event_store_url=arguments["event_store_url"],
            )
            if not isinstance(event_id, str):
                raise TypeError("vision_events_port returned a non-str event id")
            completed["event_id"] = event_id
            if local:
                return False, "Event written to local event store successfully"
            return False, "Vision event sent successfully"

        outcome = self._deliver(arguments, submit, completed)

        return outcome


class VisionEventBundleBlock(_EventSink):
    """Write one event as a local ``.tar.gz`` bundle per invocation.

    The bundle keeps V1's format version 1 (``payload.json`` plus
    ``images/<uuid>.jpg``), so an existing file mover and uploader can carry
    it to ``POST /vision-events/bundle``. This block never moves or uploads.

    Args:
        vision_event_bundle_port: Host adapter; the default refuses with
            guidance.
        disable_sinks: Host policy skipping every sink.
        sink_effects: Host-owned background runner for fire_and_forget.
    """

    type = "v2/vision_event_bundle@v1"
    aliases = ("VisionEventBundle",)
    outputs = {
        **_OUTPUTS,
        "bundle_path": Output(
            STRING_KIND, description="Written bundle path; empty unless it ran now."
        ),
    }
    metadata = {
        "name": "Write Vision Event Bundle",
        "port_of": "roboflow_core/vision_event_bundle@v1",
    }
    background_message = "Vision event bundle written in background task"
    empty_outcome = {"event_id": "", "bundle_path": ""}

    class Params(_EventParams):
        target_directory: str | Ref(STRING_KIND) = Field(
            description="Output directory; created when missing. The host may "
            "restrict it to an allowed root."
        )
        file_name: Optional[str] | Ref(STRING_KIND) = Field(
            default=None,
            description="Bundle name replacing event_<UTC time>_<eventId>.tar.gz; "
            ".tar.gz is appended when absent. Letters, digits, '.', '_' and '-', "
            "starting with a letter or digit, at most 200 characters. Must be "
            "unique: an existing file fails the step.",
        )
        solution: Optional[str] | Ref(STRING_KIND) = Field(
            default=None,
            description="Use case stored as useCaseId; leave unset to let the "
            "uploader supply it.",
        )

    def __init__(
        self,
        *,
        vision_event_bundle_port: VisionEventBundlePort = (
            _OfflineVisionEventBundlePort()
        ),
        disable_sinks: bool = False,
        sink_effects: Optional[BackgroundEffects] = None,
    ) -> None:
        super().__init__(disable_sinks=disable_sinks, sink_effects=sink_effects)
        self._port = vision_event_bundle_port

    def run(self, **arguments: Any) -> Dict[str, Any]:
        """Write one bundle unless disabled or cooling down.

        Args:
            **arguments: Resolved ``Params`` values of one invocation.

        Returns:
            ``error_status``, ``throttling_status``, ``event_id``,
            ``bundle_path`` and ``message``.

        Raises:
            RuntimeError: When the host forbids local filesystem access.
            ValueError: On an unsafe or taken destination, or an unknown
                ``event_type``. As in V1 this fails the run, also with
                ``fire_and_forget``.
            PlatformUnavailableError: Without a ``vision_event_bundle_port``.
        """
        skipped = self._skipped(arguments)
        if skipped is not None:
            return skipped

        destination = {
            "target_directory": arguments["target_directory"],
            "file_name": arguments["file_name"],
        }
        # V1 checks the destination before dispatch, so misconfiguration
        # fails loudly here instead of vanishing into a background failure.
        self._port.prepare_destination(**destination)
        request = _detached_request(arguments)
        completed = {}

        def write() -> EffectOutcome:
            bundle = self._port.create_bundle(request, **destination)
            if not isinstance(bundle, EventBundle):
                raise TypeError("vision_event_bundle_port returned no EventBundle")
            completed.update(event_id=bundle.event_id, bundle_path=bundle.bundle_path)
            return False, "Vision event bundle written successfully"

        outcome = self._deliver(arguments, write, completed)

        return outcome
