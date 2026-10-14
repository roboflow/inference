"""Host ports, input snapshot and cooldown delivery shared by the event blocks.

The blocks port V1 ``roboflow_core/roboflow_vision_events@v1`` and
``roboflow_core/vision_event_bundle@v1``. Both share one flow::

    disabled?     ─▶ error_status=False, V1's disabled message; no key, no port
    cooling?      ─▶ throttling_status=True (one window per step, as in V1)
    events:  cloud ─▶ api_key as Secret, or V1's missing-key error
             write_to_event_store ─▶ no key needed
    events:  valid event_type ─▶ missing default port fails the run now
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

Each block owns its Params, outputs, constructor and the skip checks. This
module keeps what both share: the ports and their offline defaults, the
event schema table, the input snapshot (``_detached_request``) and the
cooldown window start before delivery (``_start_window_and_deliver``).

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
from typing import Any, Callable, Dict, NoReturn, Optional, Protocol
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BackgroundEffects,
    Cooldown,
    EffectOutcome,
    deliver,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret

VISION_EVENTS_PORT_RESOURCE = "vision_events_port"

VISION_EVENT_BUNDLE_PORT_RESOURCE = "vision_event_bundle_port"

ADAPTER_MODULE = "inference.roboflow_workflows_plugin.v2.platform_events_adapter"

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
        """Write one bundle and refuse a name that is already taken.

        With hard links the refusal is atomic: of two concurrent writes of
        one name, exactly one wins. Without them (e.g. FAT removable media)
        the host adapter falls back to V1's check and then a replacing
        rename, so two concurrent writes of one custom ``file_name`` can race
        and one bundle can overwrite the other. Leave ``file_name`` unset for
        unique generated names, or serialize writes that share a custom name.

        Args:
            request: Detached inputs.
            target_directory: Output directory, prepared before.
            file_name: Custom bundle name, or None for a generated one.

        Returns:
            The bundle's event id and path.

        Raises:
            RuntimeError: When the host forbids local filesystem access.
            PlatformRequestError: When building the archive fails; the host
                reports a fixed diagnostic.
            ValueError: When the destination was taken or became unsafe, or
                the bundle is too large.
            OSError: When writing or publishing the file fails.
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
    def ensure_available(self) -> NoReturn:
        raise _offline(
            VISION_EVENTS_PORT_RESOURCE,
            needs="the Roboflow Vision Events API or a local event store",
            example="RoboflowVisionEventsAdapter()",
        )

    def submit(self, request: VisionEventRequest, **arguments: Any) -> str:
        self.ensure_available()


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


def _cooldown_seconds(value: float) -> float:
    if math.isnan(value):
        raise ValueError("cooldown_seconds must be a number, got NaN")

    # A selector bypasses the ge=0 check; V1 treated a negative value as 0.
    seconds = max(float(value), 0.0)

    return seconds


def _event_fields(event_type: str) -> Dict[str, str]:
    fields = EVENT_DATA_FIELDS.get(event_type)
    if fields is None:
        raise ValueError(
            f"Unknown event_type {event_type!r}; expected one of "
            f"{', '.join(EVENT_DATA_FIELDS)}"
        )
    return fields


def _detached_request(
    *,
    input_image: Optional[ImageData],
    output_image: Optional[ImageData],
    predictions: Any,
    event_type: str,
    solution: Optional[str],
    custom_metadata: Dict[str, Any],
    event_values: Dict[str, Any],
) -> VisionEventRequest:
    # Copies every input, so a background delivery never reads a workflow
    # buffer that a later frame reuses. event_values holds the schema
    # parameters by name; only those of event_type are kept, None dropped.
    fields = _event_fields(event_type)

    event_data = {
        name: event_values[parameter]
        for name, parameter in fields.items()
        if event_values[parameter] is not None
    }
    images = {
        name: None if image is None else image.copy(device="cpu")
        for name, image in (
            ("input_image", input_image),
            ("output_image", output_image),
        )
    }
    request = VisionEventRequest(
        **images,
        predictions=copy.deepcopy(predictions),
        event_type=event_type,
        solution=solution,
        event_data=copy.deepcopy(event_data),
        custom_metadata=copy.deepcopy(custom_metadata),
    )

    return request


def _start_window_and_deliver(
    effect: Callable[[], EffectOutcome],
    *,
    cooldown: Cooldown,
    cooldown_seconds: float,
    fire_and_forget: bool,
    sink_effects: Optional[BackgroundEffects],
    label: str,
) -> Optional[EffectOutcome]:
    # As in V1, the window starts once the event is ready, so a failed or
    # refused delivery also starts it. Another call may have started one
    # while this call prepared: None tells the caller it is throttled.
    if not cooldown.try_start(cooldown_seconds):
        return None

    # (False, BACKGROUND_MESSAGE) when sink_effects admitted the effect.
    # Otherwise the effect ran in this thread (now, or inline because
    # sink_effects was full) or was refused.
    outcome = deliver(
        effect,
        fire_and_forget=fire_and_forget,
        sink_effects=sink_effects,
        label=label,
    )

    return outcome
