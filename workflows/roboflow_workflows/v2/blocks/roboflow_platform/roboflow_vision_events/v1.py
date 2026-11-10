"""Send one event with images and predictions per invocation.

Canonical type: ``roboflow_core/roboflow_vision_events@v1``.
"""

from typing import Any, Dict, Literal, Optional, Union
from pydantic import Field, StrictBool
from roboflow_workflows.v2.blocks.roboflow_platform.ports import api_key_for_dispatch
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    BackgroundEffects,
    Cooldown,
    EffectOutcome,
    disabled_sink_message,
)
from roboflow_workflows.v2.blocks.roboflow_platform._shared import vision_events
from roboflow_workflows.v2.blocks.roboflow_platform._shared.vision_events import (
    VisionEventsPort,
    _cooldown_seconds,
    _event_fields,
    _OfflineVisionEventsPort,
    _start_window_and_deliver,
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

THROTTLED_MESSAGE = "Sink cooldown applies"


class RoboflowVisionEventsBlock(Block):
    """Send one event with images and predictions per invocation.

    Args:
        vision_events_port: Host adapter; the default refuses with guidance.
        api_key: Roboflow API key (str or Secret); the local event store
            does not need it.
        disable_sinks: Host policy skipping every sink.
        sink_effects: Host-owned background runner for fire_and_forget.
    """

    type = "roboflow_core/roboflow_vision_events@v1"
    aliases = ("RoboflowVisionEvents",)
    outputs = {
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
    metadata = {
        "name": "Roboflow Vision Events",
        "port_of": "roboflow_core/roboflow_vision_events@v1",
    }

    class Params(BlockParams):
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
        event_type: Literal[
            "quality_check",
            "inventory_count",
            "safety_alert",
            "custom",
            "operator_feedback",
        ] | Ref(STRING_KIND) = Field(
            description="Event schema: quality_check, inventory_count, safety_alert, "
            "custom or operator_feedback."
        )
        custom_metadata: Dict[str, Union[str, int, float, bool, Ref()]] | Ref(
            DICTIONARY_KIND
        ) = Field(
            default_factory=dict, description="Flat metadata attached to the event."
        )
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
        self._port = vision_events_port
        self._api_key = api_key
        self._disable_sinks = disable_sinks
        self._sink_effects = sink_effects
        self._cooldown = Cooldown()

    def run(
        self,
        *,
        input_image: Optional[ImageData],
        output_image: Optional[ImageData],
        predictions: Any,
        event_type: str,
        custom_metadata: Dict[str, Any],
        fire_and_forget: bool,
        disable_sink: bool,
        cooldown_seconds: float,
        external_id: Optional[str],
        qc_result: Optional[str],
        location: Optional[str],
        item_count: Optional[int],
        item_type: Optional[str],
        alert_type: Optional[str],
        severity: Optional[str],
        alert_description: Optional[str],
        custom_value: Optional[str],
        related_event_id: Optional[str],
        feedback: Optional[str],
        solution: str,
        write_to_event_store: bool,
        event_store_url: str,
    ) -> Dict[str, Any]:
        """Send one event unless disabled or cooling down.

        Args:
            input_image: Original image, or None.
            output_image: Display image, or None.
            predictions: Native predictions attached as annotations, or None.
            event_type: Vision Events schema name.
            custom_metadata: Flat metadata attached to the event.
            fire_and_forget: Hand delivery to the host's ``sink_effects``.
            disable_sink: Skip this step.
            cooldown_seconds: Least seconds between two events of this step.
            external_id: Correlation id; not sent with operator_feedback.
            qc_result: quality_check result.
            location: inventory_count location.
            item_count: inventory_count number of items.
            item_type: inventory_count item type.
            alert_type: safety_alert type.
            severity: safety_alert severity.
            alert_description: safety_alert description.
            custom_value: custom event value.
            related_event_id: operator_feedback reviewed event id.
            feedback: operator_feedback verdict.
            solution: Use case the event belongs to.
            write_to_event_store: Post to a local event store instead of the
                Roboflow API.
            event_store_url: Local event store base URL.

        Returns:
            ``error_status``, ``throttling_status``, ``event_id`` and
            ``message``.

        Raises:
            ValueError: Without an API key in cloud mode, on an unknown
                ``event_type`` or a NaN ``cooldown_seconds``.
            SecretUnavailableError: When the API key is an unavailable secret.
            PlatformUnavailableError: When the active step has no vision_events_port,
                before snapshotting or background admission.
        """
        if self._disable_sinks or disable_sink:
            message = disabled_sink_message(by_policy=self._disable_sinks)
            return self._outcome(message=message)

        seconds = _cooldown_seconds(cooldown_seconds)
        if self._cooldown.cooling(seconds):
            return self._outcome(throttling_status=True, message=THROTTLED_MESSAGE)

        api_key = None
        if not write_to_event_store:
            api_key = api_key_for_dispatch(self._api_key, block_name="VisionEvents")
        # Missing host setup must fail here, before copying inputs or admission.
        # Check event_type before reporting missing host setup.
        if isinstance(self._port, _OfflineVisionEventsPort):
            _event_fields(event_type)
            self._port.ensure_available()

        # Called through the module: tests patch _detached_request there.
        request = vision_events._detached_request(
            input_image=input_image,
            output_image=output_image,
            predictions=predictions,
            event_type=event_type,
            solution=solution,
            custom_metadata=custom_metadata,
            event_values={
                "external_id": external_id,
                "qc_result": qc_result,
                "location": location,
                "item_count": item_count,
                "item_type": item_type,
                "alert_type": alert_type,
                "severity": severity,
                "alert_description": alert_description,
                "custom_value": custom_value,
                "related_event_id": related_event_id,
                "feedback": feedback,
            },
        )
        completed = {}

        def submit() -> EffectOutcome:
            event_id = self._port.submit(
                request,
                api_key=api_key,
                write_to_event_store=write_to_event_store,
                event_store_url=event_store_url,
            )
            if not isinstance(event_id, str):
                raise TypeError("vision_events_port returned a non-str event id")
            completed["event_id"] = event_id
            if write_to_event_store:
                return False, "Event written to local event store successfully"
            return False, "Vision event sent successfully"

        delivered = _start_window_and_deliver(
            submit,
            cooldown=self._cooldown,
            cooldown_seconds=seconds,
            fire_and_forget=fire_and_forget,
            sink_effects=self._sink_effects,
            label=self.type,
        )
        # None: another call started the window while this one prepared.
        if delivered is None:
            return self._outcome(throttling_status=True, message=THROTTLED_MESSAGE)

        error_status, message = delivered
        if (error_status, message) == (False, BACKGROUND_MESSAGE):
            return self._outcome(message="Vision event sent in background task")

        # Only an effect that succeeded in this thread filled completed.
        outcome = self._outcome(
            error_status=error_status,
            message=message,
            event_id=completed.get("event_id", ""),
        )

        return outcome

    @staticmethod
    def _outcome(
        *,
        message: str,
        error_status: bool = False,
        throttling_status: bool = False,
        event_id: str = "",
    ) -> Dict[str, Any]:
        outcome = {
            "error_status": error_status,
            "throttling_status": throttling_status,
            "event_id": event_id,
            "message": message,
        }

        return outcome
