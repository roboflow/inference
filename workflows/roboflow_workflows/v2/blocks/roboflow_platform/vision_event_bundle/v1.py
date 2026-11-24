"""Write one event as a local ``.tar.gz`` bundle per invocation.

Canonical type: ``roboflow_core/vision_event_bundle@v1``.
"""

from typing import Any, Dict, Literal, Optional, Union
from pydantic import Field, StrictBool
from roboflow_workflows.v2.blocks.roboflow_platform._shared import vision_events
from roboflow_workflows.v2.blocks.roboflow_platform._shared.vision_events import (
    EventBundle,
    VisionEventBundlePort,
    _cooldown_seconds,
    _OfflineVisionEventBundlePort,
    _start_window_and_deliver,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    BackgroundEffects,
    Cooldown,
    EffectOutcome,
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
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

THROTTLED_MESSAGE = "Sink cooldown applies"


class VisionEventBundleBlock(Block):
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

    type = "roboflow_core/vision_event_bundle@v1"
    aliases = ("VisionEventBundle",)
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
        "bundle_path": Output(
            STRING_KIND, description="Written bundle path; empty unless it ran now."
        ),
    }
    metadata = {
        "name": "Write Vision Event Bundle",
        "port_of": "roboflow_core/vision_event_bundle@v1",
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
        self._port = vision_event_bundle_port
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
        target_directory: str,
        file_name: Optional[str],
        solution: Optional[str],
    ) -> Dict[str, Any]:
        """Write one bundle unless disabled or cooling down.

        Args:
            input_image: Original image, or None.
            output_image: Display image, or None.
            predictions: Native predictions attached as annotations, or None.
            event_type: Vision Events schema name.
            custom_metadata: Flat metadata attached to the event.
            fire_and_forget: Hand the write to the host's ``sink_effects``.
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
            target_directory: Output directory.
            file_name: Custom bundle name, or None for a generated one.
            solution: Use case stored as useCaseId, or None.

        Returns:
            ``error_status``, ``throttling_status``, ``event_id``,
            ``bundle_path`` and ``message``.

        Raises:
            RuntimeError: When the host forbids local filesystem access.
            ValueError: On an unsafe or taken destination, an unknown
                ``event_type`` or a NaN ``cooldown_seconds``. As in V1 this
                fails the run, also with ``fire_and_forget``.
            PlatformUnavailableError: Without a ``vision_event_bundle_port``.
        """
        if self._disable_sinks or disable_sink:
            message = disabled_sink_message(by_policy=self._disable_sinks)
            return self._outcome(message=message)

        seconds = _cooldown_seconds(cooldown_seconds)
        if self._cooldown.cooling(seconds):
            return self._outcome(throttling_status=True, message=THROTTLED_MESSAGE)

        # V1 checks the destination before dispatch, so misconfiguration
        # fails loudly here instead of vanishing into a background failure.
        self._port.prepare_destination(
            target_directory=target_directory, file_name=file_name
        )

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

        def write() -> EffectOutcome:
            bundle = self._port.create_bundle(
                request, target_directory=target_directory, file_name=file_name
            )
            if not isinstance(bundle, EventBundle):
                raise TypeError("vision_event_bundle_port returned no EventBundle")
            completed.update(event_id=bundle.event_id, bundle_path=bundle.bundle_path)
            return False, "Vision event bundle written successfully"

        delivered = _start_window_and_deliver(
            write,
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
            return self._outcome(
                message="Vision event bundle written in background task"
            )

        # Only an effect that succeeded in this thread filled completed.
        outcome = self._outcome(
            error_status=error_status,
            message=message,
            event_id=completed.get("event_id", ""),
            bundle_path=completed.get("bundle_path", ""),
        )

        return outcome

    @staticmethod
    def _outcome(
        *,
        message: str,
        error_status: bool = False,
        throttling_status: bool = False,
        event_id: str = "",
        bundle_path: str = "",
    ) -> Dict[str, Any]:
        outcome = {
            "error_status": error_status,
            "throttling_status": throttling_status,
            "event_id": event_id,
            "bundle_path": bundle_path,
            "message": message,
        }

        return outcome
