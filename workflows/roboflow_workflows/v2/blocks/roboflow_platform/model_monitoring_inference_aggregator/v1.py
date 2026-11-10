"""Frequency-gated Model Monitoring reports without an inference dependency.

The engine owns one block instance per step/session, across invocations and
sources. Each instance owns its buffer; processing resets discard it. Timing
belongs to ``model_monitoring_cache`` and is keyed only by the explicit
``unique_aggregator_key``, as in V1. Supply a shared host WorkflowsCache for V1
cross-session timing; the default is instance-local. Sharing a key shares time,
never predictions. Different models compete for one representative per class,
including ties (first collected model wins), exactly as V1.

Reports detach before workspace lookup. One report may be pending while new
observations accumulate. The caller owns the optional BackgroundEffects runner;
without it, reporting is synchronous. See the example README for failure,
namespace budget, reset and host drain/close policies.
"""

from datetime import datetime
from math import isnan
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Protocol, Union
from pydantic import Field, StrictBool, StrictInt
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import (
    WorkspacePort,
    api_key_for_dispatch,
)
from roboflow_workflows.v2.blocks.sinks.runtime import (
    FULL_MESSAGE,
    BackgroundEffects,
)
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


MODEL_MONITORING_PORT_RESOURCE = "model_monitoring_port"

_CACHE_PREFIX = (
    "workflows:steps_cache:" "roboflow_core/model_monitoring_inference_aggregator@v1:"
)

SKIPPED_MESSAGE = "Not in reporting range, skipping report. (Ok)"

SUCCESS_MESSAGE = "Data sent successfully"

FAILURE_MESSAGE = (
    "Error while uploading inference data. Model monitoring report failed."
)

PENDING_MESSAGE = "Report pending; collecting for the next window. (Ok)"

WORKSPACE_FAILURE_MESSAGE = "Model monitoring workspace lookup failed."

UNAVAILABLE_MESSAGE = (
    'Model monitoring requires resources={"model_monitoring_port": '
    "RoboflowModelMonitoringAdapter()} from "
    "inference.roboflow_workflows_plugin.v2.platform_monitoring_adapter."
)

DISABLED_MESSAGE = "Sink was disabled by workflow execution policy"

_PREDICTION_KINDS = (
    CLASSIFICATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)

class ModelMonitoringPort(WorkspacePort, Protocol):
    """Host operations for Model Monitoring; only the host reveals credentials."""

    def send_report(
        self,
        *,
        api_key: Secret,
        workspace_id: str,
        predictions: List[Dict[str, Any]],
    ) -> None:
        """Send scalar representative rows with host report metadata.

        Args:
            api_key: Credential to reveal inside the host operation only.
            workspace_id: Workspace from ``get_workspace``.
            predictions: Rows containing model_id, class_name, confidence,
                inference_id and model_type, using V1 wire names.

        Raises:
            PlatformRequestError: When the report fails.
        """

class MonitoringCache(Protocol):
    """Caller-owned cache for V1 reporting times, not prediction buffers.

    Implementations must support concurrent get/set when shared by sessions.
    Their separate operations do not make a reporting window atomic across
    instances or processes; V1 likewise provides no distributed claim.
    """

    def get(self, key: str) -> Optional[str]:
        """Read an ISO datetime, or None for an uninitialized window.

        Args:
            key: V1 aggregator reporting-time key.

        Returns:
            Stored ISO datetime, or None.
        """

    def set(self, key: str, value: str) -> None:
        """Store a reporting time without expiry, as V1.

        Args:
            key: V1 aggregator reporting-time key.
            value: ISO datetime from the injected clock.
        """

class _LocalMonitoringCache:
    def __init__(self) -> None:
        self._values: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._values.get(key)

    def set(self, key: str, value: str) -> None:
        self._values[key] = value

class _OfflineMonitoringPort:
    def get_workspace(self, *, api_key: Secret) -> str:
        raise PlatformUnavailableError(UNAVAILABLE_MESSAGE)

    def send_report(self, **arguments: Any) -> None:
        raise PlatformUnavailableError(UNAVAILABLE_MESSAGE)

def format_predictions_for_model_monitoring(
    predictions: Any, *, model_id: str, inference_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Snapshot native predictions into V1 monitoring rows, without mutation.

    Single-label scores are thresholded before rounding to four decimals and
    sorted descending. Multilabel scores stay unrounded and class-id ordered.
    Detection rows preserve confidence and use zero when it is absent.

    Args:
        predictions: One native classification, detection or keypoint tuple.
        model_id: Explicit platform model identifier, as in V1.
        inference_id: Explicit V2 producer id, overriding prediction metadata.

    Returns:
        Newly allocated scalar rows, empty for missing or empty predictions.
    """
    if predictions is None:
        return []

    def row(name: str, score: float, metadata: dict) -> Dict[str, Any]:
        result = {
            "model_id": model_id,
            "class_name": name,
            "confidence": score,
            "inference_id": (
                inference_id
                if inference_id is not None
                else metadata.get("inference_id", "")
            ),
            "model_type": metadata.get("prediction_type", ""),
        }

        return result

    results = []
    if isinstance(predictions, ClassificationPrediction):
        metadata_list = predictions.images_metadata or []
        for index in range(len(predictions.class_id)):
            metadata = metadata_list[index] if index < len(metadata_list) else {}
            metadata = metadata or {}
            names = metadata.get("class_names") or {}
            threshold = metadata.get("classification_confidence_threshold")
            confidence = predictions.confidence[index].detach().cpu()
            per_class = []
            for class_id in sorted(names):
                score = float(confidence[class_id])
                if threshold is not None and score < threshold:
                    continue

                per_class.append(row(names[class_id], round(score, 4), metadata))
            per_class.sort(key=lambda item: item["confidence"], reverse=True)
            results.extend(per_class)
    elif isinstance(predictions, MultiLabelClassificationPrediction):
        metadata = predictions.image_metadata or {}
        names = metadata.get("class_names") or {}
        threshold = metadata.get("classification_confidence_threshold")
        confidence = predictions.confidence.detach().cpu()
        for class_id in sorted(names):
            score = float(confidence[class_id])
            if threshold is not None and score < threshold:
                continue

            results.append(row(names[class_id], score, metadata))
    else:
        # As V1's tensor splitter, monitoring requires the bbox component
        # of a keypoint tuple; per-keypoint scores are not box confidences.
        detections = predictions[1] if isinstance(predictions, tuple) else predictions
        if detections is None:
            raise ValueError("Model monitoring requires detections for keypoint rows")

        metadata = detections.image_metadata or {}
        names = metadata.get("class_names") or {}
        class_ids = detections.class_id.detach().cpu().tolist()
        confidences = (
            detections.confidence.detach().cpu().tolist()
            if detections.confidence is not None
            else [0.0] * len(class_ids)
        )
        for class_id, confidence in zip(class_ids, confidences):
            results.append(
                row(names.get(int(class_id), ""), float(confidence), metadata)
            )

    return results

class ModelMonitoringInferenceAggregatorBlock(Block):
    """Collect one most-confident representative per class and report periodically.

    Args:
        model_monitoring_port: Host API adapter; omitted means every
            non-disabled call raises PlatformUnavailableError.
        model_monitoring_cache: Host-owned shared timing cache for V1 parity;
            omitted means this instance's private timing cache.
        model_monitoring_clock: Datetime clock; defaults to local wall time as V1.
        api_key: Host credential, kept as a Secret until host dispatch.
        disable_sinks: Skip validation, collection and all platform operations.
        sink_effects: Optional caller-owned runner for fire-and-forget reports.
        model_monitoring_max_entries: Maximum combined model slots and model/class
            slots in a collection window. Overflow rejects the entire collection.
            Defaults to 4096; no silent eviction or retained row history.

    Collection and completion are serialized; host I/O runs outside the lock.
    Different instances have independent buffers even when their cache is shared.
    Only scalar snapshots are retained, never input tensors.
    """

    type = "roboflow_core/model_monitoring_inference_aggregator@v1"
    aliases = ("ModelMonitoringInferenceAggregator",)
    accepts_empty = True
    metadata = {
        "name": "Model Monitoring Inference Aggregator",
        "port_of": "roboflow_core/model_monitoring_inference_aggregator@v1",
        "requires_rf_key": True,
    }
    outputs = {
        "error_status": Output(BOOLEAN_KIND, description="Whether reporting failed."),
        "message": Output(STRING_KIND, description="Collection or reporting outcome."),
    }

    class Params(BlockParams):
        predictions: Ref(*_PREDICTION_KINDS) = Field(
            description="Native predictions to collect; empty predictions may flush.",
            examples=["$steps.model.predictions"],
        )
        model_id: str | Ref(STRING_KIND) = Field(
            min_length=1,
            description="Platform model identifier recorded verbatim in reports.",
            examples=["project/3", "$inputs.model_id"],
        )
        frequency: StrictInt | Ref(INTEGER_KIND) = Field(
            default=5,
            ge=1,
            description="Minimum whole seconds between reporting attempts.",
            examples=[5, "$inputs.frequency"],
        )
        unique_aggregator_key: str = Field(
            min_length=1,
            description="Stable timing-cache key; unique per intended aggregation session.",
            examples=["camera-monitor-1"],
        )
        inference_id: Optional[str] | Ref(STRING_KIND) = Field(
            default=None,
            description="V2 producer id; overrides the prediction metadata when provided.",
            examples=["$steps.model.inference_id"],
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description="Use caller-owned sink_effects when provided; otherwise wait for completion.",
            examples=[True, False],
        )

    def __init__(
        self,
        *,
        model_monitoring_port: ModelMonitoringPort = _OfflineMonitoringPort(),
        model_monitoring_cache: Optional[MonitoringCache] = None,
        model_monitoring_clock: Callable[[], datetime] = datetime.now,
        api_key: Optional[Union[str, Secret]] = None,
        disable_sinks: bool = False,
        sink_effects: Optional[BackgroundEffects] = None,
        model_monitoring_max_entries: int = 4096,
    ) -> None:
        if (
            type(model_monitoring_max_entries) is not int
            or model_monitoring_max_entries < 1
        ):
            raise ValueError("model_monitoring_max_entries must be an integer >= 1")

        self._effects = sink_effects
        self._max_entries = model_monitoring_max_entries
        self._entries = 0
        self._pending = False
        self._port = model_monitoring_port
        self._cache = (
            model_monitoring_cache
            if model_monitoring_cache is not None
            else _LocalMonitoringCache()
        )
        self._clock = model_monitoring_clock
        self._api_key = api_key
        self._disabled = disable_sinks
        self._predictions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = Lock()

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Declare the platform model referenced by the reports.

        Args:
            params: Validated step parameters.

        Returns:
            One model dependency; no weights are loaded by this block.
        """
        resources = [
            DependentResource(
                resource_type="roboflow_platform_model", identifier=params.model_id
            )
        ]

        return resources

    def run(
        self,
        *,
        predictions: Any,
        model_id: str,
        frequency: int,
        unique_aggregator_key: str,
        inference_id: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> Dict[str, Any]:
        """Collect, then flush when the reporting interval has elapsed.

        Args:
            predictions: Native predictions or None to drive an empty flush.
            model_id: Model identifier preserved in each row.
            frequency: Positive integer number of seconds.
            unique_aggregator_key: Stable host-cache timing key.
            inference_id: Explicit V2 id, or None to use metadata.
            fire_and_forget: Use the caller-owned runner if supplied.

        Returns:
            Error status and a credential-free outcome message.

        Raises:
            ValueError: Invalid parameters, collected NaN score, or a collection
                exceeding the namespace budget. Rejected collections change no state.
            SecretUnavailableError: When a replayed credential has no value.
            PlatformUnavailableError: When no host port was supplied, on every
                non-disabled call before collection, or when a synchronous
                report reaches an unavailable configured port.
            PlatformRequestError: When workspace lookup fails.
        """
        if self._disabled:
            return {"error_status": False, "message": DISABLED_MESSAGE}

        if type(frequency) is not int or frequency < 1:
            raise ValueError("frequency must be an integer greater than or equal to 1")

        if not model_id or not unique_aggregator_key:
            raise ValueError("model_id and unique_aggregator_key must not be empty")

        api_key = api_key_for_dispatch(
            self._api_key, block_name="ModelMonitoringInferenceAggregator"
        )
        # A missing adapter is a host setup error: report it on the first call,
        # not at the first due report or only to a background runner.
        if isinstance(self._port, _OfflineMonitoringPort):
            raise PlatformUnavailableError(UNAVAILABLE_MESSAGE)

        with self._lock:
            rows = format_predictions_for_model_monitoring(
                predictions, model_id=model_id, inference_id=inference_id
            )
            # NaN breaks the ordering required for an associative maximum.
            # Reject explicitly rather than silently changing V1 sort behavior.
            if any(isnan(row["confidence"]) for row in rows):
                raise ValueError("Model monitoring does not accept NaN confidence")

            if predictions:
                existing = self._predictions.get(model_id, {})
                added = len({row["class_name"] for row in rows} - existing.keys())
                added += int(model_id not in self._predictions)
                if self._entries + added > self._max_entries:
                    raise ValueError("Model monitoring namespace budget exceeded")

                model_rows = self._predictions.setdefault(model_id, {})
                self._entries += added
                for row in rows:
                    self._keep_best(model_rows, row)

            if self._pending:
                return {"error_status": False, "message": PENDING_MESSAGE}

            key = f"{_CACHE_PREFIX}{unique_aggregator_key}:last_report_time"
            now = self._clock()
            last = self._cache.get(key)
            if last is None:
                self._cache.set(key, now.isoformat())
                last = now.isoformat()

            if int((now - datetime.fromisoformat(last)).total_seconds()) < frequency:
                return {"error_status": False, "message": SKIPPED_MESSAGE}

            class_rows = {}
            for model_rows in self._predictions.values():
                for row in model_rows.values():
                    self._keep_best(class_rows, row)
            representatives = list(class_rows.values())
            self._predictions = {}
            self._entries = 0
            self._pending = True

        if fire_and_forget and self._effects is not None:
            # Admission can run inline under backpressure: never hold our lock.
            try:
                outcome = self._effects.deliver(
                    lambda: self._background_report(
                        representatives, api_key=api_key, key=key
                    ),
                    label=self.type,
                )
            except Exception:
                with self._lock:
                    self._pending = False
                raise

            if outcome == (True, FULL_MESSAGE):
                with self._lock:
                    self._pending = False
        else:
            outcome = self._report(
                representatives, api_key=api_key, key=key, background=False
            )

        return {"error_status": outcome[0], "message": outcome[1]}

    @staticmethod
    def _keep_best(rows: Dict[str, Dict[str, Any]], row: Dict[str, Any]) -> None:
        name = row["class_name"]
        previous = rows.get(name)
        if previous is None or row["confidence"] > previous["confidence"]:
            rows[name] = row

    def _background_report(
        self, rows: List[Dict[str, Any]], *, api_key: Secret, key: str
    ) -> tuple[bool, str]:
        try:
            outcome = self._report(rows, api_key=api_key, key=key, background=True)
        except Exception:
            # Cache/clock adapters also must not leak exception text to the runner.
            outcome = (True, FAILURE_MESSAGE)

        return outcome

    def _report(
        self, rows: List[Dict[str, Any]], *, api_key: Secret, key: str, background: bool
    ) -> tuple[bool, str]:
        advance_time = False
        try:
            try:
                workspace_id = self._port.get_workspace(api_key=api_key)
            except PlatformUnavailableError:
                if not background:
                    raise

                return True, UNAVAILABLE_MESSAGE
            except Exception:
                if not background:
                    raise

                return True, WORKSPACE_FAILURE_MESSAGE

            try:
                self._port.send_report(
                    api_key=api_key, workspace_id=workspace_id, predictions=rows
                )
            except PlatformUnavailableError:
                if not background:
                    raise

                return True, UNAVAILABLE_MESSAGE
            except Exception:
                outcome = (True, FAILURE_MESSAGE)
            else:
                outcome = (False, SUCCESS_MESSAGE)
            advance_time = True

            return outcome
        finally:
            # A new report cannot start before the completion timestamp is stored.
            # The host's metadata/network operation has already returned.
            with self._lock:
                try:
                    if advance_time:
                        self._cache.set(key, self._clock().isoformat())
                finally:
                    self._pending = False
