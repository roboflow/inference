"""Native OBB model block with the same explicit task identity as the server."""

from typing import List, Literal, Optional, Union

from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.oriented import (
    native_obb_from_inference,
    native_obb_to_tensor,
)
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_batch_of_sv_detections,
)
from roboflow_workflows.environment import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    ORIENTED_OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
    ImageInputField,
    RoboflowModelField,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    Severity,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
    roboflow_platform_model,
)
from roboflow_workflows.prototypes.models_provider import ModelsProvider

from inference_sdk import InferenceConfiguration, InferenceHTTPClient

# Native OBB runs on API v1 only (/infer/oriented_object_detection); the hosted per-task model APIs (detect.roboflow.com
# and the like) are API v0 and serve no oriented-box model, so a remote step needs a self-hosted inference server.
HOSTED_REMOTE_UNSUPPORTED_RESTRICTION = RuntimeRestriction(
    code="oriented_object_detection_hosted_remote_unsupported",
    severity=Severity.HARD,
    note=(
        "Block raises ValueError as a remote step with WORKFLOWS_REMOTE_API_TARGET=hosted: no hosted API serves "
        "native oriented-box models; run it locally or against a self-hosted inference server."
    ),
    applies_to_step_execution_modes=[StepExecutionMode.REMOTE],
    applies_to_configuration={"WORKFLOWS_REMOTE_API_TARGET": "hosted"},
)


class BlockManifest(WorkflowBlockManifest):
    """Native OBB inputs and output kind, without ordinary detection compatibility."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Oriented Object Detection Model",
            "version": "v1",
            "block_type": "model",
            "short_description": "Predict native rectangles and class-specific fronts.",
            "license": "Apache-2.0",
        }
    )
    type: Literal["roboflow_core/roboflow_oriented_object_detection_model@v1"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    model_id: Union[Selector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = RoboflowModelField
    confidence: float = Field(
        default=0.5, ge=0, le=1, description="Minimum sigmoid class confidence."
    )
    max_detections: int = Field(
        default=300, gt=0, description="Maximum native detections per image."
    )
    class_filter: Optional[List[str]] = Field(
        default=None, description="Optional frozen class names to retain."
    )

    @classmethod
    def get_compatible_task_types(cls):
        return ["oriented-object-detection"]

    @classmethod
    def get_parameters_accepting_batches(cls):
        return ["images"]

    @classmethod
    def describe_outputs(cls):
        return [
            OutputDefinition(
                name="predictions", kind=[ORIENTED_OBJECT_DETECTION_PREDICTION_KIND]
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls):
        return ">=1.3.0,<2.0.0"

    def discover_work_operations(self) -> List[WorkOperation]:
        """Declare native model inference.

        Returns:
            The work performed by this block.
        """
        return [WorkOperation.MODEL_INFERENCE]

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [HOSTED_REMOTE_UNSUPPORTED_RESTRICTION]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Declare that remote steps need a self-hosted inference server.

        Args:
            ignore_environment_restrictions: Whether to omit host restrictions.

        Returns:
            The hosted-remote restriction, for this step.
        """
        return actual_restrictions_of(
            declared=[HOSTED_REMOTE_UNSUPPORTED_RESTRICTION],
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        """Declare the selected native model.

        Returns:
            The model dependency, retaining selectors for discovery.
        """
        return [roboflow_platform_model(model_id=self.model_id)]


class RoboflowOrientedObjectDetectionModelBlockV1(WorkflowBlock):
    """Execute the native model and preserve geometry/front in Workflow transport."""

    def __init__(self, model_manager: ModelsProvider, api_key, step_execution_mode):
        self._model_manager = model_manager
        self._api_key = api_key
        self._mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls):
        return ["model_manager", "api_key", "step_execution_mode"]

    @classmethod
    def get_manifest(cls):
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id,
        confidence,
        max_detections,
        class_filter,
    ):
        """Infer and attach the exact image/crop lineage to native detections.

        Args:
            images: Workflow input images or crops.
            model_id: Native registered model version.
            confidence: Numeric sigmoid threshold.
            max_detections: Per-image output cap.
            class_filter: Optional accepted class names.

        Returns:
            Native prediction output for each input image.
        """
        if self._mode is StepExecutionMode.LOCAL:
            self._model_manager.add_model(model_id=model_id, api_key=self._api_key)
            responses = self._model_manager.run_oriented_object_detection(
                model_id=model_id,
                images=[i.to_inference_format(numpy_preferred=True) for i in images],
                api_key=self._api_key,
                confidence=confidence,
                max_detections=max_detections,
                class_filter=class_filter,
            )
        elif self._mode is StepExecutionMode.REMOTE:
            if WORKFLOWS_REMOTE_API_TARGET == "hosted":
                raise ValueError(
                    "Oriented object detection cannot run as a remote step with WORKFLOWS_REMOTE_API_TARGET=hosted: "
                    "no hosted API serves native oriented-box models. Run the step locally, or set "
                    "WORKFLOWS_REMOTE_API_TARGET to target a self-hosted inference server (LOCAL_INFERENCE_API_URL)."
                )
            client = InferenceHTTPClient(
                api_url=LOCAL_INFERENCE_API_URL, api_key=self._api_key
            )
            client.select_api_v1()
            client.configure(
                InferenceConfiguration(
                    confidence_threshold=confidence,
                    max_detections=max_detections,
                    class_filter=class_filter,
                    client_downsizing_disabled=True,
                )
            )
            responses = client.infer(
                [i.base64_image for i in images], model_id=model_id
            )
            responses = responses if isinstance(responses, list) else [responses]
        else:
            raise ValueError("Unknown native OBB step execution mode")
        detections = [native_obb_from_inference(r) for r in responses]
        detections = attach_parents_coordinates_to_batch_of_sv_detections(
            images=images, predictions=detections
        )
        if ENABLE_TENSOR_DATA_REPRESENTATION:
            detections = [native_obb_to_tensor(d) for d in detections]
        return [{"predictions": d} for d in detections]


class NativeOBBVisualizationManifest(WorkflowBlockManifest):
    """Draw native footprints and fronts without axis-aligned replacement."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Oriented Box Visualization",
            "version": "v1",
            "block_type": "visualization",
            "short_description": "Draw oriented footprints and class-specific front edges.",
            "license": "Apache-2.0",
        }
    )
    type: Literal["roboflow_core/oriented_box_visualization@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image in the same frame as the detections."
    )
    predictions: Selector(kind=[ORIENTED_OBJECT_DETECTION_PREDICTION_KIND]) = Field(
        description="Native rectangles or ordered quadrilaterals."
    )
    thickness: int = Field(default=2, gt=0, description="Outline thickness in pixels.")

    @classmethod
    def describe_outputs(cls):
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls):
        return ">=1.3.0,<2.0.0"

    def discover_work_operations(self) -> List[WorkOperation]:
        """Declare native footprint visualization.

        Returns:
            The work performed by this block.
        """
        return [WorkOperation.VISUALIZATION]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Declare unrestricted execution.

        Args:
            ignore_environment_restrictions: Whether to omit host restrictions.

        Returns:
            A complete empty restriction declaration.
        """
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        """Declare that no external resources are required.

        Returns:
            A complete empty resource declaration.
        """
        return []


class NativeOBBVisualizationBlockV1(WorkflowBlock):
    """Render the current ordered corner carrier in either representation."""

    @classmethod
    def get_manifest(cls):
        return NativeOBBVisualizationManifest

    def run(self, image, predictions, thickness):
        """Draw a copy of the input image and return it.

        Args:
            image: Workflow image.
            predictions: Native prediction carrier.
            thickness: Outline width.

        Returns:
            Annotated image output.
        """
        from roboflow_workflows.core_steps.common.oriented import (
            _native_obb_to_sv,
            annotate_native_obb,
        )

        scene = image.numpy_image.copy()
        annotated = annotate_native_obb(
            scene, _native_obb_to_sv(predictions), thickness=thickness
        )
        return {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated
            )
        }
