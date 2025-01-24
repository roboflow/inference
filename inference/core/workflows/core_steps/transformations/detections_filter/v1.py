# # from typing import Any, Dict, List, Literal, Optional, Type, Union

# # import supervision as sv
# # from pydantic import ConfigDict, Field

# # from inference.core.workflows.core_steps.common.query_language.entities.operations import (
# #     AllOperationsType,
# #     OperationDefinition,
# # )
# # from inference.core.workflows.core_steps.transformations.detections_transformation.v1 import (
# #     execute_transformation,
# # )
# # from inference.core.workflows.execution_engine.entities.base import (
# #     Batch,
# #     OutputDefinition,
# # )
# # from inference.core.workflows.execution_engine.entities.types import (
# #     INSTANCE_SEGMENTATION_PREDICTION_KIND,
# #     KEYPOINT_DETECTION_PREDICTION_KIND,
# #     OBJECT_DETECTION_PREDICTION_KIND,
# #     Selector,
# # )
# # from inference.core.workflows.prototypes.block import (
# #     BlockResult,
# #     WorkflowBlock,
# #     WorkflowBlockManifest,
# # )

# # SHORT_DESCRIPTION = "Conditionally filter out model predictions."

# # OPERATIONS_EXAMPLE = [
# #     {
# #         "type": "DetectionsFilter",
# #         "filter_operation": {
# #             "type": "StatementGroup",
# #             "statements": [
# #                 {
# #                     "type": "BinaryStatement",
# #                     "left_operand": {
# #                         "type": "DynamicOperand",
# #                         "operations": [
# #                             {
# #                                 "type": "ExtractDetectionProperty",
# #                                 "property_name": "class_name",
# #                             }
# #                         ],
# #                     },
# #                     "comparator": {"type": "in (Sequence)"},
# #                     "right_operand": {
# #                         "type": "DynamicOperand",
# #                         "operand_name": "classes",
# #                     },
# #                 },
# #             ],
# #         },
# #     }
# # ]


# # class BlockManifest(WorkflowBlockManifest):
# #     model_config = ConfigDict(
# #         json_schema_extra={
# #             "name": "Detections Filter",
# #             "version": "v1",
# #             "short_description": SHORT_DESCRIPTION,
# #             "long_description": SHORT_DESCRIPTION,
# #             "license": "Apache-2.0",
# #             "block_type": "transformation",
# #         }
# #     )
# #     type: Literal["roboflow_core/detections_filter@v1", "DetectionsFilter"]
# #     predictions: Selector(
# #         kind=[
# #             OBJECT_DETECTION_PREDICTION_KIND,
# #             INSTANCE_SEGMENTATION_PREDICTION_KIND,
# #             KEYPOINT_DETECTION_PREDICTION_KIND,
# #         ]
# #     ) = Field(
# #         description="Reference to detection-like predictions",
# #         examples=["$steps.object_detection_model.predictions"],
# #     )
# #     operations: List[AllOperationsType] = Field(
# #         description="Definition of filtering operations", examples=[OPERATIONS_EXAMPLE]
# #     )
# #     operations_parameters: Dict[
# #         str,
# #         Selector(),
# #     ] = Field(
# #         description="References to additional parameters that may be provided in runtime to parametrise operations",
# #         examples=[
# #             {
# #                 "classes": "$inputs.classes",
# #             }
# #         ],
# #         default_factory=lambda: {},
# #     )

# #     @classmethod
# #     def get_parameters_accepting_batches(cls) -> List[str]:
# #         return ["predictions"]

# #     @classmethod
# #     def describe_outputs(cls) -> List[OutputDefinition]:
# #         return [
# #             OutputDefinition(
# #                 name="predictions",
# #                 kind=[
# #                     OBJECT_DETECTION_PREDICTION_KIND,
# #                     INSTANCE_SEGMENTATION_PREDICTION_KIND,
# #                     KEYPOINT_DETECTION_PREDICTION_KIND,
# #                 ],
# #             )
# #         ]

# #     @classmethod
# #     def get_execution_engine_compatibility(cls) -> Optional[str]:
# #         return ">=1.3.0,<2.0.0"


# # class DetectionsFilterBlockV1(WorkflowBlock):

# #     @classmethod
# #     def get_manifest(cls) -> Type[WorkflowBlockManifest]:
# #         return BlockManifest

# #     def run(
# #         self,
# #         predictions: Batch[sv.Detections],
# #         operations: List[OperationDefinition],
# #         operations_parameters: Dict[str, Any],
# #     ) -> BlockResult:
# #         return execute_transformation(
# #             predictions=predictions,
# #             operations=operations,
# #             operations_parameters=operations_parameters,
# #         )

# # detections_transformation/v1.py

# from typing import Any, Dict, List, Literal, Optional, Type, Union, Callable

# import supervision as sv
# from pydantic import ConfigDict, Field

# from inference.core.workflows.core_steps.common.query_language.entities.operations import (
#     AllOperationsType,
#     OperationDefinition,
# )
# from inference.core.workflows.core_steps.transformations.detections_transformation.v1 import (
#     execute_transformation,
# )
# from inference.core.workflows.execution_engine.entities.base import (
#     Batch,
#     BlockResult,
# )
# from inference.core.workflows.execution_engine.entities.types import (
#     INSTANCE_SEGMENTATION_PREDICTION_KIND,
#     KEYPOINT_DETECTION_PREDICTION_KIND,
#     OBJECT_DETECTION_PREDICTION_KIND,
#     Selector,
# )
# from inference.core.workflows.prototypes.block import (
#     WorkflowBlock,
#     WorkflowBlockManifest,
# )

# SHORT_DESCRIPTION = "Conditionally filter out model predictions."

# OPERATIONS_EXAMPLE = [
#     {
#         "type": "DetectionsFilter",
#         "filter_operation": {
#             "type": "StatementGroup",
#             "statements": [
#                 {
#                     "type": "BinaryStatement",
#                     "left_operand": {
#                         "type": "DynamicOperand",
#                         "operations": [
#                             {
#                                 "type": "ExtractDetectionProperty",
#                                 "property_name": "class_name",
#                             }
#                         ],
#                     },
#                     "comparator": {"type": "in (Sequence)"},
#                     "right_operand": {
#                         "type": "DynamicOperand",
#                         "operand_name": "classes",
#                     },
#                 },
#             ],
#         },
#     }
# ]


# class BlockManifest(WorkflowBlockManifest):
#     model_config = ConfigDict(
#         json_schema_extra={
#             "name": "Detections Filter",
#             "version": "v1",
#             "short_description": SHORT_DESCRIPTION,
#             "long_description": SHORT_DESCRIPTION,
#             "license": "Apache-2.0",
#             "block_type": "transformation",
#         }
#     )
#     type: Literal["roboflow_core/detections_filter@v1", "DetectionsFilter"]
#     predictions: Selector(
#         kind=[
#             OBJECT_DETECTION_PREDICTION_KIND,
#             INSTANCE_SEGMENTATION_PREDICTION_KIND,
#             KEYPOINT_DETECTION_PREDICTION_KIND,
#         ]
#     ) = Field(
#         description="Reference to detection-like predictions",
#         examples=["$steps.object_detection_model.predictions"],
#     )
#     operations: List[AllOperationsType] = Field(
#         description="Definition of filtering operations", examples=[OPERATIONS_EXAMPLE]
#     )
#     operations_parameters: Dict[str, Any] = Field(
#         description="References to additional parameters that may be provided in runtime to parametrize operations",
#         default_factory=lambda: {},
#     )

#     @classmethod
#     def get_parameters_accepting_batches(cls) -> List[str]:
#         return ["predictions"]

#     @classmethod
#     def describe_outputs(cls):
#         from inference.core.workflows.execution_engine.entities.base import OutputDefinition
#         return [
#             OutputDefinition(
#                 name="predictions",
#                 kind=[
#                     OBJECT_DETECTION_PREDICTION_KIND,
#                     INSTANCE_SEGMENTATION_PREDICTION_KIND,
#                     KEYPOINT_DETECTION_PREDICTION_KIND,
#                 ],
#             )
#         ]

#     @classmethod
#     def get_execution_engine_compatibility(cls) -> Optional[str]:
#         return ">=1.3.0,<2.0.0"


# class DetectionsFilterBlockV1(WorkflowBlock):
#     """Block that filters detections (e.g., by class, confidence, size, or dynamic zones)."""

#     @classmethod
#     def get_manifest(cls) -> Type[WorkflowBlockManifest]:
#         return BlockManifest

#     def run(
#         self,
#         predictions: Batch[sv.Detections],
#         operations: List[OperationDefinition],
#         operations_parameters: Dict[str, Any],
#     ) -> BlockResult:
#         """
#         Runs the DetectionsFilter transformation on a batch of detections.
        
#         Args:
#             predictions (Batch[sv.Detections]): Batch of predictions to filter.
#             operations (List[OperationDefinition]): List of filter operations.
#             operations_parameters (Dict[str, Any]): Additional runtime parameters (zones, classes, etc.).
        
#         Returns:
#             BlockResult: The filtered results.
#         """
#         return execute_transformation(
#             predictions=predictions,
#             operations=operations,
#             operations_parameters=operations_parameters,
#             custom_comparators={
#                 "(Detection) zone check": self._compare_zone_membership
#             }
#         )

#     def _compare_zone_membership(
#         self,
#         detection: sv.Detection,
#         zone_params: Dict[str, Any]
#     ) -> bool:
#         """
#         Custom comparator that checks if a detection passes the 'in-zone' logic.
        
#         zone_params keys:
#             zone_detections: the detections used as zone bounding boxes
#             zone_classes: which classes define valid zone bounding boxes
#             restricted_classes: detection classes that must be within a zone
#             ignored_classes: detection classes that we do not filter by zone
#             in_zone_mode: 'must_be_in_zone' or 'must_be_out_of_zone'
#         """

#         # Get class name from detection
#         if hasattr(detection, "class_name"):
#             detection_class = detection.class_name
#         else:
#             detection_class = str(detection.class_id)

#         zone_classes = zone_params.get("zone_classes", [])
#         restricted_classes = zone_params.get("restricted_classes", [])
#         ignored_classes = zone_params.get("ignored_classes", [])
#         in_zone_mode = zone_params.get("in_zone_mode", "must_be_in_zone")
#         zone_detections: sv.Detections = zone_params.get("zone_detections", sv.Detections.empty())

#         # If ignored class, always pass
#         if detection_class in ignored_classes:
#             return True

#         # If not restricted, don't filter by zone
#         if detection_class not in restricted_classes:
#             return True

#         # Filter zone_detections to only those that match zone_classes
#         zone_boxes = [
#             z for z in zone_detections 
#             if getattr(z, "class_name", None) in zone_classes
#         ]

#         # Check if detection center is inside at least one zone bounding box
#         is_in_zone = False
#         cx, cy = detection.x, detection.y
#         for z in zone_boxes:
#             left  = z.x - z.width / 2
#             right = z.x + z.width / 2
#             top   = z.y - z.height / 2
#             bot   = z.y + z.height / 2

#             if (left <= cx <= right) and (top <= cy <= bot):
#                 is_in_zone = True
#                 break

#         # Decide pass/fail based on in_zone_mode
#         if in_zone_mode == "must_be_in_zone":
#             return is_in_zone
#         elif in_zone_mode == "must_be_out_of_zone":
#             return not is_in_zone
#         else:
#             return is_in_zone

from typing import Any, Dict, List, Literal, Optional, Type, Union, Callable

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
    OperationDefinition,
)
from inference.core.workflows.core_steps.transformations.detections_transformation.v1 import (
    execute_transformation,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    BlockResult,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Conditionally filter out model predictions."

OPERATIONS_EXAMPLE = [
    {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "class_name",
                            }
                        ],
                    },
                    "comparator": {"type": "in (Sequence)"},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "classes",
                    },
                },
            ],
        },
    }
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": SHORT_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["roboflow_core/detections_filter@v1", "DetectionsFilter"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    operations: List[AllOperationsType] = Field(
        description="Definition of filtering operations", examples=[OPERATIONS_EXAMPLE]
    )
    operations_parameters: Dict[str, Any] = Field(
        description="References to additional parameters that may be provided in runtime to parametrize operations",
        default_factory=lambda: {},
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls):
        from inference.core.workflows.execution_engine.entities.base import OutputDefinition
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsFilterBlockV1(WorkflowBlock):
    """Block that filters detections (e.g., by class, confidence, size, or dynamic zones)."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        operations: List[OperationDefinition],
        operations_parameters: Dict[str, Any],
    ) -> BlockResult:
        """
        Runs the DetectionsFilter transformation on a batch of detections.
        
        Args:
            predictions (Batch[sv.Detections]): Batch of predictions to filter.
            operations (List[OperationDefinition]): List of filter operations.
            operations_parameters (Dict[str, Any]): Additional runtime parameters (zones, classes, etc.).
        
        Returns:
            BlockResult: The filtered results.
        """
        return execute_transformation(
            predictions=predictions,
            operations=operations,
            operations_parameters=operations_parameters,
            custom_comparators={
                "(Detection) zone check": self._compare_zone_membership
            }
        )

    def _compare_zone_membership(
        self,
        detection: sv.Detection,
        zone_params: Dict[str, Any]
    ) -> bool:
        """
        Custom comparator that checks if a detection passes the 'in-zone' logic.
        
        zone_params keys:
            zone_detections: the detections used as zone bounding boxes
            zone_classes: which classes define valid zone bounding boxes
            restricted_classes: detection classes that must be within a zone
            ignored_classes: detection classes that we do not filter by zone
            in_zone_mode: 'must_be_in_zone' or 'must_be_out_of_zone'
        """

        # Get class name from detection
        if hasattr(detection, "class_name"):
            detection_class = detection.class_name
        else:
            detection_class = str(detection.class_id)

        zone_classes = zone_params.get("zone_classes", [])
        restricted_classes = zone_params.get("restricted_classes", [])
        ignored_classes = zone_params.get("ignored_classes", [])
        in_zone_mode = zone_params.get("in_zone_mode", "must_be_in_zone")
        zone_detections: sv.Detections = zone_params.get("zone_detections", sv.Detections.empty())

        # If ignored class, always pass
        if detection_class in ignored_classes:
            return True

        # If not restricted, don't filter by zone
        if detection_class not in restricted_classes:
            return True

        # Filter zone_detections to only those that match zone_classes
        zone_boxes = [
            z for z in zone_detections 
            if getattr(z, "class_name", None) in zone_classes
        ]

        # Check if detection center is inside at least one zone bounding box
        is_in_zone = False
        cx, cy = detection.x, detection.y
        for z in zone_boxes:
            left  = z.x - z.width / 2
            right = z.x + z.width / 2
            top   = z.y - z.height / 2
            bot   = z.y + z.height / 2

            if (left <= cx <= right) and (top <= cy <= bot):
                is_in_zone = True
                break

        # Decide pass/fail based on in_zone_mode
        if in_zone_mode == "must_be_in_zone":
            return is_in_zone
        elif in_zone_mode == "must_be_out_of_zone":
            return not is_in_zone
        else:
            return is_in_zone
