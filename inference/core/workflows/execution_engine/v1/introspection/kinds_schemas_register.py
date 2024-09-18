from inference.core.workflows.execution_engine.entities.base import VideoMetadata
from inference.core.workflows.execution_engine.v1.introspection.kinds_schemas import (
    ImageSchema, ImageKeyPoints, MultiClassClassificationSchema, MultiLabelClassificationSchema,
    ObjectDetectionSchema, InstanceSegmentationSchema, KeyPointsDetectionSchema, CodeDetectionSchema
)
from inference.core.workflows.execution_engine.entities.types import IMAGE_KIND, VIDEO_METADATA_KIND, \
    IMAGE_KEYPOINTS_KIND, CLASSIFICATION_PREDICTION_KIND, OBJECT_DETECTION_PREDICTION_KIND, \
    INSTANCE_SEGMENTATION_PREDICTION_KIND, KEYPOINT_DETECTION_PREDICTION_KIND, QR_CODE_DETECTION_KIND

KIND_TO_SCHEMA_REGISTER = {
    IMAGE_KIND.name: ImageSchema.schema(),
    VIDEO_METADATA_KIND.name: VideoMetadata.schema(),
    IMAGE_KEYPOINTS_KIND.name: ImageKeyPoints.schema(),
    CLASSIFICATION_PREDICTION_KIND.name: [
        MultiClassClassificationSchema.schema(),
        MultiLabelClassificationSchema.schema(),
    ],
    OBJECT_DETECTION_PREDICTION_KIND.name: ObjectDetectionSchema.schema(),
    INSTANCE_SEGMENTATION_PREDICTION_KIND.name: InstanceSegmentationSchema.schema(),
    KEYPOINT_DETECTION_PREDICTION_KIND.name: KeyPointsDetectionSchema.schema(),
    QR_CODE_DETECTION_KIND.name: CodeDetectionSchema.schema(),
}
