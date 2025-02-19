from enum import Enum

from inference.core.workflows.core_steps.analytics.line_counter.v2 import (
    DETECTIONS_IN_OUT_PARAM,
)
from inference.core.workflows.core_steps.analytics.velocity.v1 import (
    SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS,
    SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS,
    SPEED_KEY_IN_SV_DETECTIONS,
    VELOCITY_KEY_IN_SV_DETECTIONS,
)
from inference.core.workflows.execution_engine.constants import (
    BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS,
    BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS,
    IMAGE_DIMENSIONS_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PATH_DEVIATION_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    TIME_IN_ZONE_KEY_IN_SV_DETECTIONS,
)


class NumberCastingMode(Enum):
    INT = "int"
    FLOAT = "float"


class SequenceAggregationFunction(Enum):
    MIN = "min"
    MAX = "max"


class SequenceAggregationMode(Enum):
    FIRST = "first"
    LAST = "last"
    MOST_COMMON = "most_common"
    LEAST_COMMON = "least_common"


class ClassificationProperty(Enum):
    TOP_CLASS = "top_class"
    TOP_CLASS_CONFIDENCE = "top_class_confidence"
    TOP_CLASS_CONFIDENCE_SINGLE = "top_class_confidence_single"
    ALL_CLASSES = "all_classes"
    ALL_CONFIDENCES = "all_confidences"


class DetectionsProperty(Enum):
    CONFIDENCE = "confidence"
    CLASS_NAME = "class_name"
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    SIZE = "size"
    CLASS_ID = "class_id"
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    IN_OUT = DETECTIONS_IN_OUT_PARAM
    PATH_DEVIATION = PATH_DEVIATION_KEY_IN_SV_DETECTIONS
    POLYGON = POLYGON_KEY_IN_SV_DETECTIONS
    TIME_IN_ZONE = TIME_IN_ZONE_KEY_IN_SV_DETECTIONS
    TRACKER_ID = "tracker_id"
    VELOCITY = VELOCITY_KEY_IN_SV_DETECTIONS
    SPEED = SPEED_KEY_IN_SV_DETECTIONS
    SMOOTHED_VELOCITY = SMOOTHED_VELOCITY_KEY_IN_SV_DETECTIONS
    SMOOTHED_SPEED = SMOOTHED_SPEED_KEY_IN_SV_DETECTIONS
    DIMENSIONS = IMAGE_DIMENSIONS_KEY
    PREDICTION_TYPE = PREDICTION_TYPE_KEY
    KEYPOINTS_XY = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
    BOUNDING_RECT = BOUNDING_RECT_RECT_KEY_IN_SV_DETECTIONS
    BOUNDING_RECT_WIDTH = BOUNDING_RECT_WIDTH_KEY_IN_SV_DETECTIONS
    BOUNDING_RECT_HEIGHT = BOUNDING_RECT_HEIGHT_KEY_IN_SV_DETECTIONS
    BOUNDING_RECT_ANGLE = BOUNDING_RECT_ANGLE_KEY_IN_SV_DETECTIONS


class DetectionsSortProperties(Enum):
    CONFIDENCE = "confidence"
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    SIZE = "size"
    CENTER_X = "center_x"
    CENTER_Y = "center_y"


class StatementsGroupsOperator(Enum):
    AND = "and"
    OR = "or"


class ImageProperty(Enum):
    SIZE = "size"
    HEIGHT = "height"
    WIDTH = "width"
    ASPECT_RATIO = "aspect_ratio"


class DetectionsSelectionMode(Enum):
    LEFT_MOST = "left_most"
    RIGHT_MOST = "right_most"
    TOP_CONFIDENCE = "top_confidence"
    FIRST = "first"
    LAST = "last"
