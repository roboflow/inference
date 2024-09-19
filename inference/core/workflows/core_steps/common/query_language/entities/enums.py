from enum import Enum


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


class DetectionsSelectionMode(Enum):
    LEFT_MOST = "left_most"
    RIGHT_MOST = "right_most"
    TOP_CONFIDENCE = "top_confidence"
