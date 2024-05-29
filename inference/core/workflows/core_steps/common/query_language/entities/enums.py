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


class SequenceUnwrapMethod(Enum):
    FIRST = "first"
    LAST = "last"


class StatementsGroupsOperator(Enum):
    AND = "and"
    OR = "or"


class ImageProperty(Enum):
    SIZE = "size"
    HEIGHT = "height"
    WIDTH = "width"
