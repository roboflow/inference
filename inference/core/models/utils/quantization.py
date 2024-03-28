from enum import Enum

class QuantizationMode(Enum):
    unquantized = "unquantized"
    int8 = "int8"
    fp16 = "fp16"
