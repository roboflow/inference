from enum import Enum


class OutputFileType(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"


class ProcessingTarget(str, Enum):
    API = "api"
    INFERENCE_PACKAGE = "inference_package"
