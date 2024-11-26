from enum import Enum


class OutputFileType(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"
