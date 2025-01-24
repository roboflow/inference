from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

ImagePath = str
WorkflowExecutionMetadataResultPath = str
WorkflowOutputField = str


class OutputFileType(str, Enum):
    JSONL = "jsonl"
    CSV = "csv"


class ProcessingTarget(str, Enum):
    API = "api"
    INFERENCE_PACKAGE = "inference_package"


@dataclass(frozen=True)
class ImageResultsIndexEntry:
    metadata_output_path: WorkflowExecutionMetadataResultPath
    image_outputs: Dict[WorkflowOutputField, List[ImagePath]]


@dataclass(frozen=True)
class ImagesDirectoryProcessingDetails:
    output_directory: str
    processed_images: int
    failures: int
    result_metadata_paths: List[Tuple[ImagePath, WorkflowExecutionMetadataResultPath]]
    result_images_paths: Dict[
        WorkflowOutputField, List[Tuple[ImagePath, List[ImagePath]]]
    ]
    aggregated_results_path: Optional[str] = field(default=None)
    failures_report_path: Optional[str] = field(default=None)


@dataclass(frozen=True)
class VideoProcessingDetails:
    structured_results_file: Optional[str]
    video_outputs: Optional[Dict[str, str]]
