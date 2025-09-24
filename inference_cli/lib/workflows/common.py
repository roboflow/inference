import json
import os.path
import re
from collections import defaultdict
from copy import copy
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple

import cv2
import numpy as np
import pandas as pd
import pybase64
from rich.progress import track

from inference_cli.lib.utils import dump_json, dump_jsonl, read_json
from inference_cli.lib.workflows.entities import (
    ImagePath,
    ImageResultsIndexEntry,
    OutputFileType,
    WorkflowExecutionMetadataResultPath,
    WorkflowOutputField,
)

BASE64_DATA_TYPE_PATTERN = re.compile(r"^data:image\/[a-z]+;base64,")

TYPE_KEY = "type"
BASE_64_TYPE = "base64"
VALUE_KEY = "value"
DEDUCTED_IMAGE = "<deducted_image>"


def open_progress_log(output_directory: str) -> Tuple[TextIO, Set[str]]:
    os.makedirs(output_directory, exist_ok=True)
    log_path = get_progress_log_path(output_directory=output_directory)
    if not os.path.exists(log_path):
        file_descriptor = open(log_path, "w+")
    else:
        file_descriptor = open(log_path, "r+")
    all_processed_files = set(line.strip() for line in file_descriptor.readlines())
    return file_descriptor, all_processed_files


def denote_image_processed(
    log_file: TextIO, image_path: str, lock: Optional[Lock] = None
) -> None:
    image_name = os.path.basename(image_path)
    if lock is None:
        log_file.write(f"{image_name}\n")
        log_file.flush()
        return None
    with lock:
        log_file.write(f"{image_name}\n")
        log_file.flush()
        return None


def get_progress_log_path(output_directory: str) -> str:
    return os.path.abspath(os.path.join(output_directory, "progress.log"))


def dump_image_processing_results(
    result: Dict[str, Any],
    image_path: ImagePath,
    output_directory: str,
    save_image_outputs: bool,
) -> ImageResultsIndexEntry:
    images_in_result = []
    if save_image_outputs:
        images_in_result = extract_images_from_result(result=result)
    structured_content = deduct_images(result=result)
    image_results_dir = construct_image_output_dir_path(
        image_path=image_path,
        output_directory=output_directory,
    )
    os.makedirs(image_results_dir, exist_ok=True)
    structured_results_path = os.path.join(image_results_dir, "results.json")
    dump_json(
        path=structured_results_path,
        content=structured_content,
    )
    image_outputs = dump_images_outputs(
        image_results_dir=image_results_dir,
        images_in_result=images_in_result,
    )
    return ImageResultsIndexEntry(
        metadata_output_path=structured_results_path,
        image_outputs=image_outputs,
    )


def dump_images_outputs(
    image_results_dir: str,
    images_in_result: List[Tuple[str, np.ndarray]],
) -> Dict[WorkflowOutputField, List[ImagePath]]:
    result = defaultdict(list)
    for image_key, image in images_in_result:
        target_path = os.path.join(image_results_dir, f"{image_key}.jpg")
        target_path_dir = os.path.dirname(target_path)
        os.makedirs(target_path_dir, exist_ok=True)
        cv2.imwrite(target_path, image)
        workflow_field = _extract_workflow_field_from_image_key(image_key=image_key)
        result[workflow_field].append(target_path)
    return result


def _extract_workflow_field_from_image_key(image_key: str) -> WorkflowOutputField:
    return image_key.split("/")[0]


def construct_image_output_dir_path(image_path: str, output_directory: str) -> str:
    image_file_name = os.path.basename(image_path)
    return os.path.abspath(os.path.join(output_directory, image_file_name))


def deduct_images(result: Any) -> Any:
    if isinstance(result, list):
        return [deduct_images(result=e) for e in result]
    if isinstance(result, set):
        return {deduct_images(result=e) for e in result}
    if (
        isinstance(result, dict)
        and result.get(TYPE_KEY) == BASE_64_TYPE
        and VALUE_KEY in result
    ):
        return DEDUCTED_IMAGE
    if isinstance(result, np.ndarray):
        return DEDUCTED_IMAGE
    if isinstance(result, dict):
        return {k: deduct_images(result=v) for k, v in result.items()}
    return result


def extract_images_from_result(
    result: Any, key_prefix: str = ""
) -> List[Tuple[str, np.ndarray]]:
    if (
        isinstance(result, dict)
        and result.get(TYPE_KEY) == BASE_64_TYPE
        and VALUE_KEY in result
    ):
        loaded_image = decode_base64_image(result[VALUE_KEY])
        return [(key_prefix, loaded_image)]
    if isinstance(result, np.ndarray):
        return [(key_prefix, result)]
    current_result = []
    if isinstance(result, dict):
        for key, value in result.items():
            current_result.extend(
                extract_images_from_result(
                    result=value, key_prefix=f"{key_prefix}/{key}".lstrip("/")
                )
            )
    elif isinstance(result, list):
        for idx, element in enumerate(result):
            current_result.extend(
                extract_images_from_result(
                    result=element, key_prefix=f"{key_prefix}/{idx}".lstrip("/")
                )
            )
    return current_result


def decode_base64_image(payload: str) -> np.ndarray:
    value = BASE64_DATA_TYPE_PATTERN.sub("", payload)
    value = pybase64.b64decode(value)
    image_np = np.frombuffer(value, np.uint8)
    result = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError("Could not decode image")
    return result


def report_failed_files(
    failed_files: List[Tuple[str, str]], output_directory: str
) -> Optional[str]:
    if not failed_files:
        return None
    os.makedirs(output_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    failed_files_path = os.path.abspath(
        os.path.join(output_directory, f"failed_files_processing_{timestamp}.jsonl")
    )
    content = [{"file_path": e[0], "cause": e[1]} for e in failed_files]
    dump_jsonl(path=failed_files_path, content=content)
    print(
        f"Detected {len(failed_files)} processing failures. Details saved under: {failed_files_path}"
    )
    return failed_files_path


def aggregate_batch_processing_results(
    output_directory: str,
    aggregation_format: OutputFileType,
) -> str:
    file_descriptor, all_processed_files = open_progress_log(
        output_directory=output_directory
    )
    file_descriptor.close()
    all_results = [
        os.path.join(output_directory, f, "results.json")
        for f in sorted(all_processed_files)
        if os.path.exists(os.path.join(output_directory, f, "results.json"))
    ]
    decoded_content = []
    for result_path in track(all_results, description="Grabbing processing results..."):
        content = read_json(path=result_path)
        processed_file = extract_processed_image_name_from_predictions_path(
            predictions_path=result_path
        )
        content["image"] = processed_file
        decoded_content.append(content)
    if aggregation_format is OutputFileType.JSONL:
        aggregated_results_path = os.path.join(
            output_directory, "aggregated_results.jsonl"
        )
        dump_jsonl(
            path=aggregated_results_path,
            content=track(
                decoded_content, description="Dumping aggregated results to JSONL..."
            ),
        )
        return aggregated_results_path
    dumped_results = []
    for decoded_result in track(
        decoded_content, description="Dumping aggregated results to CSV..."
    ):
        dumped_results.append(
            {k: dump_objects_to_json(value=v) for k, v in decoded_result.items()}
        )
    data_frame = pd.DataFrame(dumped_results)
    aggregated_results_path = os.path.join(output_directory, "aggregated_results.csv")
    data_frame.to_csv(aggregated_results_path, index=False)
    return aggregated_results_path


def extract_processed_image_name_from_predictions_path(predictions_path: str) -> str:
    # we expect path to be <out_dir>/<image_name>/results.json - hence we extract basename of parent dir
    return os.path.basename(os.path.dirname(predictions_path))


def dump_objects_to_json(value: Any) -> Any:
    if isinstance(value, set):
        value = list(value)
    if isinstance(value, list) or isinstance(value, dict) or isinstance(value, set):
        return json.dumps(value)
    return value


class WorkflowsImagesProcessingIndex:

    @classmethod
    def init(cls) -> "WorkflowsImagesProcessingIndex":
        return cls(index_content={}, registered_output_images=set())

    def __init__(
        self,
        index_content: Dict[ImagePath, ImageResultsIndexEntry],
        registered_output_images: Set[WorkflowOutputField],
    ):
        self._index_content = index_content
        self._registered_output_images = registered_output_images

    @property
    def registered_output_images(self) -> Set[WorkflowOutputField]:
        return copy(self._registered_output_images)

    def collect_entry(
        self, image_path: ImagePath, entry: ImageResultsIndexEntry
    ) -> None:
        self._index_content[image_path] = entry
        for image_output_name in entry.image_outputs.keys():
            self._registered_output_images.add(image_output_name)

    def export_metadata(
        self,
    ) -> List[Tuple[ImagePath, WorkflowExecutionMetadataResultPath]]:
        return [
            (image_path, index_entry.metadata_output_path)
            for image_path, index_entry in self._index_content.items()
        ]

    def export_images(
        self,
    ) -> Dict[WorkflowOutputField, List[Tuple[ImagePath, List[ImagePath]]]]:
        result = {}
        for field_name in self._registered_output_images:
            result[field_name] = self.export_images_for_field(field_name=field_name)
        return result

    def export_images_for_field(
        self, field_name: WorkflowOutputField
    ) -> List[Tuple[ImagePath, List[ImagePath]]]:
        results = []
        for image_path, index_entry in self._index_content.items():
            if field_name not in index_entry.image_outputs:
                continue
            registered_images = index_entry.image_outputs[field_name]
            results.append((image_path, registered_images))
        return results
