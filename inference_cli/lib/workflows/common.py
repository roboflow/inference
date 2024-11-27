import json
import os.path
import re
from datetime import datetime
from functools import lru_cache
from threading import Lock
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple

import cv2
import numpy as np
import pandas as pd
import pybase64
import supervision as sv
from rich.progress import track

from inference_cli.lib.utils import dump_json, dump_jsonl, read_json
from inference_cli.lib.workflows.entities import OutputFileType

BASE64_DATA_TYPE_PATTERN = re.compile(r"^data:image\/[a-z]+;base64,")

TYPE_KEY = "type"
BASE_64_TYPE = "base64"
VALUE_KEY = "value"
DEDUCTED_IMAGE = "<deducted_image>"

IMAGES_EXTENSIONS = [
    "bmp",
    "BMP",
    "dib",
    "DIB",
    "jpeg",
    "JPEG",
    "jpg",
    "JPG",
    "jpe",
    "JPE",
    "jp2",
    "JP2",
    "png",
    "PNG",
    "webp",
    "WEBP",
]


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
    image_path: str,
    output_directory: str,
    save_image_outputs: bool,
) -> None:
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
    dump_images_outputs(
        image_results_dir=image_results_dir,
        images_in_result=images_in_result,
    )


def dump_images_outputs(
    image_results_dir: str,
    images_in_result: List[Tuple[str, np.ndarray]],
) -> None:
    for image_key, image in images_in_result:
        target_path = os.path.join(image_results_dir, f"{image_key}.jpg")
        target_path_dir = os.path.dirname(target_path)
        os.makedirs(target_path_dir, exist_ok=True)
        cv2.imwrite(target_path, image)


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


def get_all_images_in_directory(input_directory: str) -> List[str]:
    file_system_is_case_sensitive = _is_file_system_case_sensitive()
    if file_system_is_case_sensitive:
        return [
            path.as_posix()
            for path in sv.list_files_with_extensions(
                directory=input_directory,
                extensions=IMAGES_EXTENSIONS,
            )
        ]
    return list(
        {
            path.as_posix().lower()
            for path in sv.list_files_with_extensions(
                directory=input_directory,
                extensions=IMAGES_EXTENSIONS,
            )
        }
    )


@lru_cache()
def _is_file_system_case_sensitive() -> bool:
    fs_is_case_insensitive = os.path.exists(__file__.upper()) and os.path.exists(
        __file__.lower()
    )
    return not fs_is_case_insensitive


def report_failed_files(
    failed_files: List[Tuple[str, str]], output_directory: str
) -> None:
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


def aggregate_batch_processing_results(
    output_directory: str,
    aggregation_format: OutputFileType,
) -> None:
    file_descriptor, all_processed_files = open_progress_log(
        output_directory=output_directory
    )
    file_descriptor.close()
    all_results = [
        os.path.join(output_directory, f, "results.json")
        for f in all_processed_files
        if os.path.exists(os.path.join(output_directory, f, "results.json"))
    ]
    decoded_content = []
    for result_path in track(all_results, description="Grabbing processing results..."):
        decoded_content.append(read_json(path=result_path))
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
        return None
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


def dump_objects_to_json(value: Any) -> Any:
    if isinstance(value, set):
        value = list(value)
    if isinstance(value, list) or isinstance(value, dict) or isinstance(value, set):
        return json.dumps(value)
    return value
