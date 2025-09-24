import json
import os.path
import subprocess
from functools import lru_cache
from typing import Dict, Generator, Iterable, List, Optional, TypeVar, Union

import supervision as sv
from supervision.utils.file import read_yaml_file

from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.exceptions import InferencePackageMissingError
from inference_cli.lib.logger import CLI_LOGGER
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

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

VIDEOS_EXTENSIONS = [
    "mp4",
    "m4a",
    "m4p",
    "m4b",
    "m4r",
    "m4v",
    "MP$",
    "M4A",
    "M4P",
    "M4B",
    "M4R",
    "M4V",
    "avi",
    "AVI",
    "mov",
    "MOV",
    "mkv",
    "MKV",
    "flv",
    "FLV",
    "wmv",
    "WMV",
    "mpeg",
    "MPEG",
    "mpg",
    "MPG",
    "3gp",
    "3GP",
    "webm",
    "WEBM",
    "ogg",
    "OGG",
]

B = TypeVar("B")


def ensure_inference_is_installed() -> None:
    try:
        from inference import get_model
    except Exception as error:
        if (
            os.getenv("ALLOW_INTERACTIVE_INFERENCE_INSTALLATION", "True").lower()
            == "false"
        ):
            raise InferencePackageMissingError(
                "You need to install `inference` package to use this feature. Run `pip install inference`"
            ) from error
        print(
            "You need to have `inference` package installed. Do you want the package to be installed? [YES/no]"
        )
        user_choice = input()
        if user_choice.lower() != "yes":
            raise InferencePackageMissingError(
                "You need to install `inference` package to use this feature. Run `pip install inference`"
            ) from error
        try:
            subprocess.run("pip install inference".split(), check=True)
            import inference
        except Exception as inner_error:
            raise InferencePackageMissingError(
                f"Installation of package failed. Cause: {inner_error}"
            ) from inner_error


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def read_env_file(path: str) -> Dict[str, str]:
    file_lines = read_file_lines(path=path)
    result = {}
    for line in file_lines:
        chunks = line.split("=")
        if len(chunks) != 2:
            CLI_LOGGER.warning(
                f"Line: `{line}` in {path} file does not match pattern NAME=VALUE"
            )
            continue
        name, value = chunks[0], chunks[1]
        result[name] = value
    return result


def read_file_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines() if len(line.strip()) > 0]


def dump_json(path: str, content: Union[dict, list]) -> None:
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f)


def dump_jsonl(path: str, content: Iterable[dict]) -> None:
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w") as f:
        for line in content:
            f.write(f"{json.dumps(line)}\n")


def initialise_client(
    host: str, api_key: Optional[str], model_configuration: Optional[str], **kwargs
) -> InferenceHTTPClient:
    if api_key is None:
        api_key = ROBOFLOW_API_KEY
    client = InferenceHTTPClient(
        api_url=host,
        api_key=api_key,
    )
    raw_configuration = {}
    if model_configuration is not None:
        raw_configuration = read_yaml_file(file_path=model_configuration)
    raw_configuration.update(kwargs)
    config = InferenceConfiguration(**raw_configuration)
    client.configure(inference_configuration=config)
    return client


def ensure_target_directory_is_empty(
    output_directory: str, allow_override: bool, only_files: bool = True
) -> None:
    if allow_override:
        return None
    if not os.path.exists(output_directory):
        return None
    files_in_directory = [
        f
        for f in os.listdir(output_directory)
        if not only_files or os.path.isfile(os.path.join(output_directory, f))
    ]
    if files_in_directory:
        raise RuntimeError(
            f"Detected content in output directory: {output_directory}. "
            f"Command cannot run, as content override is forbidden. Use `--allow_override` to proceed."
        )


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


def get_all_videos_in_directory(input_directory: str) -> List[str]:
    file_system_is_case_sensitive = _is_file_system_case_sensitive()
    if file_system_is_case_sensitive:
        return [
            path.as_posix()
            for path in sv.list_files_with_extensions(
                directory=input_directory,
                extensions=VIDEOS_EXTENSIONS,
            )
        ]
    return list(
        {
            path.as_posix().lower()
            for path in sv.list_files_with_extensions(
                directory=input_directory,
                extensions=VIDEOS_EXTENSIONS,
            )
        }
    )


@lru_cache()
def _is_file_system_case_sensitive() -> bool:
    fs_is_case_insensitive = os.path.exists(__file__.upper()) and os.path.exists(
        __file__.lower()
    )
    return not fs_is_case_insensitive


def create_batches(
    sequence: Iterable[B], batch_size: int
) -> Generator[List[B], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if len(current_batch) > 0:
        yield current_batch
