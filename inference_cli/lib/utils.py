import json
import os.path
from typing import Dict, List, Optional, Union

from supervision.utils.file import read_yaml_file

from inference_cli.lib.env import ROBOFLOW_API_KEY
from inference_cli.lib.logger import CLI_LOGGER
from inference_sdk import InferenceConfiguration, InferenceHTTPClient


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
