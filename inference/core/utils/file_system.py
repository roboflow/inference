import json
import os.path
from typing import Optional, Union


def read_json(path: str) -> Optional[Union[dict, list]]:
    with open(path) as f:
        return json.load(f)


def dump_json(
    path: str, content: Union[dict, list], allow_override: bool = False, **kwargs
) -> None:
    absolute_path = os.path.abspath(path)
    if os.path.exists(absolute_path) and not allow_override:
        raise RuntimeError(f"File {absolute_path} exists and override is forbidden.")
    parent_dir = os.path.basename(absolute_path)
    os.makedirs(parent_dir, exist_ok=True)
    with open(absolute_path, "w") as f:
        json.dump(content, fp=f, **kwargs)
