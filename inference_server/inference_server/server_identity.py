import functools
import random
import string
from pathlib import Path
from typing import Optional

from inference_server import configuration

JETSON_SERIAL_PATH = Path("/proc/device-tree/serial-number")
RANDOM_PART_LENGTH = 6


def _random_string(length: int) -> str:
    letters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(letters) for _ in range(length))

    return random_string


def _gpu_index() -> Optional[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        return None

    return None


def _jetson_serial() -> Optional[str]:
    try:
        if not JETSON_SERIAL_PATH.exists():
            return None

        serial_number = JETSON_SERIAL_PATH.read_text().strip()
    except Exception:
        return None

    return serial_number or None


@functools.cache
def _generated_server_id() -> str:
    try:
        random_part = _random_string(RANDOM_PART_LENGTH)
        jetson_serial = _jetson_serial()
        if jetson_serial is not None:
            return f"{random_part}-JETSON-{jetson_serial}"

        gpu_index = _gpu_index()
        if gpu_index is not None:
            return f"{random_part}-GPU-{gpu_index}"

        return random_part
    except Exception:
        return "UNKNOWN"


def get_inference_server_id() -> str:
    if configuration.INFERENCE_SERVER_ID:
        return configuration.INFERENCE_SERVER_ID

    server_id = _generated_server_id()

    return server_id
