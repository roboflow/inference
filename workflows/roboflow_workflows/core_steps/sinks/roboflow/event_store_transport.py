"""Shared-memory image transport for the local Event Ingestion Service.

Sinks that write to the local event store hand JPEGs over through files in a
shared-memory directory instead of base64 in the request body when the event
store reports the same shared-memory identity as this process.
"""

import base64
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import requests
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from roboflow_workflows.utils.images import encode_image_to_jpeg_bytes

EVENT_STORE_SHARED_MEMORY_PROTOCOL = 1
EVENT_STORE_SHM_DIR_ENV = "EVENT_STORE_SHM_DIR"
EVENT_STORE_SHARED_MEMORY_ENV = "EVENT_STORE_SHARED_MEMORY"
DEFAULT_SHM_DIR = "/dev/shm"
SHM_FILE_PREFIX = "rfevt-"
SHM_IMAGE_UNAVAILABLE_ERROR = "shm_image_unavailable"
PROBE_CACHE_SECONDS = 30.0
PROBE_TIMEOUT_SECONDS = 2.0
MIN_FREE_SHM_BYTES = 64 * 1024 * 1024
JPEG_QUALITY = 95
SHARED_MEMORY_DISABLED_VALUES = {"false", "0", "off", "disabled"}
IMAGE_FIELDS = (
    ("base64Image", "shmImage"),
    ("inputBase64Image", "inputShmImage"),
)
EVENTS_PATH = "/v2/events"

logger = logging.getLogger(__name__)

_probe_cache: Dict[str, Tuple[float, bool]] = {}


PreparedImage = Tuple[str, str, Union[bytes, str]]


def prepare_local_images(
    output_image: Optional[WorkflowImageData],
    input_image: Optional[WorkflowImageData],
) -> List[PreparedImage]:
    """Capture the images of a local event before it is queued.

    Images that already carry base64 keep it; others are JPEG-encoded now so the
    event sends the pixels as they were when it was created, while the choice
    between shared memory and base64 is made later by ``post_local_event``.

    Args:
        output_image: Image sent as ``shmImage`` / ``base64Image``.
        input_image: Image sent as ``inputShmImage`` / ``inputBase64Image``.

    Returns:
        ``(base64_key, shm_key, value)`` per image, where ``value`` is JPEG bytes
        or a base64 string.
    """
    prepared: List[PreparedImage] = []
    for (base64_key, shm_key), image in zip(IMAGE_FIELDS, (output_image, input_image)):
        if image is None:
            continue

        if (
            not _shared_memory_enabled()
            or getattr(image, "_base64_image", None) is not None
        ):
            prepared.append((base64_key, shm_key, image.base64_image))
        else:
            jpeg = encode_image_to_jpeg_bytes(
                image.numpy_image, jpeg_quality=JPEG_QUALITY
            )
            prepared.append((base64_key, shm_key, jpeg))
    return prepared


def post_local_event(
    url: str,
    *,
    payload: Dict[str, Any],
    prepared_images: Sequence[PreparedImage] = (),
    **requests_kwargs: Any,
) -> requests.Response:
    """POST an event to the local event store with its prepared images.

    The images are attached to the payload's first image entry, through shared
    memory when the event store shares this process's shared-memory directory
    and as base64 otherwise. The event store deletes the shared-memory files
    when it answers 201; on any other outcome they are deleted here. A 400
    ``shm_image_unavailable`` answer re-sends the event once with the images
    inlined as base64.

    Args:
        url: The ``/v2/events`` URL of the Event Ingestion Service.
        payload: The event payload.
        prepared_images: Images from ``prepare_local_images``.
        **requests_kwargs: Passed through to ``requests.post``.

    Returns:
        The final response of the event store.
    """
    event_store_url = url.removesuffix(EVENTS_PATH)
    shm_images: Dict[str, bytes] = {}
    response = None
    try:
        if prepared_images:
            _attach_images(
                payload["images"][0],
                prepared_images=prepared_images,
                event_store_url=event_store_url,
                shm_images=shm_images,
            )

        response = requests.post(url, json=payload, **requests_kwargs)
        if shm_images and _is_shm_image_unavailable(response):
            _inline_shm_images(payload, shm_images=shm_images)
            _remove_shm_files(shm_images)
            _probe_cache[_event_store_key(event_store_url)] = (time.monotonic(), False)
            response = requests.post(url, json=payload, **requests_kwargs)
        return response
    finally:
        if response is None or response.status_code != 201:
            _remove_shm_files(shm_images)


def _attach_images(
    entry: Dict[str, Any],
    *,
    prepared_images: Sequence[PreparedImage],
    event_store_url: str,
    shm_images: Dict[str, bytes],
) -> None:
    identity = None
    if any(isinstance(value, bytes) for _, _, value in prepared_images):
        identity = _shared_memory_identity(event_store_url)

    for base64_key, shm_key, value in prepared_images:
        if isinstance(value, str):
            entry[base64_key] = value
            continue

        name = _write_shm_image(value) if identity is not None else None
        if name is None:
            entry[base64_key] = base64.b64encode(value).decode("ascii")
            continue

        shm_images[name] = value
        entry[shm_key] = {"name": name, "size": len(value), "identity": identity}


def _shared_memory_enabled() -> bool:
    mode = os.getenv(EVENT_STORE_SHARED_MEMORY_ENV, "auto").strip().lower()

    return mode not in SHARED_MEMORY_DISABLED_VALUES


def _shm_dir() -> Path:
    directory = Path(os.getenv(EVENT_STORE_SHM_DIR_ENV, DEFAULT_SHM_DIR))

    return directory


def _local_identity() -> Optional[str]:
    try:
        st = _shm_dir().stat()
    except OSError:
        return None

    return f"{st.st_dev}:{st.st_ino}"


def _event_store_key(event_store_url: str) -> str:
    key = event_store_url.rstrip("/")

    return key


def _shared_memory_identity(event_store_url: str) -> Optional[str]:
    if not _shared_memory_enabled():
        return None

    identity = _local_identity()
    if identity is None:
        return None

    key = _event_store_key(event_store_url)
    now = time.monotonic()
    cached = _probe_cache.get(key)
    if cached is None or now - cached[0] >= PROBE_CACHE_SECONDS:
        cached = (now, _probe_event_store(key, identity=identity))
        _probe_cache[key] = cached

    return identity if cached[1] else None


def _probe_event_store(event_store_url: str, *, identity: str) -> bool:
    try:
        response = requests.get(
            f"{event_store_url}/health", timeout=PROBE_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        supported = response.json().get("shm_identity") == identity
    except Exception as error:
        logger.debug("Event store shared-memory probe failed: %s", error)
        supported = False

    return supported


def _write_shm_image(jpeg: bytes) -> Optional[str]:
    directory = _shm_dir()
    try:
        if shutil.disk_usage(directory).free - len(jpeg) < MIN_FREE_SHM_BYTES:
            return None

        name = f"{SHM_FILE_PREFIX}{uuid4().hex}"
        fd = os.open(
            directory / name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
        )
    except OSError as error:
        logger.warning("Cannot write event image to shared memory: %s", error)
        return None

    try:
        view = memoryview(jpeg)
        while view:
            view = view[os.write(fd, view) :]
    except OSError as error:
        logger.warning("Cannot write event image to shared memory: %s", error)
        _remove_shm_files([name])
        return None
    finally:
        os.close(fd)

    return name


def _is_shm_image_unavailable(response: requests.Response) -> bool:
    if response.status_code != 400:
        return False

    try:
        unavailable = response.json().get("error") == SHM_IMAGE_UNAVAILABLE_ERROR
    except Exception:
        unavailable = False

    return unavailable


def _inline_shm_images(
    payload: Dict[str, Any], *, shm_images: Dict[str, bytes]
) -> None:
    for entry in payload.get("images", []):
        for base64_key, shm_key in IMAGE_FIELDS:
            reference = entry.pop(shm_key, None)
            if reference is not None:
                jpeg = shm_images[reference["name"]]
                entry[base64_key] = base64.b64encode(jpeg).decode("ascii")


def _remove_shm_files(shm_names: Sequence[str]) -> None:
    directory = _shm_dir()
    for name in shm_names:
        try:
            (directory / name).unlink()
        except FileNotFoundError:
            pass
