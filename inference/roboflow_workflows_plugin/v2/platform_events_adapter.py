"""Host adapters of the V2 vision event ports.

::

    RoboflowVisionEventsAdapter.submit ── call_host(VISION_EVENT) ──┐
      cloud: POST /vision-events/upload, output image then input image
             POST /vision-events with a client-generated eventId ─▶ eventId
      local: POST <event_store_url>/v2/events, base64 images ─▶ 201 "id"
    VisionEventBundleAdapter
      prepare_destination: permission, name, allowed root, writable, free
      create_bundle: same checks; with an allowed root, open the directory
                     by walking from the root without following links
                     call_host(EVENT_BUNDLE) builds the archive
                     size limit, publish (hard link refuses a taken name,
                     else V1's checked replacing rename), directory fsync
                     with an allowed root, publish inside the open directory
                     and fail if a new walk no longer reaches it; a published
                     bundle may remain for host-controlled recovery

The wire and archive contracts are the pinned V1 helpers, called directly:
prediction conversion and payload/tar builders of ``v1_tensor``, cloud
headers of ``vision_events.v1`` (``ROBOFLOW_API_EXTRA_HEADERS``; a configured
``Authorization`` wins), which ``v1_tensor`` lacks. V1's wrappers that turn
failures into messages with response bodies are not used: a failure raises
inside ``call_host``, which replaces it with a fixed diagnostic.

The bundle path carries no credential. Its destination checks and publication
run outside the boundary so their V1 messages stay actionable.
"""

import base64
import errno
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
from roboflow_workflows.v2.blocks.roboflow_platform._shared.vision_events import EventBundle, VisionEventRequest
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret

from inference.core.env import API_BASE_URL
from inference.core.exceptions import (
    MalformedRoboflowAPIResponseError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.utils.requests import api_key_safe_raise_for_status
from inference.core.utils.url_utils import wrap_url
from inference.roboflow_workflows_plugin.sinks.vision_events import v1 as v1_events
from inference.roboflow_workflows_plugin.sinks.vision_events import (
    v1_tensor as v1_tensor_events,
)
from inference.roboflow_workflows_plugin.sinks.vision_events_bundle import (
    v1_tensor as v1_bundles,
)
from inference.roboflow_workflows_plugin.v2.credential_boundary import (
    HostOperation,
    call_host,
)

REQUEST_TIMEOUT_SECONDS = 30

logger = logging.getLogger(__name__)

# Confinement to an allowed root opens each directory relative to its parent
# and never follows a link. Windows has none of these calls. os.replace takes
# directory descriptors wherever os.rename does.
DIRECTORY_RELATIVE_FILES = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and {os.open, os.mkdir, os.stat, os.link, os.rename, os.unlink}
    <= os.supports_dir_fd
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)

# call_host names the category of these classes; the text is discarded.
_HTTP_STATUS_ERRORS = {
    401: RoboflowAPINotAuthorizedError,
    403: RoboflowAPIForbiddenError,
    404: RoboflowAPINotNotFoundError,
}


def _status_error(status_code: int) -> Exception:
    error_class = _HTTP_STATUS_ERRORS.get(
        status_code, RoboflowAPIUnsuccessfulRequestError
    )
    error = error_class(f"HTTP {status_code}")

    return error


def _jpeg(image: ImageData, *, quality: int) -> bytes:
    # V1 encoders take BGR (or single-channel) numpy pixels.
    pixels = image.tensor_image.detach().to("cpu").permute(1, 2, 0).numpy()
    pixels = pixels[:, :, ::-1] if pixels.shape[2] == 3 else pixels[:, :, 0]
    encoded = encode_image_to_jpeg_bytes(pixels.copy(), jpeg_quality=quality)

    return encoded


def _image_entries(
    references: Dict[str, str], annotations: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """V1's one image entry: image references, label, then annotations."""
    if not references:
        return []

    entry = {**references, "label": "workflow", **annotations}

    return [entry]


class RoboflowVisionEventsAdapter:
    """``vision_events_port`` for the Roboflow API and local event stores.

    Args:
        api_base_url: Roboflow API base URL; the host's setting by default.
        event_ingestion_api_key: Optional ``X-API-Key`` of a local Event
            Ingestion Service. V1 read ``EVENT_INGESTION_API_KEY`` from the
            environment; a host that wants that passes it here.
    """

    def __init__(
        self,
        *,
        api_base_url: str = API_BASE_URL,
        event_ingestion_api_key: Optional[Secret] = None,
    ) -> None:
        self._api_base_url = api_base_url.rstrip("/")
        self._event_ingestion_api_key = event_ingestion_api_key

    def submit(
        self,
        request: VisionEventRequest,
        *,
        api_key: Optional[Secret],
        write_to_event_store: bool,
        event_store_url: str,
    ) -> str:
        """Upload the images, if any, then create one event.

        Runs the boundary on the calling thread, which for a background
        effect is the ``sink_effects`` worker. An image uploaded before a
        failed event request stays uploaded, as in V1.

        Args:
            request: Detached inputs.
            api_key: Roboflow API key; None with ``write_to_event_store``.
            write_to_event_store: Post to the local event store instead.
            event_store_url: Base URL of the local event store.

        Returns:
            The cloud ``eventId`` or the local service's ``id`` (``""`` when
            the service reports none, as in V1).

        Raises:
            PlatformRequestError: On any failure, with a fixed message naming
                the operation and a category, e.g. "authorization failed".
        """
        event_id = call_host(
            HostOperation.VISION_EVENT,
            self._submit,
            platform=True,
            request=request,
            api_key=api_key,
            local=write_to_event_store,
            event_store_url=event_store_url,
        )

        return event_id

    def _submit(
        self,
        *,
        request: VisionEventRequest,
        api_key: Optional[Secret],
        local: bool,
        event_store_url: str,
    ) -> str:
        try:
            if local:
                return self._submit_local(request, event_store_url=event_store_url)
            return self._submit_cloud(request, api_key=api_key)
        # requests' own classes are outside call_host's categories.
        except requests.exceptions.Timeout:
            raise RoboflowAPITimeoutError("timed out") from None
        except requests.exceptions.ConnectionError:
            raise RoboflowAPIConnectionError("connection failed") from None
        except requests.exceptions.HTTPError as error:
            status_code = getattr(error.response, "status_code", 0)
            raise _status_error(status_code) from None

    def _submit_cloud(
        self, request: VisionEventRequest, *, api_key: Optional[Secret]
    ) -> str:
        if api_key is None:
            raise ValueError("cloud vision events need an API key")

        key = api_key.reveal()
        # sourceId is the display (output) image, inputSourceId the original.
        references = {}
        for reference, image, quality in (
            ("sourceId", request.output_image, 85),
            ("inputSourceId", request.input_image, 95),
        ):
            if image is not None:
                references[reference] = self._upload(key, image, quality=quality)
        payload = v1_tensor_events._build_event_payload(
            event_type=request.event_type,
            solution=request.solution,
            images=_image_entries(
                references,
                v1_tensor_events._convert_predictions_to_annotations(
                    request.predictions
                ),
            ),
            event_data=request.event_data,
            custom_metadata=request.custom_metadata,
        )
        response = requests.post(
            wrap_url(f"{self._api_base_url}/vision-events"),
            headers=v1_events._build_cloud_request_headers(
                api_key=key, explicit_headers={"Content-Type": "application/json"}
            ),
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        api_key_safe_raise_for_status(response=response)
        # The created event's id is the one this client generated; the
        # response body has no contract, as in V1.
        event_id = payload["eventId"]

        return event_id

    def _upload(self, key: str, image: ImageData, *, quality: int) -> str:
        # The request of V1 vision_events.v1._upload_image, with its response
        # checked: a missing sourceId must not produce an event.
        response = requests.post(
            wrap_url(f"{self._api_base_url}/vision-events/upload"),
            headers=v1_events._build_cloud_request_headers(api_key=key),
            files={"file": ("image.jpg", _jpeg(image, quality=quality), "image/jpeg")},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        api_key_safe_raise_for_status(response=response)
        source_id = _json_field(response, "sourceId")
        if not isinstance(source_id, str) or not source_id:
            raise MalformedRoboflowAPIResponseError("upload without sourceId")

        return source_id

    def _submit_local(
        self, request: VisionEventRequest, *, event_store_url: str
    ) -> str:
        # base64Image is the display (output) image, inputBase64Image the
        # original; V1 embedded both at JPEG quality 95.
        references = {
            reference: base64.b64encode(_jpeg(image, quality=95)).decode("ascii")
            for reference, image in (
                ("base64Image", request.output_image),
                ("inputBase64Image", request.input_image),
            )
            if image is not None
        }
        images = _image_entries(
            references,
            v1_tensor_events._convert_predictions_to_annotations(request.predictions),
        )
        payload = {
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "solution": request.solution,
            "event_schema": request.event_type,
            "event_data": request.event_data,
            "images": images,
        }
        if request.custom_metadata:
            payload["custom_metadata"] = request.custom_metadata
        if images:
            payload["displayImagePosition"] = 0
        headers = {"Content-Type": "application/json"}
        if self._event_ingestion_api_key is not None:
            headers["X-API-Key"] = self._event_ingestion_api_key.reveal()
        response = requests.post(
            f"{event_store_url.rstrip('/')}/v2/events",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        # Only 201 means created; 529 is the service's backpressure answer.
        if response.status_code != 201:
            raise _status_error(response.status_code)

        event_id = _json_field(response, "id")
        # As in V1: a created event without an id is still created.
        event_id = "" if event_id is None else str(event_id)

        return event_id


def _json_field(response: requests.Response, name: str) -> Any:
    try:
        body = response.json()
    except ValueError:
        raise MalformedRoboflowAPIResponseError("response is not JSON") from None

    if not isinstance(body, dict):
        raise MalformedRoboflowAPIResponseError("response is not a JSON object")

    return body.get(name)


def _taken_name_error(path: str) -> ValueError:
    error = ValueError(
        f"Vision event bundle {path!r} already exists. A custom "
        "file_name must be unique for every event."
    )

    return error


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _publish_bundle_at(
    directory_fd: int, *, path: str, content: bytes
) -> os.stat_result:
    """V1's ``_publish_bundle`` with every call relative to ``directory_fd``.

    Args:
        directory_fd: Open directory that receives the bundle.
        path: The bundle's path, used for its name and in messages.
        content: Archive bytes.

    Returns:
        The status of the published file, for a later read-only diagnostic.

    Raises:
        ValueError: When the name is taken.
        OSError: When writing fails; temporary cleanup is attempted.
    """
    name = Path(path).name
    # V1's dot-prefixed temporary name, which file movers skip.
    temporary = f".tmp_{secrets.token_hex(8)}_{name}"
    file_fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(file_fd, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
            published = os.fstat(file.fileno())
        try:
            os.link(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        except FileExistsError:
            raise _taken_name_error(path) from None
        except OSError as error:
            if error.errno not in v1_bundles._LINK_UNSUPPORTED_ERRNOS:
                raise
            try:
                os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise _taken_name_error(path) from None
            logger.warning(
                "Hard links unsupported for %s - publishing the vision event "
                "bundle with a replacing rename, which cannot guarantee that "
                "a concurrent event with the same file name is not overwritten.",
                path,
            )
            os.replace(
                temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd
            )
    finally:
        # As in V1: cleanup neither fails a published bundle nor masks an error.
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError as error:
            logger.warning(
                "Failed to remove vision event bundle temporary file %s: %s",
                temporary,
                error,
            )

    return published


class VisionEventBundleAdapter:
    """``vision_event_bundle_port`` writing V1 format bundles to local disk.

    It writes ``.tar.gz`` files and nothing else; moving or uploading them is
    the deployment's business. Publication refuses a taken name. A hard link
    makes the refusal atomic: of two concurrent writers of one name, exactly
    one wins. When the filesystem refuses hard links (e.g. FAT removable
    media) it falls back, as V1 does, to a check and then a replacing
    rename. A repeated name is still refused, but two concurrent writers of
    one custom ``file_name`` can race and one bundle can overwrite the
    other; a WARNING is logged on each fallback. Leave ``file_name`` unset
    for unique generated names, or serialize writes that share a custom name.

    With an allowed root, ``create_bundle`` opens the checked directory
    before it builds the archive. It walks from the root one directory at a
    time and follows no link, so a link changed later cannot redirect the
    bundle. The temporary file, publication and fsync use that open
    directory. Afterwards the walk is repeated; if it no longer reaches the
    same directory, e.g. because the directory was moved out of the root,
    the call fails. The published basename is left untouched: portable
    name-based removal cannot exclude a replacement by another writer,
    even after an inode check. A bundle may remain in the opened directory
    for host-controlled recovery. Directory owners must be trusted not to
    move directories or replace the private temporary names; their cleanup
    is best effort. The root's own path and its ancestors are host
    configuration and is trusted. This needs directory-relative file calls
    (``DIRECTORY_RELATIVE_FILES``): Linux and macOS have them, Windows does
    not, and there an allowed root refuses every bundle.

    Args:
        allow_access_to_file_system: Host permission; False refuses every
            bundle, as V1 with ``ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE``
            off.
        allowed_write_directory: Optional root that bundles must stay in,
            after symbolic links are followed.
    """

    def __init__(
        self,
        *,
        allow_access_to_file_system: bool = False,
        allowed_write_directory: Optional[str] = None,
    ) -> None:
        self._allow_access_to_file_system = allow_access_to_file_system
        self._allowed_directory = (
            None
            if allowed_write_directory is None
            else Path(allowed_write_directory).resolve()
        )

    def prepare_destination(
        self, *, target_directory: str, file_name: Optional[str]
    ) -> None:
        """Create the directory and refuse a destination V1 would refuse.

        Args:
            target_directory: Output directory.
            file_name: Custom bundle name, or None for a generated one.

        Raises:
            RuntimeError: When filesystem access is not allowed, or an
                allowed root is set on a platform without directory-relative
                file calls.
            ValueError: When the name is unsafe, the directory is outside the
                allowed root or not writable, or the custom file exists.
        """
        self._checked_directory(target_directory=target_directory, file_name=file_name)

    def create_bundle(
        self,
        request: VisionEventRequest,
        *,
        target_directory: str,
        file_name: Optional[str],
    ) -> EventBundle:
        """Write one bundle and refuse a taken name.

        Only hard-link publication refuses a concurrent writer of the same
        name atomically; see the class for the fallback race and for how an
        allowed root stays in force while the archive is built.

        Args:
            request: Detached inputs.
            target_directory: Output directory.
            file_name: Custom bundle name, or None for
                ``event_<UTC time>_<eventId>.tar.gz``.

        Returns:
            The event id stored in the bundle and the bundle's absolute path.
            With an allowed root the path has its symbolic links resolved: it
            names the directory that was checked and written.

        Raises:
            RuntimeError: When filesystem access is not allowed, or an
                allowed root is set on a platform without directory-relative
                file calls.
            ValueError: On a refused destination, a taken name, a bundle
                over V1's 25 MiB consumer limit, or a directory that was
                replaced or moved out of the allowed root meanwhile.
            PlatformRequestError: When building the archive fails.
            OSError: When writing fails; temporary cleanup is attempted.
                A bundle published before a later failure may remain.
        """
        # A background effect writes later: check the destination again.
        directory = self._checked_directory(
            target_directory=target_directory, file_name=file_name
        )
        # Opened before the archive is built, so later link changes are inert.
        directory_fd = (
            None
            if self._allowed_directory is None
            else self._open_directory(directory, target_directory=target_directory)
        )
        try:
            event_id = str(uuid4())
            timestamp = datetime.now(timezone.utc)
            content = call_host(
                HostOperation.EVENT_BUNDLE,
                self._archive,
                platform=True,
                request=request,
                event_id=event_id,
                timestamp=timestamp,
            )
            if len(content) > v1_bundles.MAX_BUNDLE_SIZE_BYTES:
                raise ValueError(
                    f"Bundle size {len(content):,} bytes exceeds the companion "
                    f"API limit of {v1_bundles.MAX_BUNDLE_SIZE_BYTES:,} bytes "
                    "(25 MiB). Reduce the number or size of images in this event."
                )

            if directory_fd is None:
                path = v1_bundles._generate_bundle_path(
                    target_directory, timestamp, event_id, file_name
                )
                v1_bundles._publish_bundle(target_path=path, content=content)
                v1_bundles._fsync_directory(os.path.dirname(path))
            else:
                path = v1_bundles._generate_bundle_path(
                    str(directory), timestamp, event_id, file_name
                )
                self._publish_in_directory(
                    directory_fd,
                    directory=directory,
                    target_directory=target_directory,
                    path=path,
                    content=content,
                )
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
        bundle = EventBundle(event_id=event_id, bundle_path=path)

        return bundle

    def _checked_directory(
        self, *, target_directory: str, file_name: Optional[str]
    ) -> Path:
        if not self._allow_access_to_file_system:
            raise RuntimeError(
                "Vision event bundles cannot be written here: local file system "
                "access is forbidden. Use self-hosted inference or Roboflow "
                "Dedicated Deployment."
            )

        if self._allowed_directory is not None and not DIRECTORY_RELATIVE_FILES:
            raise RuntimeError(
                "Vision event bundles cannot be kept inside the allowed write "
                "directory on this platform: it lacks directory-relative file "
                "calls (dir_fd, O_DIRECTORY, O_NOFOLLOW). Run the host on Linux "
                "or macOS, or configure no allowed write directory, which "
                "removes that confinement."
            )

        if file_name is not None:
            v1_bundles.validate_bundle_file_name(file_name)
        directory = Path(target_directory).resolve()
        if self._allowed_directory is not None and not directory.is_relative_to(
            self._allowed_directory
        ):
            raise ValueError(
                f"Vision event bundles cannot be written to {target_directory!r}: "
                f"it is not inside the allowed write directory "
                f"{str(self._allowed_directory)!r}."
            )

        if self._allowed_directory is None:
            directory.mkdir(parents=True, exist_ok=True)
        else:
            # Creating by walking keeps new directories inside the root too.
            os.close(self._open_directory(directory, target_directory=target_directory))
        if not os.access(directory, os.W_OK):
            raise ValueError(
                f"Vision event bundles cannot be written to {target_directory!r}: "
                "the directory is not writable."
            )

        # A generated name holds a uuid4; only a custom name can repeat.
        if file_name is not None:
            path = v1_bundles._generate_bundle_path(
                target_directory, datetime.now(timezone.utc), "", file_name
            )
            if os.path.lexists(path):
                raise _taken_name_error(path)

        return directory

    def _open_directory(
        self, directory: Path, *, target_directory: str, create: bool = True
    ) -> int:
        """Open ``directory`` by walking from the allowed root.

        Each step opens one name relative to its parent with ``O_NOFOLLOW``.
        A link or file put in place of a checked directory fails the open
        instead of redirecting it.
        """
        root = self._allowed_directory
        if create:
            root.mkdir(parents=True, exist_ok=True)
        try:
            directory_fd = os.open(root, _DIRECTORY_FLAGS)
            try:
                for name in directory.relative_to(root).parts:
                    if create:
                        try:
                            os.mkdir(name, dir_fd=directory_fd)
                        except FileExistsError:
                            pass
                    child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                    os.close(directory_fd)
                    directory_fd = child_fd
            except BaseException:
                os.close(directory_fd)
                raise
        except OSError as error:
            if error.errno not in (errno.ELOOP, errno.ENOTDIR):
                raise
            raise ValueError(
                f"Vision event bundles cannot be written to {target_directory!r}: "
                "a symbolic link or file replaced part of the checked directory."
            ) from None

        return directory_fd

    def _publish_in_directory(
        self,
        directory_fd: int,
        *,
        directory: Path,
        target_directory: str,
        path: str,
        content: bytes,
    ) -> None:
        published = _publish_bundle_at(directory_fd, path=path, content=content)
        name = Path(path).name
        if not self._reaches(directory, directory_fd=directory_fd):
            # This snapshot only describes the error. It cannot authorize an
            # unlink: another writer can replace the name just after stat.
            try:
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                name_state = "absent"
            except OSError:
                name_state = "unreadable"
            else:
                name_state = (
                    "this bundle" if _same_file(current, published) else "another file"
                )
            raise ValueError(
                f"Vision event bundles cannot be written to {target_directory!r}: "
                "the directory was moved or replaced while the bundle was "
                "written. Automatic cleanup was skipped to avoid deleting "
                "another writer's file; the bundle may remain in the opened "
                f"directory (published name at last check: {name_state})."
            )

        os.fsync(directory_fd)

    def _reaches(self, directory: Path, *, directory_fd: int) -> bool:
        """Whether a fresh walk from the root still ends at ``directory_fd``."""
        try:
            check_fd = self._open_directory(
                directory, target_directory=str(directory), create=False
            )
        except (OSError, ValueError):
            return False

        try:
            reached = _same_file(os.fstat(check_fd), os.fstat(directory_fd))
        finally:
            os.close(check_fd)

        return reached

    def _archive(
        self, *, request: VisionEventRequest, event_id: str, timestamp: datetime
    ) -> bytes:
        # The display (output) image is "file", the original "inputFile".
        members, references = {}, {}
        for reference, image, quality in (
            ("file", request.output_image, 85),
            ("inputFile", request.input_image, 95),
        ):
            if image is not None:
                member = f"images/{uuid4()}.jpg"
                members[member] = _jpeg(image, quality=quality)
                references[reference] = member
        payload = v1_bundles._build_bundle_payload(
            event_id=event_id,
            timestamp=timestamp,
            event_type=request.event_type,
            solution=request.solution,
            # V1 caps each annotation list at the consumer schema's 1000 items.
            images=_image_entries(
                references,
                v1_bundles._cap_annotation_lists(
                    v1_tensor_events._convert_predictions_to_annotations(
                        request.predictions
                    )
                ),
            ),
            event_data=request.event_data,
            custom_metadata=request.custom_metadata,
        )
        content = v1_bundles._build_tar_bytes(
            payload=payload, image_members=members, mtime=int(timestamp.timestamp())
        )

        return content
