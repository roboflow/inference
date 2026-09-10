"""The port through which Workflows load images.

Every image source carrying a *host policy* goes through here:

* URL fetching - the SSRF surface. `inference.core.utils.image_utils` enforces
  `OFFLINE_MODE`, `ALLOW_URL_INPUT`, the scheme and FQDN rules, the
  `WHITELISTED_/BLACKLISTED_DESTINATIONS_FOR_URL_INPUT` lists, per-hop redirect
  re-validation, `MAX_IMAGE_URL_REDIRECTS`, and non-global-address rejection
  with IP pinning (`inference.core.utils.url_input`).
* Local-filesystem reads - `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM`.
* Pickled-numpy payloads - `ALLOW_NUMPY_INPUT`, the gate before `pickle.loads`.

None of those rules is reimplemented here and none may be. The Roboflow
inference server injects its guarded implementation
(`inference.core.interfaces.workflows_image_codec`); standalone Workflows falls
back to `WorkflowsLocalImageCodec`, which refuses all three outright rather than
doing them unguarded.

TWO INJECTION PATHS, by consumer:

* Engine-scoped, via `init_parameters["workflows_core.image_codec"]`: the
  runtime-input image deserializers. `ExecutionEngineV1.init` rebinds the image
  kind deserializer on a COPIED `kinds_deserializers` map (the map is served out
  of `COMPILATION_CACHE`). This is the request boundary and the only genuinely
  per-engine consumer.
* Process-level, via this registry: `WorkflowImageData` (constructed at six
  production sites outside any engine, plus three alternative constructors), the
  ~24 module-level free functions in VLM blocks that call `load_image`, and
  `modal/modal_app.py`'s sandbox deserialization. None of those has an engine in
  scope.

In production both paths receive the same server singleton, so they cannot
disagree. `set_image_codec` is lock-synchronised and set-once: re-installing the
IDENTICAL object is a no-op, installing a different one raises. A process
serving two codecs would apply two different SSRF policies depending on request
timing, which is a bug, not a configuration.
"""

import threading
from typing import Any, Optional, Protocol, Tuple, Union

import cv2
import numpy as np

from inference.core.workflows.errors import (
    WorkflowEnvironmentConfigurationError,
    WorkflowImageLoadError,
)
from inference.core.workflows.utils.image_encoding import (
    choose_image_decoding_flags,
    convert_gray_image_to_bgr,
    decode_base64_image,
    decode_encoded_image_bytes,
    ensure_valid_numpy_image,
)

_LOCAL_FILESYSTEM_REFUSAL = (
    "Loading images from the local filesystem is not available: no host image "
    "codec is installed, so Workflows refuses filesystem access rather than "
    "reading a path unguarded."
)
_URL_REFUSAL = (
    "Loading images from a URL is not available: no host image codec is "
    "installed, so Workflows refuses network access rather than fetching a URL "
    "without the host's SSRF policy."
)
_PICKLE_REFUSAL = (
    "Serialised numpy image payloads are not supported: deserializing them is "
    "arbitrary code execution and is gated by the host, not by Workflows."
)


class ImageCodec(Protocol):
    """Host-supplied image loading.

    Every signature mirrors the `inference.core.utils.image_utils` function it
    fronts, `cv_imread_flags` included - dropping it silently re-enables EXIF
    auto-orientation on the paths that asked for it to be off.

    Deliberately NOT `runtime_checkable`: nothing does an `isinstance` check, and
    a structural check would not verify the guarantees that matter here.
    """

    def load_image(
        self, value: Any, disable_preproc_auto_orient: bool = False
    ) -> Tuple[np.ndarray, bool]:
        """Load an `inference`-format image and return `(bgr_image, is_bgr)`.

        `value` is a `{"type": ..., "value": ...}` dict as produced by
        `WorkflowImageData.to_inference_format()` - `numpy_object`, `base64`,
        `url` or `file` - or a bare `np.ndarray` / string. The `url` and `file`
        branches carry the host's URL and filesystem policy.
        """
        ...

    def fetch_url(
        self, value: str, cv_imread_flags: int = cv2.IMREAD_COLOR
    ) -> np.ndarray:
        """Fetch an image over http(s) under the host's SSRF policy, BGR."""
        ...

    def decode_string(
        self,
        value: Union[str, bytes, bytearray],
        cv_imread_flags: int = cv2.IMREAD_COLOR,
    ) -> Tuple[np.ndarray, bool]:
        """Decode an in-memory payload (base64 / raw encoded bytes) to
        `(bgr_image, is_bgr)`. The host implementation also honours its
        `ALLOW_NUMPY_INPUT` pickle gate."""
        ...

    def ensure_local_file_load_allowed(self, path: str) -> None:
        """Raise unless the host permits reading images off the local disk.

        A gate rather than a loader on purpose: Workflows reads local files with
        two decoders (`cv2.imread` for numpy, `torchvision.io.read_file` +
        `decode_image` for tensors) whose EXIF behaviour must not change, and
        both belong inside Workflows. Only the *permission* is the host's, and on
        the server side it has exactly one owner:
        `inference.core.utils.image_utils.ensure_local_file_load_allowed`.
        """
        ...


class WorkflowsLocalImageCodec:
    """The standalone default: in-memory decoding only, everything else refused.

    Decodes `np.ndarray`, `{"type": "numpy_object"}`, `{"type": "base64"}` and
    raw encoded bytes / base64 strings, forwarding `cv_imread_flags` on every
    path. Refuses URL fetching, local-filesystem reads and serialised numpy
    payloads with `WorkflowImageLoadError`. It never opens a socket, never
    touches the filesystem and never imports `pickle`.
    """

    def load_image(
        self, value: Any, disable_preproc_auto_orient: bool = False
    ) -> Tuple[np.ndarray, bool]:
        flags = choose_image_decoding_flags(
            disable_preproc_auto_orient=disable_preproc_auto_orient
        )
        if isinstance(value, dict) and "type" in value and "value" in value:
            declared, payload = value["type"], value["value"]
            if declared == "url":
                raise WorkflowImageLoadError(
                    public_message=_URL_REFUSAL,
                    context="workflow_execution | image_loading",
                )
            if declared == "file":
                raise WorkflowImageLoadError(
                    public_message=_LOCAL_FILESYSTEM_REFUSAL,
                    context="workflow_execution | image_loading",
                )
            if declared == "numpy":
                raise WorkflowImageLoadError(
                    public_message=_PICKLE_REFUSAL,
                    context="workflow_execution | image_loading",
                )
            if declared == "numpy_object":
                return (
                    convert_gray_image_to_bgr(ensure_valid_numpy_image(payload)),
                    True,
                )
            if declared == "base64":
                return (
                    convert_gray_image_to_bgr(
                        decode_base64_image(payload, cv_imread_flags=flags)
                    ),
                    True,
                )
            raise WorkflowImageLoadError(
                public_message=f"Unsupported declared image type: `{declared}`.",
                context="workflow_execution | image_loading",
            )
        if isinstance(value, (np.ndarray, np.generic)):
            return convert_gray_image_to_bgr(ensure_valid_numpy_image(value)), True
        if isinstance(value, str) and value.startswith("http"):
            raise WorkflowImageLoadError(
                public_message=_URL_REFUSAL,
                context="workflow_execution | image_loading",
            )
        if isinstance(value, (str, bytes, bytearray)):
            # Flags are FORWARDED, not recomputed: `disable_preproc_auto_orient`
            # must survive the inferred-type path exactly as it does in
            # `image_utils.load_image_with_inferred_type`. The grayscale->BGR
            # promotion belongs to THIS layer (image_utils.py:111), not to
            # `decode_string` - see the note there.
            decoded, is_bgr = self.decode_string(value, cv_imread_flags=flags)
            return convert_gray_image_to_bgr(decoded), is_bgr
        raise WorkflowImageLoadError(
            public_message=f"Could not load an image from a value of type "
            f"`{type(value).__name__}`.",
            context="workflow_execution | image_loading",
        )

    def fetch_url(
        self, value: str, cv_imread_flags: int = cv2.IMREAD_COLOR
    ) -> np.ndarray:
        raise WorkflowImageLoadError(
            public_message=_URL_REFUSAL,
            context="workflow_execution | image_loading",
        )

    def decode_string(
        self,
        value: Union[str, bytes, bytearray],
        cv_imread_flags: int = cv2.IMREAD_COLOR,
    ) -> Tuple[np.ndarray, bool]:
        # NO grayscale->BGR promotion here. `attempt_loading_image_from_string`
        # (image_utils.py:238-262) returns the decoder's result untouched and
        # `load_image` promotes at image_utils.py:111; doing it here diverges on
        # cv2.IMREAD_GRAYSCALE ((6, 9) upstream vs (6, 9, 3) here).
        try:
            return decode_base64_image(value, cv_imread_flags=cv_imread_flags), True
        except WorkflowImageLoadError:
            pass
        try:
            return (
                decode_encoded_image_bytes(value, cv_imread_flags=cv_imread_flags),
                True,
            )
        except (WorkflowImageLoadError, TypeError, ValueError):
            pass
        raise WorkflowImageLoadError(
            public_message="Input image format could not be inferred from the payload. "
            + _PICKLE_REFUSAL,
            context="workflow_execution | image_loading",
        )

    def ensure_local_file_load_allowed(self, path: str) -> None:
        raise WorkflowImageLoadError(
            public_message=_LOCAL_FILESYSTEM_REFUSAL,
            context="workflow_execution | image_loading",
        )


_DEFAULT_CODEC = WorkflowsLocalImageCodec()
_INSTALLED_CODEC: Optional[ImageCodec] = None
_INSTALL_LOCK = threading.Lock()


def get_image_codec() -> ImageCodec:
    """The process-wide codec, or the refusing default if none is installed."""
    codec = _INSTALLED_CODEC
    return codec if codec is not None else _DEFAULT_CODEC


def set_image_codec(codec: ImageCodec) -> None:
    """Install the host's codec for this process.

    Re-installing the IDENTICAL object is a no-op - the server calls this once
    per request with a module-level singleton. Installing a DIFFERENT object
    raises: a process serving two codecs would apply two different SSRF policies
    depending on request timing.

    The check and the assignment are under one lock. Without it a concurrent
    pair of installs can both observe `None` and both write, which is exactly
    what the round-1 review reproduced.
    """
    global _INSTALLED_CODEC
    with _INSTALL_LOCK:
        if _INSTALLED_CODEC is not None and _INSTALLED_CODEC is not codec:
            raise WorkflowEnvironmentConfigurationError(
                public_message="A different image codec is already installed for "
                "this process. The image codec is process-wide and may only be "
                "installed once; re-installing the identical object is allowed.",
                context="workflow_compilation | engine_initialisation",
            )
        _INSTALLED_CODEC = codec


def reset_image_codec() -> None:
    """Clear the installed codec. Tests only - never call this from a server."""
    global _INSTALLED_CODEC
    with _INSTALL_LOCK:
        _INSTALLED_CODEC = None
