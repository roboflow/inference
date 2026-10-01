"""The Roboflow inference server's implementation of the Workflows `ImageCodec`.

Four thin forwards to `inference.core.utils.image_utils`, so every guard the
server already owns keeps applying to images loaded inside Workflows:
`OFFLINE_MODE`, `ALLOW_URL_INPUT`, the scheme and FQDN rules, the
`WHITELISTED_/BLACKLISTED_DESTINATIONS_FOR_URL_INPUT` lists, per-hop redirect
re-validation with `MAX_IMAGE_URL_REDIRECTS`, non-global-address rejection with
IP pinning (`inference.core.utils.url_input`), `ALLOW_NUMPY_INPUT` in front of
`pickle.loads`, and `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM`.

NOTHING here reimplements a rule, and nothing here holds a snapshot of one:

* the MODULE is imported and its attributes are called, never `from ... import
  load_image_from_url`. A function alias captured at import time would make
  `mock.patch.object(image_utils, "load_image_from_url")` a no-op, so tests
  believing they had neutered a guard would silently exercise the real one;
* the local-filesystem permission is `image_utils.ensure_local_file_load_allowed`,
  the same function `load_image_with_known_type` calls. One flag read, one
  message, one owner.

Two entry points for the two injection paths (see
`inference.core.workflows.prototypes.image_codec` for which consumer uses which):
`resolve_image_codec()` for `init_parameters["workflows_core.image_codec"]`, and
`install_guarded_image_codec()` for the process registry. Both hand out the same
singleton, so the two paths cannot disagree.
"""

from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np

from inference.core.utils import image_utils
from inference.core.workflows.prototypes.image_codec import ImageCodec, set_image_codec


class ServerImageCodec:
    """`ImageCodec` backed by the server's guarded image loaders."""

    def load_image(
        self, value: Any, disable_preproc_auto_orient: bool = False
    ) -> Tuple[np.ndarray, bool]:
        return image_utils.load_image(
            value, disable_preproc_auto_orient=disable_preproc_auto_orient
        )

    def fetch_url(
        self, value: str, cv_imread_flags: int = cv2.IMREAD_COLOR
    ) -> np.ndarray:
        return image_utils.load_image_from_url(
            value=value, cv_imread_flags=cv_imread_flags
        )

    def decode_string(
        self,
        value: Union[str, bytes, bytearray],
        cv_imread_flags: int = cv2.IMREAD_COLOR,
    ) -> Tuple[np.ndarray, bool]:
        return image_utils.attempt_loading_image_from_string(
            value=value, cv_imread_flags=cv_imread_flags
        )

    def ensure_local_file_load_allowed(self, path: str) -> None:
        # `path` is accepted for the port's sake and deliberately not consulted:
        # the server's policy is a single global switch, and restating it here
        # would create the second owner the constraints forbid.
        image_utils.ensure_local_file_load_allowed()


GUARDED_IMAGE_CODEC = ServerImageCodec()


def resolve_image_codec() -> ServerImageCodec:
    """The codec the composition roots put in `init_parameters`."""
    return GUARDED_IMAGE_CODEC


def install_guarded_image_codec() -> None:
    """Make the guarded codec the process-wide codec.

    A module-level singleton, so repeated calls (one per request at the HTTP
    roots) are identity no-ops rather than conflicting installs.
    """
    set_image_codec(GUARDED_IMAGE_CODEC)


def bind_image_codec(init_parameters: Dict[str, Any]) -> ImageCodec:
    """Give BOTH injection paths the same codec object, and return it.

    Call this at every composition root AFTER any caller-supplied overrides have
    been merged into `init_parameters` - `inference_cli/lib/workflows/
    local_image_adapter.py:453-454` merges them after the dict is built, so a
    codec written before that point would be silently replaced on Path A while
    Path B kept the guarded one. The two paths are not alternatives for the same
    image: Path A deserializes the input and stores `image_reference`
    (`deserializers.py:138`), and a later block re-loads that reference through
    Path B (`base.py:573`, or `to_inference_format()` -> `load_image`), so a
    split would apply two different SSRF policies inside one run.

    `setdefault` honours a deliberate override; `set_image_codec` then makes the
    same object process-wide and, by its set-once rule, turns an override that
    disagrees with an already-installed codec into a loud
    `WorkflowEnvironmentConfigurationError` instead of a silent split.
    """
    codec = init_parameters.setdefault(
        "workflows_core.image_codec", GUARDED_IMAGE_CODEC
    )
    set_image_codec(codec)
    return codec
