import threading
import time

import requests
from packaging import version as packaging_version

from inference.core.env import DISABLE_VERSION_CHECK, VERSION_CHECK_MODE

# Hand the Workflows module its configuration before anything can READ it.
# The invariant: `install_workflows_configuration()` runs before any import
# of `inference.core.workflows.environment` (the constants facade) or any
# other configuration-consuming workflows module, so `core_steps/loader.py`'s
# import-time tensor branches and every facade constant see the server's
# values. A few configuration-independent workflows modules are already on
# the bootstrap path above this point (`inference.core.env` ->
# `utils/environment.py` -> `core/exceptions.py` ->
# `workflows/prototypes/platform_errors.py`, and the builder's own import of
# `workflows/configuration.py`); they must stay configuration-independent -
# none of them may import the facade. `inference.core.interfaces
# .workflows_configuration` imports only `inference.core.env` (already fully
# imported above) and `inference.core.workflows.configuration`, so this adds
# no import weight.
from inference.core.interfaces.workflows_configuration import (
    install_workflows_configuration,
)
from inference.core.logger import logger
from inference.core.version import __version__
from inference.core.workflows.prototypes.image_codec import (
    set_default_image_codec_factory,
)


def _resolve_workflows_image_codec():
    from inference.core.interfaces.workflows_image_codec import resolve_image_codec

    return resolve_image_codec()


install_workflows_configuration()
# Direct WorkflowImageData callers need the same guarded loader as the server.
# Resolve it on first use so startup does not import the server image utilities
# and callers can still bind an explicit codec before loading an image.
set_default_image_codec_factory(_resolve_workflows_image_codec)

latest_release = None
last_checked = 0
cache_duration = 86400  # 24 hours
log_frequency = 300  # 5 minutes


def get_latest_release_version():
    global latest_release, last_checked
    if DISABLE_VERSION_CHECK:
        # guard at the network call itself so every caller is covered
        # (github.com is unreachable behind SECURE_GATEWAY / air gaps)
        return
    now = time.time()
    if latest_release is None or now - last_checked > cache_duration:
        try:
            logger.debug("Checking for latest inference release version...")
            response = requests.get(
                "https://api.github.com/repos/roboflow/inference/releases/latest",
                timeout=5,
            )
            response.raise_for_status()
            latest_release = response.json()["tag_name"].lstrip("v")
            last_checked = now
        except (requests.exceptions.RequestException, KeyError, ValueError, TypeError):
            # KeyError/ValueError/TypeError: a 200 response whose body is not
            # the expected GitHub payload (proxy interstitials, rate-limit
            # bodies) must degrade like a network failure, not crash the
            # import or kill the continuous-check thread.
            pass


def check_latest_release_against_current():
    get_latest_release_version()
    if latest_release is not None and latest_release != __version__:

        running_ver = packaging_version.parse(__version__)
        current_ver = packaging_version.parse(latest_release)

        if running_ver < current_ver:
            logger.warning(
                f"Your inference package version {__version__} is out of date! Please upgrade to version {latest_release} of inference for the latest features and bug fixes by running `pip install --upgrade inference`."
            )


def check_latest_release_against_current_continuous():
    while True:
        check_latest_release_against_current()
        time.sleep(log_frequency)


if not DISABLE_VERSION_CHECK:
    if VERSION_CHECK_MODE == "continuous":
        _version_check_target = check_latest_release_against_current_continuous
    else:
        # run the single check off the import path too - a slow or blackholed
        # network must not delay interpreter startup
        _version_check_target = check_latest_release_against_current
    t = threading.Thread(target=_version_check_target)
    t.daemon = True
    t.start()
