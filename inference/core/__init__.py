import threading
import time

import requests
from packaging import version as packaging_version

from inference.core.env import DISABLE_VERSION_CHECK, VERSION_CHECK_MODE
from inference.core.logger import logger
from inference.core.version import __version__

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
