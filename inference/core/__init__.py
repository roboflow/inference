import threading
import time

import requests

from inference.core.env import DISABLE_VERSION_CHECK, VERSION_CHECK_MODE
from inference.core.logger import logger
from inference.core.version import __version__

latest_release = None
last_checked = 0
cache_duration = 86400  # 24 hours
log_frequency = 300  # 5 minutes


def get_latest_release_version():
    global latest_release, last_checked
    now = time.time()
    if latest_release is None or now - last_checked > cache_duration:
        try:
            logger.debug("Checking for latest inference release version...")
            response = requests.get(
                "https://api.github.com/repos/roboflow/inference/releases/latest"
            )
            response.raise_for_status()
            latest_release = response.json()["tag_name"].lstrip("v")
            last_checked = now
        except requests.exceptions.RequestException:
            pass


def check_latest_release_against_current():
    get_latest_release_version()
    if latest_release is not None and latest_release != __version__:
        logger.warning(
            f"Your inference package version {__version__} is out of date! Please upgrade to version {latest_release} of inference for the latest features and bug fixes by running `pip install --upgrade roboflow-inference`."
        )


def check_latest_release_against_current_continuous():
    while True:
        check_latest_release_against_current()
        time.sleep(log_frequency)


if not DISABLE_VERSION_CHECK:
    if VERSION_CHECK_MODE == "continuous":
        t = threading.Thread(target=check_latest_release_against_current_continuous)
        t.daemon = True
        t.start()
    else:
        check_latest_release_against_current()
