import threading

from inference.core.env import DISABLE_VERSION_CHECK, VERSION_CHECK_MODE
from inference.core.version import (
    check_latest_release_against_current,
    check_latest_release_against_current_continuous,
)

if not DISABLE_VERSION_CHECK:
    if VERSION_CHECK_MODE == "continuous":
        t = threading.Thread(target=check_latest_release_against_current_continuous)
        t.daemon = True
        t.start()
    else:
        check_latest_release_against_current()
