import threading

from inference.core.env import DISABLE_VERSION_CHECK
from inference.core.version import check_latest_release_against_current

if not DISABLE_VERSION_CHECK:
    threading.Thread(target=check_latest_release_against_current).start()
