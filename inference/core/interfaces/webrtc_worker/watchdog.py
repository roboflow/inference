import datetime
import threading
import time
from typing import Callable


class Watchdog:
    def __init__(self, timeout_seconds: int, on_timeout: Callable[[], None]):
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = datetime.datetime.now()
        self.on_timeout = on_timeout
        self._thread = threading.Thread(target=self._watchdog_thread)
        self._stopping = False

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopping = True
        self._thread.join()

    def _watchdog_thread(self):
        while not self._stopping:
            if not self.is_alive():
                self.on_timeout()
                break
            time.sleep(0.1)

    def heartbeat(self):
        self.last_heartbeat = datetime.datetime.now()

    def is_alive(self) -> bool:
        return (
            datetime.datetime.now() - self.last_heartbeat
        ).total_seconds() < self.timeout_seconds
