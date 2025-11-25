import datetime
import threading
import time
from typing import Callable

from inference.core.logger import logger


class Watchdog:
    def __init__(self, timeout_seconds: int, on_timeout: Callable[[], None]):
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = datetime.datetime.now()
        self.on_timeout = on_timeout
        self._thread = threading.Thread(target=self._watchdog_thread)
        self._stopping = False
        self._last_log_ts = datetime.datetime.now()
        self._log_interval_seconds = 10
        self._heartbeats = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopping = True
        self._thread.join()

    def _watchdog_thread(self):
        while not self._stopping:
            if not self.is_alive():
                logger.error("Watchdog timeout reached")
                self.on_timeout()
                break
            time.sleep(0.1)

    def heartbeat(self):
        self.last_heartbeat = datetime.datetime.now()
        self._heartbeats += 1
        if (
            datetime.datetime.now() - self._last_log_ts
        ).total_seconds() > self._log_interval_seconds:
            logger.info("Watchdog heartbeat (%s since last)", self._heartbeats)
            self._last_log_ts = datetime.datetime.now()
            self._heartbeats = 0

    def is_alive(self) -> bool:
        return (
            datetime.datetime.now() - self.last_heartbeat
        ).total_seconds() < self.timeout_seconds
