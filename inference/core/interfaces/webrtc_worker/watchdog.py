import datetime
import threading
import time
from typing import Callable, Optional

from inference.core.env import WEBRTC_MODAL_USAGE_QUOTA_ENABLED
from inference.core.interfaces.webrtc_worker.utils import is_over_quota
from inference.core.logger import logger


class Watchdog:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        on_timeout: Optional[Callable[[], None]] = None,
    ):
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.last_heartbeat = datetime.datetime.now()
        self.on_timeout: Optional[Callable[[], None]] = on_timeout
        self._thread = threading.Thread(target=self._watchdog_thread)
        self._stopping = False
        self._last_log_ts = datetime.datetime.now()
        self._log_interval_seconds = 10
        self._heartbeats = 0

    def start(self):
        logger.info("Starting watchdog with timeout %s", self.timeout_seconds)
        if not self.on_timeout:
            raise ValueError(
                "on_timeout callback must be provided before starting the watchdog"
            )
        self._thread.start()

    def stop(self):
        self._stopping = True
        if self._thread.is_alive():
            self._thread.join()

    def _watchdog_thread(self):
        logger.info("Watchdog thread started")
        while not self._stopping:
            if not self.is_alive():
                logger.error(
                    "Watchdog timeout reached, heartbeats: %s", self._heartbeats
                )
                self.on_timeout(
                    message=f"Timeout reached, heartbeats: {self._heartbeats}"
                )
                break
            if WEBRTC_MODAL_USAGE_QUOTA_ENABLED and is_over_quota(self._api_key):
                logger.error("API key over quota, heartbeats: %s", self._heartbeats)
                self.on_timeout(
                    message=f"API key over quota, heartbeats: {self._heartbeats}"
                )
                break
            time.sleep(1)
        logger.info("Watchdog thread stopped, heartbeats: %s", self._heartbeats)

    def heartbeat(self):
        self.last_heartbeat = datetime.datetime.now()
        self._heartbeats += 1
        if (
            datetime.datetime.now() - self._last_log_ts
        ).total_seconds() > self._log_interval_seconds:
            logger.info("Watchdog heartbeat (%s since last)" % self._heartbeats)
            self._last_log_ts = datetime.datetime.now()
            self._heartbeats = 0

    def is_alive(self) -> bool:
        return (
            datetime.datetime.now() - self.last_heartbeat
        ).total_seconds() < self.timeout_seconds
