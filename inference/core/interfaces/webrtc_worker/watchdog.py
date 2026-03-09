import datetime
import threading
import time
from typing import Callable, Optional

import requests

from inference.core.env import (
    WEBRTC_MODAL_USAGE_QUOTA_ENABLED,
    WEBRTC_SESSION_HEARTBEAT_INTERVAL_SECONDS,
)
from inference.core.interfaces.webrtc_worker.utils import is_over_quota
from inference.core.logger import logger


class Watchdog:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: int,
        on_timeout: Optional[Callable[[], None]] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        heartbeat_url: Optional[str] = None,
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
        self._total_heartbeats = 0
        self._workspace_id = workspace_id
        self._session_id = session_id
        self._heartbeat_url = heartbeat_url
        self._last_session_heartbeat_ts = datetime.datetime.now()

    @property
    def total_heartbeats(self) -> int:
        return self._total_heartbeats

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
        self._send_session_heartbeat_stop()

    def _send_session_heartbeat(self):
        """Send heartbeat to keep the session alive in the quota system.

        This is used to sign that the session is alive so the system
        doesnt allow more than N concurrent sessions from a single workspace.
        """
        if not all(
            [
                self._heartbeat_url,
                self._workspace_id,
                self._session_id,
            ]
        ):
            logger.info(
                "Skipping session heartbeat: url=%s, workspace=%s, session=%s",
                bool(self._heartbeat_url),
                bool(self._workspace_id),
                bool(self._session_id),
            )
            return

        # Mask API key for logging (show first 4 and last 4 chars)
        api_key_masked = (
            f"{self._api_key[:4]}...{self._api_key[-4:]}"
            if self._api_key and len(self._api_key) > 8
            else "***"
        )

        logger.info(
            "[HEARTBEAT] Sending request: url=%s, workspace=%s, session=%s, api_key=%s",
            self._heartbeat_url,
            self._workspace_id,
            self._session_id,
            api_key_masked,
        )

        try:
            response = requests.post(
                self._heartbeat_url,
                json={
                    "session_id": self._session_id,
                    "api_key": self._api_key,
                },
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if response.status_code == 200:
                logger.info(
                    "[HEARTBEAT] Success: workspace=%s, session=%s",
                    self._workspace_id,
                    self._session_id,
                )
            else:
                # Log detailed error info
                try:
                    response_body = response.text[:500]  # Limit response body size
                except Exception:
                    response_body = "<could not read body>"

                logger.error(
                    "[HEARTBEAT] Failed: url=%s, status=%s, "
                    "workspace=%s, session=%s, response_headers=%s, response_body=%s",
                    self._heartbeat_url,
                    response.status_code,
                    self._workspace_id,
                    self._session_id,
                    dict(response.headers),
                    response_body,
                )
        except requests.exceptions.Timeout as e:
            logger.error(
                "[HEARTBEAT] Timeout: url=%s, workspace=%s, session=%s, error=%s",
                self._heartbeat_url,
                self._workspace_id,
                self._session_id,
                e,
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(
                "[HEARTBEAT] Connection error: url=%s, workspace=%s, session=%s, error=%s",
                self._heartbeat_url,
                self._workspace_id,
                self._session_id,
                e,
            )
        except Exception as e:
            logger.error(
                "[HEARTBEAT] Unexpected error: url=%s, workspace=%s, session=%s, "
                "error_type=%s, error=%s",
                self._heartbeat_url,
                self._workspace_id,
                self._session_id,
                type(e).__name__,
                e,
                exc_info=True,
            )

    def _send_session_heartbeat_stop(self):
        """Send session end to immediately free the quota slot."""
        if not all([self._heartbeat_url, self._session_id]):
            logger.info(
                "[HEARTBEAT_STOP] Skipping: url=%s, session=%s",
                bool(self._heartbeat_url),
                bool(self._session_id),
            )
            return

        url = self._heartbeat_url + "/end"

        logger.info(
            "[HEARTBEAT_STOP] Sending request: url=%s, workspace=%s, session=%s",
            url,
            self._workspace_id,
            self._session_id,
        )

        try:
            response = requests.post(
                url,
                json={
                    "session_id": self._session_id,
                    "api_key": self._api_key,
                },
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if response.status_code == 200:
                logger.info(
                    "[HEARTBEAT_STOP] Success: workspace=%s, session=%s",
                    self._workspace_id,
                    self._session_id,
                )
            else:
                try:
                    response_body = response.text[:500]
                except Exception:
                    response_body = "<could not read body>"

                logger.error(
                    "[HEARTBEAT_STOP] Failed: url=%s, status=%s, "
                    "workspace=%s, session=%s, response_headers=%s, response_body=%s",
                    url,
                    response.status_code,
                    self._workspace_id,
                    self._session_id,
                    dict(response.headers),
                    response_body,
                )
        except requests.exceptions.Timeout as e:
            logger.error(
                "[HEARTBEAT_STOP] Timeout: url=%s, workspace=%s, session=%s, error=%s",
                url,
                self._workspace_id,
                self._session_id,
                e,
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(
                "[HEARTBEAT_STOP] Connection error: url=%s, workspace=%s, session=%s, error=%s",
                url,
                self._workspace_id,
                self._session_id,
                e,
            )
        except Exception as e:
            logger.error(
                "[HEARTBEAT_STOP] Unexpected error: url=%s, workspace=%s, session=%s, "
                "error_type=%s, error=%s",
                url,
                self._workspace_id,
                self._session_id,
                type(e).__name__,
                e,
                exc_info=True,
            )

    def _watchdog_thread(self):
        logger.info("Watchdog thread started")

        # Send first heartbeat immediately to prevent session expiry if we have a cold start
        self._send_session_heartbeat()
        self._last_session_heartbeat_ts = datetime.datetime.now()

        while not self._stopping:
            if not self.is_alive():
                logger.error(
                    "Watchdog timeout reached, heartbeats: %s", self._total_heartbeats
                )
                self.on_timeout(
                    message=f"Timeout reached, heartbeats: {self._total_heartbeats}"
                )
                break
            if WEBRTC_MODAL_USAGE_QUOTA_ENABLED and is_over_quota(self._api_key):
                logger.error(
                    "API key over quota, heartbeats: %s", self._total_heartbeats
                )
                self.on_timeout(
                    message=f"API key over quota, heartbeats: {self._total_heartbeats}"
                )
                break

            if (
                datetime.datetime.now() - self._last_session_heartbeat_ts
            ).total_seconds() > WEBRTC_SESSION_HEARTBEAT_INTERVAL_SECONDS:
                self._send_session_heartbeat()
                self._last_session_heartbeat_ts = datetime.datetime.now()
            time.sleep(1)
        logger.info("Watchdog thread stopped, heartbeats: %s", self._total_heartbeats)

    def heartbeat(self):
        self.last_heartbeat = datetime.datetime.now()
        self._heartbeats += 1
        self._total_heartbeats += 1
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
