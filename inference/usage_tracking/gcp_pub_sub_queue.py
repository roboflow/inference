import atexit
from functools import lru_cache
import json
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional

try:
    from google.cloud import pubsub_v1
except ModuleNotFoundError:
    pass
import requests

from inference.core.logger import logger
from inference.core.roboflow_api import build_roboflow_api_headers
from inference.usage_tracking.payload_helpers import APIKey, APIKeyUsage, UsagePayload


class GCPPubSubQueue:
    """
    Push to GCP pub/sub topic
    """

    def __init__(
        self,
        topic: str,
        resolve_workspace_id_url: str,
        workspace_id_response_key: str,
        workspace_id_skip_prefix: int,
    ):
        self._resolve_workspace_id_url: str = resolve_workspace_id_url
        self._workspace_id_response_key: str = workspace_id_response_key
        self._workspace_id_skip_prefix: int = workspace_id_skip_prefix
        self._lock: Lock = Lock()
        self._pub_sub_client = pubsub_v1.PublisherClient()
        self._topic = topic

        self._queue: "Queue[UsagePayload]" = Queue()

        self._terminate_publisher_thread = Event()
        self._publisher_thread: Thread = Thread(target=self._publisher, daemon=True)
        self._publisher_thread.start()

        atexit.register(self._cleanup)

    def put(self, payload: UsagePayload):
        self._queue.put(payload)

    def _publisher(self):
        while not self._terminate_publisher_thread.is_set():
            payload = self._queue.get()
            if payload is None:
                break
            failed_payloads: Dict[APIKey, APIKeyUsage] = {}
            for api_key, api_key_usage in payload.items():
                if not isinstance(api_key_usage, dict) or not api_key_usage:
                    logger.error("Empty usage report for API key %s, dropping", api_key)
                    continue
                try:
                    workspace_id = self._resolve_workspace_id(api_key=api_key)
                except Exception as exc:
                    logger.error(exc)
                    failed_payloads[api_key] = api_key_usage
                    continue
                if not workspace_id:
                    logger.error(
                        "Could not resolve workspace id from API key: %s - dropping usage payload",
                        api_key,
                    )
                    continue
                for resource_usage in api_key_usage.values():
                    try:
                        if "api_key_hash" in resource_usage:
                            del resource_usage["api_key_hash"]
                        resource_usage["api_key"] = api_key[:3] + "***"
                        resource_usage["reporter_api_key"] = api_key[:3] + "***"
                        resource_usage["workspace_id"] = workspace_id
                        resource_usage["hosted"] = True
                        resource_usage = json.dumps(resource_usage).encode()
                        try:
                            future = self._pub_sub_client.publish(
                                topic=self._topic, data=resource_usage
                            )
                            future.result()
                        except Exception as exc:
                            logger.error(
                                "Failed to store usage records '%s', %s",
                                resource_usage,
                                exc,
                            )

                    except Exception as exc:
                        logger.error(
                            "Failed to parse payload '%s' to JSON - %s",
                            resource_usage,
                            exc,
                        )
                        return
            if failed_payloads:
                self.put(failed_payloads)

    @lru_cache(maxsize=100000)
    def _resolve_workspace_id(self, api_key: APIKey) -> Optional[str]:
        headers = build_roboflow_api_headers(
            explicit_headers={"Authorization": f"Bearer {api_key}"}
        )
        ssl_verify = True
        if "localhost" in self._resolve_workspace_id_url.lower():
            ssl_verify = False
        if "127.0.0.1" in self._resolve_workspace_id_url.lower():
            ssl_verify = False

        response = requests.get(
            self._resolve_workspace_id_url,
            verify=ssl_verify,
            headers=headers,
            timeout=1,
        )

        if 400 <= response.status_code < 500:
            return None
        if response.status_code != 200:
            raise Exception(
                f"Failed to resolve workspace id for api key {api_key} due to API error (code {response.status_code})"
            )
        return str(response.json()[self._workspace_id_response_key])[
            self._workspace_id_skip_prefix :
        ]

    @staticmethod
    def full() -> bool:
        return False

    def empty(self) -> bool:
        return True

    def get_nowait(self) -> List[Dict[str, Any]]:
        return []

    def _cleanup(self):
        self._terminate_publisher_thread.set()
        self.put(None)
        self._publisher_thread.join()
