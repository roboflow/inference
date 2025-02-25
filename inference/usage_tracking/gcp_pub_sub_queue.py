import json
import time
from threading import Lock
from typing import Any, Dict, List

from inference.core.logger import logger


class GCPPubSubQueue:
    """
    Push to GCP pub/sub topic
    """

    def __init__(
        self,
        topic: str,
        dry_run: bool = False,
    ):
        self._lock: Lock = Lock()
        self._google_cloud_pub_sub_client = ""
        self._dry_run: bool = dry_run

    def put(self, payload: Any):
        if not isinstance(payload, str):
            try:
                payload = json.dumps(payload)
            except Exception as exc:
                logger.error("Failed to parse payload '%s' to JSON - %s", payload, exc)
                return
        with self._lock:
            try:
                self._increment += 1
                redis_key = f"{self._prefix}:{self._increment}"
                # https://redis.io/docs/latest/develop/interact/transactions/
                redis_pipeline = self._redis_cache.client.pipeline()
                redis_pipeline.set(
                    name=redis_key,
                    value=payload,
                )
                redis_pipeline.zadd(
                    name="UsageCollector",
                    mapping={redis_key: time.time()},
                )
                results = redis_pipeline.execute()
                if not all(results):
                    # TODO: partial insert, retry
                    logger.error(
                        "Failed to store payload and sorted set (partial insert): %s",
                        results,
                    )
            except Exception as exc:
                logger.error("Failed to store usage records '%s', %s", payload, exc)

    @staticmethod
    def full() -> bool:
        return False

    def empty(self) -> bool:
        return True

    def get_nowait(self) -> List[Dict[str, Any]]:
        return []
