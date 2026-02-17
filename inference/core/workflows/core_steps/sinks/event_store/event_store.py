import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from inference.core import logger


class EventStore:
    """Thread-safe in-memory event store with optional JSONL persistence.

    Stores structured events consisting of an event type, arbitrary payload,
    and optional metadata. Events are assigned a unique UUID and ISO 8601
    timestamp on creation. The store can be flushed to disk as a JSONL file
    for durable storage.
    """

    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(
        self,
        event_type: str,
        payload: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append a new event to the store.

        Args:
            event_type: A string label categorising the event
                (e.g. ``"detection"``, ``"alert"``).
            payload: Arbitrary JSON-serialisable data for the event body.
            metadata: Optional dictionary of supplementary information.

        Returns:
            The ``event_id`` (UUID4 string) assigned to the new event.
        """
        event_id = str(uuid.uuid4())
        event: Dict[str, Any] = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
            "metadata": metadata or {},
        }
        with self._lock:
            self._events.append(event)
        logger.debug("EventStore: appended event %s (type=%s)", event_id, event_type)
        return event_id

    def get_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return stored events, optionally filtered by type.

        Args:
            limit: Maximum number of events to return. Events are returned in
                insertion order (oldest first), capped at *limit*.
            event_type: If provided, only events matching this type are
                returned.

        Returns:
            A list of event dictionaries.
        """
        with self._lock:
            if event_type is not None:
                filtered = [
                    e for e in self._events if e["event_type"] == event_type
                ]
                return filtered[:limit]
            return list(self._events[:limit])

    def flush_to_disk(self, file_path: str) -> None:
        """Write all stored events to a JSONL file.

        Each event is serialised as a single JSON line. Parent directories are
        created automatically when they do not exist.

        Args:
            file_path: Destination path for the JSONL output file.
        """
        with self._lock:
            events_snapshot = list(self._events)

        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(file_path, "w") as f:
            for event in events_snapshot:
                f.write(json.dumps(event) + "\n")

        logger.info(
            "EventStore: flushed %d events to %s",
            len(events_snapshot),
            file_path,
        )

    def clear(self) -> None:
        """Remove all events from the store."""
        with self._lock:
            self._events.clear()
        logger.debug("EventStore: cleared all events")

    @property
    def count(self) -> int:
        """Return the number of events currently stored."""
        with self._lock:
            return len(self._events)
