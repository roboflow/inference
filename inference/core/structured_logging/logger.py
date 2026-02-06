"""
Structured Event Logger.

This module provides a structured JSON event logger for observability.
It is completely separate from the existing inference logging system.
"""

import json
import random
import sys
from typing import Optional

from inference.core.structured_logging.events import BaseEvent


class StructuredEventLogger:
    """
    Structured JSON event logger for Cloud Logging services.

    Only active when STRUCTURED_LOGGING_ENABLED=True.
    Completely separate from the existing inference logger.

    Outputs JSON to stdout which cloud logging services automatically parse.
    """

    _instance: Optional["StructuredEventLogger"] = None

    def __new__(cls) -> "StructuredEventLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Import here to avoid circular imports
        from inference.core.env import (
            STRUCTURED_LOGGING_ENABLED,
            STRUCTURED_LOGGING_SAMPLE_RATE,
        )

        self._enabled = STRUCTURED_LOGGING_ENABLED
        self._sample_rate = STRUCTURED_LOGGING_SAMPLE_RATE

    @property
    def enabled(self) -> bool:
        """Check if structured logging is enabled."""
        return self._enabled

    @property
    def sample_rate(self) -> float:
        """Get the current sample rate for high-volume events."""
        return self._sample_rate

    def should_sample(self) -> bool:
        """
        Determine if a sampled event should be logged.

        Returns True based on sample rate probability.
        """
        if self._sample_rate >= 1.0:
            return True
        return random.random() < self._sample_rate

    def log_event(self, event: BaseEvent, sampled: bool = False) -> None:
        """
        Log a structured event.

        Args:
            event: The event to log
            sampled: If True, apply sampling rate. If False, always log.
        """
        if not self._enabled:
            return

        if sampled and not self.should_sample():
            return

        try:
            payload = self._format_event(event)
            json_line = json.dumps(payload, default=str)
            print(json_line, file=sys.stdout, flush=True)
        except Exception:
            # Silently fail to avoid breaking inference
            pass

    def _format_event(self, event: BaseEvent) -> dict:
        """
        Format event for structured JSON logging.

        Cloud logging services expect specific fields like 'severity' and 'message'.
        """
        payload = event.to_dict()

        # Cloud Logging expected fields
        formatted_payload = {
            "severity": "INFO",
            "message": event.event_type,
            **payload,
        }

        return formatted_payload


# Singleton instance
structured_event_logger = StructuredEventLogger()
