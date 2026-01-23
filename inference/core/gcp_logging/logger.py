"""
GCP Serverless Structured Logger.

This module provides a structured JSON logger for GCP Cloud Logging.
It is completely separate from the existing inference logging system.
"""

import json
import random
import sys
from typing import Optional

from inference.core.gcp_logging.events import BaseGCPEvent


class GCPServerlessLogger:
    """
    Structured JSON logger for GCP Cloud Logging.

    Only active when GCP_SERVERLESS=True and GCP_LOGGING_ENABLED=True.
    Completely separate from the existing inference logger.

    Outputs JSON to stdout which GCP Cloud Logging automatically parses.
    """

    _instance: Optional["GCPServerlessLogger"] = None

    def __new__(cls) -> "GCPServerlessLogger":
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
            GCP_LOGGING_ENABLED,
            GCP_LOGGING_SAMPLE_RATE,
            GCP_SERVERLESS,
        )

        self._gcp_serverless = GCP_SERVERLESS
        self._enabled = GCP_SERVERLESS and GCP_LOGGING_ENABLED
        self._sample_rate = GCP_LOGGING_SAMPLE_RATE

    @property
    def enabled(self) -> bool:
        """Check if GCP logging is enabled."""
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

    def log_event(self, event: BaseGCPEvent, sampled: bool = False) -> None:
        """
        Log a structured event to GCP Cloud Logging.

        Args:
            event: The event to log
            sampled: If True, apply sampling rate. If False, always log.
        """
        if not self._enabled:
            return

        if sampled and not self.should_sample():
            return

        try:
            payload = self._format_for_gcp(event)
            json_line = json.dumps(payload, default=str)
            print(json_line, file=sys.stdout, flush=True)
        except Exception:
            # Silently fail to avoid breaking inference
            pass

    def _format_for_gcp(self, event: BaseGCPEvent) -> dict:
        """
        Format event for GCP Cloud Logging structured JSON.

        GCP Cloud Logging expects specific fields like 'severity' and 'message'.
        """
        payload = event.to_dict()

        # GCP Cloud Logging expected fields
        gcp_payload = {
            "severity": "INFO",
            "message": event.event_type,
            **payload,
        }

        return gcp_payload


# Singleton instance
gcp_logger = GCPServerlessLogger()
