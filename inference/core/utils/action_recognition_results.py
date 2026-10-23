"""Bound collected action-recognition results before HTTP serialization."""

import json
from typing import Iterable

from inference.core.exceptions import PayloadTooLargeError


def _json_size(value) -> int:
    # ASCII escaping is an upper bound for both JSON response encoders.
    return sum(
        len(chunk) for chunk in json.JSONEncoder(ensure_ascii=True).iterencode(value)
    )


class ActionRecognitionResultBudget:
    """Limit candidate count and the encoded response retained by one request."""

    def __init__(self, *, max_candidates: int, max_bytes: int):
        if max_candidates <= 0 or max_bytes <= 0:
            raise ValueError("Action-recognition result limits must be positive")
        self.max_candidates = max_candidates
        self.max_bytes = max_bytes
        self.candidate_count = 0
        self.candidate_bytes = 2

    def check_candidate_count(self, additional: int) -> None:
        """Refuse candidate allocation beyond the configured count.

        Args:
            additional (int): Number of new or estimated candidate records.

        Raises:
            PayloadTooLargeError: The request exceeds the candidate budget.
        """
        if self.candidate_count + additional > self.max_candidates:
            message = (
                f"Action-recognition candidates exceed the limit of {self.max_candidates}. "
                "Use fewer classes or disable include_candidates. The server administrator "
                "can raise MAX_ACTION_RECOGNITION_CANDIDATES. No partial results were returned."
            )
            raise PayloadTooLargeError(message=message, public_message=message)

    def add_candidates(self, candidates: Iterable) -> None:
        """Account for new candidates before retaining the final response.

        Args:
            candidates (Iterable): Newly converted candidate records.

        Raises:
            PayloadTooLargeError: Candidate count or encoded bytes exceed a limit.
        """
        for candidate in candidates:
            self.check_candidate_count(1)
            self.candidate_count += 1
            self.candidate_bytes += _json_size(candidate.model_dump(mode="json")) + 2
            self._check_bytes(self.candidate_bytes)

    def check_response(self, timeline: Iterable, metadata: dict) -> None:
        """Bound both result lists and response metadata before serialization.

        Args:
            timeline (Iterable): Merged predictions.
            metadata (dict): Response fields other than the two result lists.

        Raises:
            PayloadTooLargeError: The encoded response exceeds its byte budget.
        """
        size = self.candidate_bytes + _json_size(metadata) + 64
        self._check_bytes(size)
        for prediction in timeline:
            size += _json_size(prediction.model_dump(mode="json")) + 2
            self._check_bytes(size)

    def _check_bytes(self, size: int) -> None:
        if size > self.max_bytes:
            message = (
                f"Action-recognition response exceeds the limit of {self.max_bytes} bytes. "
                "Use fewer classes or disable include_candidates. The server administrator "
                "can raise MAX_ACTION_RECOGNITION_RESPONSE_BYTES. No partial results were returned."
            )
            raise PayloadTooLargeError(message=message, public_message=message)
