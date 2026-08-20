"""Supervisor-only claim proof storage and authenticated platform mutations.

The browser-facing job security registry deliberately retains only a SHA-256
digest.  Platform mutations have a different requirement: the worker must echo
the plaintext token minted for the *current claim epoch*.  This module keeps
that plaintext out of job payloads and binds cleanup to an opaque per-claim
handle so a late callback from an old attempt cannot delete or reuse a newer
attempt's proof.
"""

from __future__ import annotations

import secrets
import threading
from typing import Any, Callable, Mapping

from security import MissingJobAccessToken, validate_job_id


def retain_claim_proof(
    claimed_job: Mapping[str, Any],
    claim_proofs: "ClaimProofStore",
    register_browser_token: Callable[[str, str | None], Any],
) -> tuple[dict[str, Any], str]:
    """Strip a claim token, retain it, and register only its browser hash."""

    job = dict(claimed_job)
    access_token = job.pop("processorAccessToken", None)
    job_id = validate_job_id(job.get("id"))
    claim_handle = claim_proofs.register(job_id, access_token)
    try:
        register_browser_token(job_id, access_token)
    except Exception:
        claim_proofs.remove(job_id, claim_handle)
        raise
    return job, claim_handle


class ClaimProofStore:
    """Retain plaintext claim proofs only in supervisor-owned memory."""

    def __init__(self, require_tokens: bool):
        self.require_tokens = bool(require_tokens)
        self._claims: dict[str, tuple[str, str | None]] = {}
        self._lock = threading.Lock()

    def register(self, job_id, token=None) -> str:
        job_id = validate_job_id(job_id)
        token = str(token or "").strip()
        if self.require_tokens and not token:
            raise MissingJobAccessToken(
                "managed video job claim is missing processorAccessToken"
            )
        claim_handle = secrets.token_hex(16)
        with self._lock:
            # Replacement is intentional: every successful claim is a new
            # authorization epoch, even when the platform reuses the job ID.
            self._claims[job_id] = (claim_handle, token or None)
        return claim_handle

    def authorization_fields(self, job_id, claim_handle) -> dict[str, str]:
        job_id = validate_job_id(job_id)
        with self._lock:
            current = self._claims.get(job_id)
        if current is None or not secrets.compare_digest(
            current[0], str(claim_handle or "")
        ):
            raise MissingJobAccessToken(
                "video job mutation has no current processor claim proof"
            )
        token = current[1]
        return {"processorAccessToken": token} if token is not None else {}

    def remove(self, job_id, claim_handle) -> bool:
        """Remove only the exact claim epoch represented by ``claim_handle``."""

        try:
            job_id = validate_job_id(job_id)
        except ValueError:
            return False
        with self._lock:
            current = self._claims.get(job_id)
            if current is None or not secrets.compare_digest(
                current[0], str(claim_handle or "")
            ):
                return False
            del self._claims[job_id]
            return True

    def contains(self, job_id, claim_handle) -> bool:
        """Expose lifecycle state for tests without exposing the plaintext."""

        try:
            job_id = validate_job_id(job_id)
        except ValueError:
            return False
        with self._lock:
            current = self._claims.get(job_id)
        return bool(
            current and secrets.compare_digest(current[0], str(claim_handle or ""))
        )


class PlatformJobMutations:
    """The only worker path for claim-bound platform job mutations."""

    def __init__(
        self,
        request: Callable[..., Any],
        claim_proofs: ClaimProofStore,
    ):
        self._request = request
        self._claim_proofs = claim_proofs

    def status(
        self,
        job_id,
        claim_handle,
        payload: Mapping[str, Any],
    ):
        return self._post(job_id, claim_handle, "status", payload)

    def results_upload_urls(
        self,
        job_id,
        claim_handle,
        payload: Mapping[str, Any],
    ):
        return self._post(job_id, claim_handle, "results/upload-urls", payload)

    def results_complete(
        self,
        job_id,
        claim_handle,
        payload: Mapping[str, Any],
    ):
        return self._post(job_id, claim_handle, "results/complete", payload)

    def _post(
        self,
        job_id,
        claim_handle,
        suffix: str,
        payload: Mapping[str, Any],
    ):
        job_id = validate_job_id(job_id)
        body = dict(payload)
        if "processorAccessToken" in body:
            raise ValueError("claim proof fields are supervisor-owned")
        body.update(self._claim_proofs.authorization_fields(job_id, claim_handle))
        return self._request(
            "POST",
            f"/video-jobs/{job_id}/{suffix}",
            json=body,
        )
