import json
import sys
from pathlib import Path

import pytest

PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from claim_proof import (  # noqa: E402
    ClaimProofStore,
    PlatformJobMutations,
    retain_claim_proof,
)
from security import JobSecurityRegistry, MissingJobAccessToken  # noqa: E402


class RequestRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        return {"ok": True}


def registered_claim(job_id="job-a", token="claim-a", *, required=True):
    proofs = ClaimProofStore(require_tokens=required)
    handle = proofs.register(job_id, token)
    recorder = RequestRecorder()
    return proofs, handle, recorder, PlatformJobMutations(recorder, proofs)


def test_claim_proof_is_added_to_every_status_and_results_mutation():
    _, handle, recorder, mutations = registered_claim()

    for state in ("running", "failing", "completed"):
        mutations.status("job-a", handle, {"state": state, "processorId": "p1"})
    mutations.results_upload_urls("job-a", handle, {"processorId": "p1"})
    mutations.results_complete(
        "job-a", handle, {"processorId": "p1", "files": ["meta.json"]}
    )

    assert [call[1] for call in recorder.calls] == [
        "/video-jobs/job-a/status",
        "/video-jobs/job-a/status",
        "/video-jobs/job-a/status",
        "/video-jobs/job-a/results/upload-urls",
        "/video-jobs/job-a/results/complete",
    ]
    assert [call[2]["json"]["processorAccessToken"] for call in recorder.calls] == [
        "claim-a"
    ] * 5


def test_concurrent_jobs_use_only_their_own_claim_proofs():
    proofs = ClaimProofStore(require_tokens=True)
    first = proofs.register("job-a", "claim-a")
    second = proofs.register("job-b", "claim-b")
    recorder = RequestRecorder()
    mutations = PlatformJobMutations(recorder, proofs)

    mutations.status("job-a", first, {"state": "running"})
    mutations.status("job-b", second, {"state": "running"})

    assert recorder.calls[0][2]["json"]["processorAccessToken"] == "claim-a"
    assert recorder.calls[1][2]["json"]["processorAccessToken"] == "claim-b"
    with pytest.raises(MissingJobAccessToken):
        mutations.status("job-a", second, {"state": "running"})


def test_cleanup_removes_plaintext_claim_proof():
    proofs, handle, recorder, mutations = registered_claim()

    assert proofs.contains("job-a", handle)
    assert proofs.remove("job-a", handle)
    assert not proofs.contains("job-a", handle)
    with pytest.raises(MissingJobAccessToken):
        mutations.status("job-a", handle, {"state": "running"})
    assert recorder.calls == []


def test_reclaimed_job_accepts_only_the_new_claim_epoch():
    proofs = ClaimProofStore(require_tokens=True)
    old_handle = proofs.register("job-a", "old-claim")
    new_handle = proofs.register("job-a", "new-claim")
    recorder = RequestRecorder()
    mutations = PlatformJobMutations(recorder, proofs)

    assert not proofs.remove("job-a", old_handle)
    with pytest.raises(MissingJobAccessToken):
        mutations.status("job-a", old_handle, {"state": "running"})
    mutations.status("job-a", new_handle, {"state": "running"})

    body = recorder.calls[0][2]["json"]
    assert body["processorAccessToken"] == "new-claim"
    assert "old-claim" not in json.dumps(recorder.calls)


def test_managed_claim_without_proof_fails_closed_before_any_mutation():
    proofs = ClaimProofStore(require_tokens=True)
    browser_security = JobSecurityRegistry(require_tokens=True)

    with pytest.raises(
        MissingJobAccessToken,
        match="managed video job claim is missing processorAccessToken",
    ):
        retain_claim_proof(
            {"id": "job-a", "sourceUrl": "rtsp://relay/live"},
            proofs,
            browser_security.register_job,
        )

    assert not proofs.contains("job-a", "anything")
    assert not browser_security.authorize("job-a", "")


def test_plaintext_proof_is_removed_from_job_and_browser_registry_keeps_only_hash():
    plaintext = "claim-secret-never-leaves-supervisor"
    proofs = ClaimProofStore(require_tokens=True)
    browser_security = JobSecurityRegistry(require_tokens=True)
    claimed = {
        "id": "job-a",
        "sourceUrl": "rtsp://relay/live",
        "processorAccessToken": plaintext,
    }

    job, handle = retain_claim_proof(claimed, proofs, browser_security.register_job)

    assert "processorAccessToken" not in job
    assert plaintext not in json.dumps(job)
    assert plaintext not in repr(browser_security.__dict__)
    assert browser_security.authorize("job-a", plaintext)
    assert proofs.contains("job-a", handle)


def test_callers_cannot_override_supervisor_owned_proof_field():
    _, handle, recorder, mutations = registered_claim()

    with pytest.raises(ValueError, match="supervisor-owned"):
        mutations.status(
            "job-a",
            handle,
            {"state": "running", "processorAccessToken": "attacker"},
        )

    assert recorder.calls == []


def test_processor_routes_every_platform_job_mutation_through_claim_client():
    source = (PROCESSOR_DIR / "processor.py").read_text()

    # Heartbeats/completion and explicit failures share the status helper.
    assert source.count("job_mutations.status(") == 2
    assert source.count("job_mutations.results_upload_urls(") == 1
    assert source.count("job_mutations.results_complete(") == 1
    # Claim is the only video-job platform call allowed to bypass the
    # claim-bound mutation client because it obtains the proof itself.
    assert source.count('"/video-jobs/claim"') == 1
    assert "results/upload-urls" not in source
    assert "results/complete" not in source
