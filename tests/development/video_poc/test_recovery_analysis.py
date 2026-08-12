import json
import sys
from pathlib import Path

import pytest


ANALYSIS_DIR = (
    Path(__file__).resolve().parents[3]
    / "development"
    / "video_poc"
    / "benchmarks"
    / "analysis"
)
sys.path.insert(0, str(ANALYSIS_DIR))

from recovery import canonical_digest, join_recovery, read_evidence  # noqa: E402


def evidence_events(run_id="recovery-run-001"):
    payloads = [
        (
            "plan",
            {
                "benchmarkRunId": run_id,
                "fault": {
                    "type": "processor-pod-loss",
                    "phase": "steady-state",
                },
                "planDigest": "plan-sha",
            },
            "2026-08-12T20:00:00Z",
        ),
        (
            "trigger",
            {
                "podName": "worker-old",
                "benchmarkJob": {"id": "job-a", "ordinal": 1},
            },
            "2026-08-12T20:00:01Z",
        ),
        (
            "target-captured",
            {"name": "worker-old", "uid": "uid-old"},
            "2026-08-12T20:00:02Z",
        ),
        (
            "target-verified",
            {"name": "worker-old", "uid": "uid-old"},
            "2026-08-12T20:00:03Z",
        ),
        (
            "fault-requested",
            {"podName": "worker-old", "podUid": "uid-old"},
            "2026-08-12T20:00:04Z",
        ),
        (
            "fault-applied",
            {"podName": "worker-old", "podUid": "uid-old"},
            "2026-08-12T20:00:05Z",
        ),
        (
            "recovered",
            {"name": "worker-new", "uid": "uid-new"},
            "2026-08-12T20:00:10Z",
        ),
    ]
    events = []
    previous = "0" * 64
    for index, (event_type, payload, at) in enumerate(payloads):
        event = {
            "schemaVersion": 1,
            "sequence": index,
            "at": at,
            "type": event_type,
            "payload": payload,
            "previousDigest": previous,
        }
        event["digest"] = canonical_digest(event)
        events.append(event)
        previous = event["digest"]
    complete = {
        "schemaVersion": 1,
        "sequence": len(events),
        "at": "2026-08-12T20:00:15Z",
        "type": "complete",
        "payload": {"outcome": "passed", "chainHead": previous},
        "previousDigest": previous,
    }
    complete["digest"] = canonical_digest(complete)
    events.append(complete)
    return events


def test_join_reports_verified_frame_recovery_upper_bound(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in evidence_events())
    )
    report = {
        "runId": "recovery-run-001",
        "success": True,
        "endedAt": "2026-08-12T20:00:14Z",
        "checkpoint": {"phase": "complete"},
        "recoveries": [
            {
                "index": 1,
                "sourcePhase": "measurement",
                "jobIds": ["job-a"],
                "outcome": "recovered",
                "before": [
                    {"id": "job-a", "processorId": "worker-old"}
                ],
                "after": [
                    {"id": "job-a", "processorId": "worker-new"}
                ],
                "runningObservedAt": "2026-08-12T20:00:11Z",
                "progressVerifiedAt": "2026-08-12T20:00:13Z",
                "observedControlPlaneRecoverySeconds": 4,
                "assertions": {
                    "job-a": {
                        "processorChanged": True,
                        "framesAdvancedAfterRunning": True,
                        "requeueIdentityChanged": True,
                    }
                },
            }
        ],
    }

    result = join_recovery(report, read_evidence(path))

    assert result["recoveryCount"] == 1
    recovery = result["recoveries"][0]
    assert recovery["verifiedFrameRecoveryUpperBoundSeconds"] == 9.0
    assert recovery["observedControlPlaneRecoverySeconds"] == 4
    assert recovery["oldProcessorId"] == "worker-old"
    assert recovery["newProcessorId"] == "worker-new"
    assert "upper bound" in result["measurementSemantics"]


def test_read_evidence_rejects_tampering(tmp_path):
    events = evidence_events()
    events[1]["payload"]["podName"] = "tampered"
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))

    with pytest.raises(ValueError, match="digest"):
        read_evidence(path)


def test_join_rejects_mismatched_run_and_missing_verified_progress():
    evidence = evidence_events()
    with pytest.raises(ValueError, match="run ID"):
        join_recovery({"runId": "another", "recoveries": []}, evidence)
    with pytest.raises(ValueError, match="exactly one recovery"):
        join_recovery(
            {
                "runId": "recovery-run-001",
                "success": False,
                "endedAt": "2026-08-12T20:00:14Z",
                "checkpoint": {"phase": "complete"},
                "recoveries": [{"outcome": "timeout"}],
            },
            evidence,
        )


def matching_report():
    return {
        "runId": "recovery-run-001",
        "success": True,
        "endedAt": "2026-08-12T20:00:14Z",
        "checkpoint": {"phase": "complete"},
        "recoveries": [
            {
                "index": 1,
                "sourcePhase": "measurement",
                "jobIds": ["job-a"],
                "outcome": "recovered",
                "before": [{"id": "job-a", "processorId": "worker-old"}],
                "after": [{"id": "job-a", "processorId": "worker-new"}],
                "progressVerifiedAt": "2026-08-12T20:00:13Z",
                "assertions": {
                    "job-a": {
                        "processorChanged": True,
                        "framesAdvancedAfterRunning": True,
                        "requeueIdentityChanged": True,
                    }
                },
            }
        ],
    }


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda events: events[0]["payload"]["fault"].update(
                {"type": "relay-pod-loss"}
            ),
            "processor-pod-loss",
        ),
        (
            lambda events: events[-1]["payload"].update({"outcome": "failed"}),
            "did not complete",
        ),
        (
            lambda events: events[2]["payload"].update({"name": "another-pod"}),
            "trigger does not match",
        ),
        (
            lambda events: events[6]["payload"].update({"name": "worker-other"}),
            "exactly one recovery",
        ),
    ],
)
def test_join_rejects_wrong_fault_or_target_identity(mutate, match):
    evidence = evidence_events()
    mutate(evidence)
    with pytest.raises(ValueError, match=match):
        join_recovery(matching_report(), evidence)


def test_join_rejects_unrelated_or_ambiguous_recovery():
    report = matching_report()
    report["recoveries"][0]["jobIds"] = ["job-b"]
    with pytest.raises(ValueError, match="exactly one recovery"):
        join_recovery(report, evidence_events())


def test_join_requires_complete_report_and_matching_fault_phase():
    report = matching_report()
    report["checkpoint"]["phase"] = "cleanup"
    with pytest.raises(ValueError, match="not complete"):
        join_recovery(report, evidence_events())

    report = matching_report()
    evidence = evidence_events()
    evidence[0]["payload"]["fault"]["phase"] = "startup"
    with pytest.raises(ValueError, match="exactly one recovery"):
        join_recovery(report, evidence)

    report = matching_report()
    report["recoveries"].append(dict(report["recoveries"][0], index=2))
    with pytest.raises(ValueError, match="exactly one recovery"):
        join_recovery(report, evidence_events())
