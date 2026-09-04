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
        "plannedConcurrency": 1,
        "starts": [{"job": {"id": "job-a"}}],
        "jobs": [{"id": "job-a", "state": "cancelled"}],
        "errors": [],
        "cancelErrors": [],
        "recoveries": [
            {
                "index": 1,
                "sourcePhase": "measurement",
                "jobIds": ["job-a"],
                "outcome": "recovered",
                "before": [{"id": "job-a", "processorId": "worker-old"}],
                "after": [{"id": "job-a", "processorId": "worker-new"}],
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
    with pytest.raises(ValueError, match="did not complete successfully"):
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
        "plannedConcurrency": 1,
        "starts": [{"job": {"id": "job-a"}}],
        "jobs": [{"id": "job-a", "state": "cancelled"}],
        "errors": [],
        "cancelErrors": [],
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
            "relay fault",
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


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda report: report.update({"success": False}), "did not complete"),
        (
            lambda report: report["errors"].append({"error": "worker failed"}),
            "did not complete",
        ),
        (
            lambda report: report["cancelErrors"].append(
                {"jobId": "job-a", "error": "cancel failed"}
            ),
            "did not complete",
        ),
        (
            lambda report: report["jobs"][0].update({"state": "error"}),
            "did not clean up",
        ),
    ],
)
def test_join_processor_recovery_fails_closed_on_run_or_cleanup_failure(mutate, match):
    report = matching_report()
    mutate(report)

    with pytest.raises(ValueError, match=match):
        join_recovery(report, evidence_events())


def relay_evidence_events():
    events = evidence_events("relay-run-001")
    events[0]["payload"]["fault"] = {
        "type": "relay-pod-loss",
        "phase": "steady-state",
    }
    events[1]["payload"] = {
        "selector": {"app.kubernetes.io/name": "mediamtx"},
        "benchmarkJob": None,
    }
    events[2]["payload"] = {
        "name": "relay-old",
        "uid": "relay-uid-old",
        "owner": {"uid": "relay-controller"},
    }
    events[3]["payload"] = {
        "name": "relay-old",
        "uid": "relay-uid-old",
        "owner": {"uid": "relay-controller"},
    }
    for index in (4, 5):
        events[index]["payload"] = {
            "podName": "relay-old",
            "podUid": "relay-uid-old",
        }
    events[6]["payload"] = {
        "name": "relay-new",
        "uid": "relay-uid-new",
        "owner": {"uid": "relay-controller"},
    }
    return events


def relay_report():
    def job(frames, published):
        return {
            "id": "job-a",
            "state": "running",
            "processorId": "worker-a",
            "attempts": 0,
            "stats": {
                "frames": frames,
                "counters": {"published": published},
            },
        }

    return {
        "runId": "relay-run-001",
        "success": True,
        "startedAt": "2026-08-12T20:00:00Z",
        "endedAt": "2026-08-12T20:01:21Z",
        "measurementStartedAt": "2026-08-12T20:00:00Z",
        "measurementEndedAt": "2026-08-12T20:01:20Z",
        "checkpoint": {"phase": "complete"},
        "plannedConcurrency": 1,
        "profiles": [{"ordinal": 1, "imageOutput": "visualization"}],
        "starts": [{"ordinal": 1, "job": {"id": "job-a"}}],
        "samples": [
            {"phase": "measurement", "elapsedSeconds": 2, "jobs": [job(20, 20)]},
            {"phase": "measurement", "elapsedSeconds": 6, "jobs": [job(20, 20)]},
            {"phase": "measurement", "elapsedSeconds": 11, "jobs": [job(20, 20)]},
            {"phase": "measurement", "elapsedSeconds": 14, "jobs": [job(30, 30)]},
        ],
        "recoveries": [],
        "errors": [],
        "cancelErrors": [],
        "jobs": [{"id": "job-a", "state": "cancelled"}],
    }


def relay_resources(report):
    def series(name, values, labels=None):
        return {
            "metric": labels or {},
            "values": [[timestamp, str(value)] for timestamp, value in values],
        }

    return {
        "schemaVersion": 1,
        "environment": "staging",
        "clusterContext": "ck8s-stg",
        "sourceReportSha256": canonical_digest(report),
        "runId": report["runId"],
        "measurementStartedAt": report["measurementStartedAt"],
        "measurementEndedAt": report["measurementEndedAt"],
        "metrics": {
            "relayPodIdentity": {
                "series": [
                    series(
                        "relayPodIdentity",
                        [(1786564800, 1), (1786564804, 1)],
                        {"pod": "relay-old", "uid": "relay-uid-old"},
                    ),
                    series(
                        "relayPodIdentity",
                        [(1786564810, 1), (1786564870, 1), (1786564875, 1)],
                        {"pod": "relay-new", "uid": "relay-uid-new"},
                    ),
                ]
            },
            "relayReaders": {
                "series": [
                    series(
                        "relayReaders",
                        [(1786564870, 1), (1786564875, 1)],
                        {"pod": "relay-new"},
                    )
                ]
            },
            "relayIngressBytesPerSecond": {
                "series": [
                    series(
                        "relayIngressBytesPerSecond",
                        [(1786564870, 100), (1786564875, 100)],
                        {"pod": "relay-new"},
                    )
                ]
            },
            "relayEgressBytesPerSecond": {
                "series": [
                    series(
                        "relayEgressBytesPerSecond",
                        [(1786564870, 100), (1786564875, 100)],
                        {"pod": "relay-new"},
                    )
                ]
            },
            "relayOutputPathCount": {
                "series": [
                    series(
                        "relayOutputPathCount",
                        [(1786564870, 1), (1786564875, 1)],
                        {"pod": "relay-new"},
                    )
                ]
            },
            "relayOutputIngressBytesPerSecond": {
                "series": [
                    series(
                        "relayOutputIngressBytesPerSecond",
                        [(1786564870, 50), (1786564875, 50)],
                        {"pod": "relay-new"},
                    )
                ]
            },
        },
    }


def test_join_relay_recovery_requires_post_replacement_frame_and_output_progress():
    report = relay_report()
    result = join_recovery(
        report, relay_evidence_events(), resources=relay_resources(report)
    )

    assert result["faultType"] == "relay-pod-loss"
    recovery = result["recoveries"][0]
    assert recovery["faultToFrameProgressUpperBoundSeconds"] == 10.0
    assert recovery["sampleGapUpperBoundSeconds"] == 12.0
    assert recovery["firstPostReplacementSampleAt"].endswith("20:00:11+00:00")
    assert recovery["processorOwnershipStable"] is True
    assert recovery["outputProgressRequired"] is True
    assert (
        result["relayMediaEvidence"]["minimumPostReplacementEgressBytesPerSecond"]
        == 100
    )
    assert (
        result["relayMediaEvidence"][
            "minimumPostReplacementOutputIngressBytesPerSecond"
        ]
        == 50
    )
    assert "not gapless media" in result["measurementSemantics"]


def test_join_relay_prefers_explicit_sample_timestamp_over_elapsed_clock():
    report = relay_report()
    report["startedAt"] = "2026-08-12T19:00:00Z"
    sampled_times = [
        "2026-08-12T20:00:02Z",
        "2026-08-12T20:00:06Z",
        "2026-08-12T20:00:11Z",
        "2026-08-12T20:00:14Z",
    ]
    for sample, sampled_at in zip(report["samples"], sampled_times):
        sample["sampledAt"] = sampled_at

    result = join_recovery(
        report, relay_evidence_events(), resources=relay_resources(report)
    )

    assert result["recoveries"][0]["faultToFrameProgressUpperBoundSeconds"] == 10.0


def test_join_relay_recovery_rejects_requeue_or_stalled_output():
    report = relay_report()
    report["samples"][-1]["jobs"][0]["processorId"] = "worker-b"
    with pytest.raises(ValueError, match="changed processor ownership"):
        join_recovery(
            report, relay_evidence_events(), resources=relay_resources(report)
        )

    report = relay_report()
    report["samples"][-1]["jobs"][0]["stats"]["counters"]["published"] = 20
    with pytest.raises(ValueError, match="did not advance"):
        join_recovery(
            report, relay_evidence_events(), resources=relay_resources(report)
        )


def test_join_relay_recovery_rejects_unrelated_controller_revision():
    evidence = relay_evidence_events()
    evidence[6]["payload"]["owner"]["uid"] = "another-controller"

    with pytest.raises(ValueError, match="captured controller"):
        report = relay_report()
        join_recovery(report, evidence, resources=relay_resources(report))


def test_join_relay_recovery_requires_downstream_relay_media_after_replacement():
    report = relay_report()
    resources = relay_resources(report)
    resources["metrics"]["relayOutputIngressBytesPerSecond"]["series"][0]["values"][-1][
        1
    ] = "0"

    with pytest.raises(ValueError, match="output ingress did not recover"):
        join_recovery(report, relay_evidence_events(), resources=resources)

    resources = relay_resources(report)
    resources["metrics"]["relayPodIdentity"]["series"][1]["metric"][
        "uid"
    ] = "unrelated-relay-uid"
    with pytest.raises(ValueError, match="do not match the injected replacement"):
        join_recovery(report, relay_evidence_events(), resources=resources)
