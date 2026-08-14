import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

MULTI_CELL_DIR = (
    Path(__file__).resolve().parents[3]
    / "development"
    / "video_poc"
    / "benchmarks"
    / "networking"
    / "multi_cell"
)
sys.path.insert(0, str(MULTI_CELL_DIR))

from analyze_campaign import EvidenceError, analyze, load_artifacts  # noqa: E402
from multi_cell_probe import (  # noqa: E402
    build_publish_command,
    decode_clock,
    expand_url,
    paint_clock,
    redact_text,
)
from render_campaign import (  # noqa: E402
    build_evidence_template,
    load_campaign,
    render,
    write,
)

TEMPLATE = MULTI_CELL_DIR / "multi-cell.staging.template.json"


def raw_campaign():
    raw = json.loads(TEMPLATE.read_text())
    raw["timing"]["warmupSeconds"] = 1
    raw["timing"]["measureSeconds"] = 2
    raw["timing"]["startupGraceSeconds"] = 1
    raw["fixture"]["fps"] = 2
    raw["maxRenderedJobs"] = 40
    now = time.time()
    raw["identitySnapshot"].update(
        {
            "status": "captured",
            "capturedAtUnixSeconds": now - 1,
            "expiresAtUnixSeconds": now + 3600,
        }
    )
    return raw


def load(tmp_path, raw=None):
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(raw or raw_campaign()))
    return load_campaign(path)


def render_test(campaign, execution_id="multicell-test-exec-001"):
    return render(campaign, execution_id)


def observed_location(manifest, location_name):
    location = manifest["locations"][location_name]
    cell = manifest["cells"][location["cell"]]
    return {
        "cell": location["cell"],
        "clusterContext": cell["clusterContext"],
        "clusterIdentity": cell["clusterIdentity"],
        "nodeName": location["nodeName"],
        "nodeUid": location["nodeUid"],
        "instanceType": location["instanceType"],
        "networkIdentity": location["networkIdentity"],
        "resolvedImageId": manifest["agentImage"],
    }


def metric(source, identity, delta, window, peak=None):
    value = {
        "source": source,
        "metric": "authoritative_metric_name",
        "identitySha256": identity,
        "querySha256": "f" * 64,
        "startUnixSeconds": window[0] - 1,
        "endUnixSeconds": window[1] + 1,
        "delta": delta,
        "counterReset": False,
        "samples": 4,
    }
    if peak is not None:
        value["peakBitsPerSecond"] = peak
    return value


def run_evidence(manifest, run):
    relay = manifest["cells"][run["relayCell"]]["relay"]
    location = manifest["locations"][run["readerLocation"]]
    relay_identity = relay["workloadIdentity"]["sha256"]
    lb_identity = relay["loadBalancer"]["configSha256"]
    network_identity = location["networkIdentity"]["sha256"]
    source = manifest["sourcePlacements"][run["source"]]
    window = (
        manifest["identitySnapshot"]["capturedAtUnixSeconds"] + 10,
        manifest["identitySnapshot"]["capturedAtUnixSeconds"] + 12,
    )
    metrics = {
        "relayBytesReceived": metric("prometheus", relay_identity, 1_000_000, window),
        "relayBytesSent": metric("prometheus", relay_identity, 1_000_000, window),
        "lbBytesIn": metric(
            "crusoe-metrics", lb_identity, 1_000_000, window, 4_000_000
        ),
        "lbBytesOut": metric(
            "crusoe-metrics", lb_identity, 1_000_000, window, 4_000_000
        ),
        "publisherVpcRxBytes": metric(
            "crusoe-metrics",
            manifest["locations"][run["publisherLocation"]]["networkIdentity"][
                "sha256"
            ],
            1_000_000,
            window,
            4_000_000,
        ),
        "publisherVpcTxBytes": metric(
            "crusoe-metrics",
            manifest["locations"][run["publisherLocation"]]["networkIdentity"][
                "sha256"
            ],
            1_000_000,
            window,
            4_000_000,
        ),
        "relayVpcRxBytes": metric(
            "crusoe-metrics",
            manifest["locations"][relay["location"]]["networkIdentity"]["sha256"],
            1_000_000,
            window,
            4_000_000,
        ),
        "relayVpcTxBytes": metric(
            "crusoe-metrics",
            manifest["locations"][relay["location"]]["networkIdentity"]["sha256"],
            1_000_000,
            window,
            4_000_000,
        ),
        "readerVpcRxBytes": metric(
            "crusoe-metrics", network_identity, 1_000_000, window, 4_000_000
        ),
        "readerVpcTxBytes": metric(
            "crusoe-metrics", network_identity, 1_000_000, window, 4_000_000
        ),
        "rtpPacketsReceived": metric("prometheus", relay_identity, 100, window),
        "rtpPacketsLost": metric("prometheus", relay_identity, 0, window),
        "rtpPacketsInError": metric("prometheus", relay_identity, 0, window),
        "publisherNodePacketDrops": metric(
            "prometheus",
            manifest["locations"][run["publisherLocation"]]["networkIdentity"][
                "sha256"
            ],
            0,
            window,
        ),
        "relayNodePacketDrops": metric(
            "prometheus",
            manifest["locations"][relay["location"]]["networkIdentity"]["sha256"],
            0,
            window,
        ),
        "readerNodePacketDrops": metric("prometheus", network_identity, 0, window),
        "publisherTcpRetransmits": metric(
            "prometheus",
            manifest["locations"][run["publisherLocation"]]["networkIdentity"][
                "sha256"
            ],
            0,
            window,
        ),
        "relayTcpRetransmits": metric(
            "prometheus",
            manifest["locations"][relay["location"]]["networkIdentity"]["sha256"],
            0,
            window,
        ),
        "readerTcpRetransmits": metric("prometheus", network_identity, 0, window),
    }
    for value in metrics.values():
        value["scopeSha256"] = run["streamIdentitySha256"]
    processor_transport = {
        "same-cell": "cluster-internal",
        "cross-cell-ingest": "cluster-internal",
        "cross-cell-direct": "public-direct",
        "cross-cell-replica": "local-replica",
        "cross-cell-ingest-and-execution": "public-direct",
    }[run["topology"]]
    return {
        "runId": run["runId"],
        "pairId": run["pairId"],
        "repetition": run["repetition"],
        "arm": run["arm"],
        "topology": run["topology"],
        "source": run["source"],
        "relayCell": run["relayCell"],
        "executionCell": run["executionCell"],
        "success": True,
        "observationStatus": "collected",
        "placement": {
            "sourceIdentitySha256": source["sourceIdentitySha256"],
            "homeCell": source["homeCell"],
            "relayShard": source["relayShard"],
            "placementGeneration": source["placementGeneration"],
            "recordRevision": source["recordRevision"],
            "executionCell": run["executionCell"],
            "remoteExecution": run["executionCell"] != run["relayCell"],
        },
        "relay": relay,
        "replicaIdentity": run["replicaIdentity"],
        "processorRuntimeIdentity": run["processorRuntimeIdentity"],
        "processor": {
            "workflow": manifest["processorWorkflow"],
            "runtimeIdentity": run["processorRuntimeIdentity"],
            "executionCell": run["executionCell"],
            "sourcePlacementGeneration": source["placementGeneration"],
            "nodeName": manifest["locations"][run["readerLocation"]]["nodeName"],
            "nodeUid": manifest["locations"][run["readerLocation"]]["nodeUid"],
            "sourceTransport": processor_transport,
            "claimCount": 1,
            "attempt": 1,
            "state": "running",
            "framesProcessed": 4,
            "firstResultUnixSeconds": window[0],
            "jobIdentitySha256": "a" * 64,
            "claimResponseSha256": "b" * 64,
        },
        "publisher": observed_location(manifest, run["publisherLocation"]),
        "reader": observed_location(manifest, run["readerLocation"]),
        "measurementWindow": {
            "startUnixSeconds": window[0],
            "endUnixSeconds": window[1],
            "warmupComplete": True,
        },
        "measurement": {
            "connectStartMonotonicNs": 1_000_000_000,
            "firstDecodedFrameMonotonicNs": 1_100_000_000,
            "decodedFrameArrivalMonotonicNs": [
                1_100_000_000,
                1_600_000_000,
                2_100_000_000,
                2_600_000_000,
            ],
            "measurementStartMonotonicNs": 1_100_000_000,
            "measurementEndMonotonicNs": 3_100_000_000,
            "pixelClockLatencyMs": [10, 11, 12, 13],
            "latencySource": "pixel-clock-v1",
            "clockSync": {
                "source": "chrony",
                "publisherUncertaintyMs": 1,
                "readerUncertaintyMs": 1,
                "checkedBeforeAndAfter": True,
            },
            "encodedPayloadBytes": 1_000_000,
            "encodedPayloadBytesSource": "pyav-packet-size",
        },
        "metrics": metrics,
    }


def valid_evidence(manifest):
    cases = []
    for case in manifest["placementCases"]:
        source = manifest["sourcePlacements"][case["source"]]
        observations = []
        for action in case["actions"]:
            rejected = action["expectedOutcome"] == "rejected"
            observations.append(
                {
                    "ordinal": action["ordinal"],
                    "action": action["action"],
                    "requestedExecutionCell": action["requestedExecutionCell"],
                    "outcome": action["expectedOutcome"],
                    "reasonCode": action["expectedReasonCode"],
                    "executionCell": action["expectedExecutionCell"],
                    "homeCell": None if rejected else source["homeCell"],
                    "relayShard": None if rejected else source["relayShard"],
                    "placementGeneration": (
                        None if rejected else source["placementGeneration"]
                    ),
                    "jobClaimed": (
                        False
                        if rejected or action["action"] == "activate-preview"
                        else True
                    ),
                }
            )
        cases.append(
            {
                "id": case["id"],
                "source": case["source"],
                "observationStatus": "collected",
                "observations": observations,
            }
        )
    return {
        "schemaVersion": 1,
        "environment": "staging",
        "campaign": manifest["campaign"],
        "campaignSha256": manifest["campaignSha256"],
        "executionId": manifest["executionId"],
        "runs": [run_evidence(manifest, run) for run in manifest["sequence"]],
        "placementCases": cases,
    }


def materialize_artifacts(tmp_path, evidence):
    root = tmp_path / "raw-artifacts"
    root.mkdir(exist_ok=True)
    index = []

    def add(artifact_id, kind, payload):
        relative = Path(artifact_id + ".json")
        data = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        (root / relative).write_bytes(data)
        index.append(
            {
                "id": artifact_id,
                "kind": kind,
                "relativePath": str(relative),
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )

    for run in evidence["runs"]:
        run_id = run["runId"]
        run["publisherReportArtifactId"] = run_id + ".publisher"
        add(
            run["publisherReportArtifactId"],
            "publisher-termination-report",
            {
                "role": "publish-clock",
                "success": True,
                "identity": {"runId": run_id},
            },
        )
        run["probeReportArtifactId"] = run_id + ".probe"
        add(
            run["probeReportArtifactId"],
            "probe-termination-report",
            {
                "role": "probe-clock",
                "success": True,
                "identity": {"runId": run_id},
                **run["measurement"],
            },
        )
        run["processorApiArtifactId"] = run_id + ".processor-api"
        add(
            run["processorApiArtifactId"],
            "processor-api-response",
            run["processor"],
        )
        run["placementApiArtifactId"] = run_id + ".placement-api"
        add(
            run["placementApiArtifactId"],
            "placement-api-response",
            run["placement"],
        )
        for name, metric_value in run["metrics"].items():
            artifact_id = "%s.metric.%s" % (run_id, name)
            metric_value["rawArtifactId"] = artifact_id
            add(
                artifact_id,
                "metric-range-response",
                {
                    key: value
                    for key, value in metric_value.items()
                    if key != "rawArtifactId"
                },
            )
    for case in evidence["placementCases"]:
        for observation in case["observations"]:
            artifact_id = "placement.%s.%d" % (case["id"], observation["ordinal"])
            observation["rawArtifactId"] = artifact_id
            add(
                artifact_id,
                "placement-api-response",
                {
                    key: value
                    for key, value in observation.items()
                    if key != "rawArtifactId"
                },
            )
    evidence["artifactIndex"] = index
    return root, load_artifacts(evidence, root)


def analyze_bound(tmp_path, manifest, evidence):
    _root, artifacts = materialize_artifacts(tmp_path, evidence)
    return analyze(manifest, evidence, artifacts)


def test_template_renders_counterbalanced_paired_jobs_by_exact_context(tmp_path):
    campaign = load(tmp_path)
    artifacts, manifest = render_test(campaign)

    assert len(manifest["sequence"]) == 18
    assert manifest["renderedJobCount"] == 36
    assert [run["arm"] for run in manifest["sequence"][:6]] == [
        "control",
        "variant",
        "variant",
        "control",
        "control",
        "variant",
    ]
    assert {value["context"] for value in artifacts.values()} == {
        "ck8s-stg-east",
        "ck8s-stg-south",
    }
    assert len(artifacts) == 36
    jobs = [value["document"]["items"][0] for value in artifacts.values()]
    assert all(job["spec"]["template"]["spec"].get("nodeName") for job in jobs)
    assert all(job["spec"]["backoffLimit"] == 0 for job in jobs)
    assert all(job["spec"]["activeDeadlineSeconds"] == 34 for job in jobs)
    publisher = next(
        job
        for job in jobs
        if job["metadata"]["labels"]["benchmark.roboflow.com/role"] == "publish-clock"
    )
    publisher_env = {
        item["name"]: item
        for item in publisher["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert publisher_env["BENCH_STARTUP_GRACE_SECONDS"]["value"] == "1.0"
    assert all(
        job["metadata"]["annotations"]["benchmark.roboflow.com/expected-node-uid"]
        for job in jobs
    )
    assert all(
        [step["action"] for step in item["steps"]]
        == [
            "apply-publisher",
            "readiness-barrier",
            "create-processor-job",
            "processor-claim-barrier",
            "apply-probe",
            "collect-and-clean-before-next-run",
        ]
        for item in manifest["execution"]
    )
    rendered = json.dumps({"artifacts": artifacts, "manifest": manifest})
    assert "publish-url-template" in rendered
    assert "rtsp://" not in rendered
    assert "token=" not in rendered
    assert all(
        "benchmark.roboflow.com/execution-id=multicell-test-exec-001" in item["command"]
        for item in manifest["cleanup"]
    )


def test_staging_dedicated_namespace_and_distinct_contexts_are_hard_guards(tmp_path):
    raw = raw_campaign()
    raw["environment"] = "production"
    with pytest.raises(ValueError, match="restricted to staging"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["cells"]["east"]["clusterContext"] = "ck8s-prod-east"
    with pytest.raises(ValueError, match="recognizably staging"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["namespace"] = "video-proc"
    with pytest.raises(ValueError, match="dedicated benchmark namespace"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["cells"]["south"]["clusterContext"] = raw["cells"]["east"]["clusterContext"]
    with pytest.raises(ValueError, match="distinct clusterContext"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["cells"]["east"]["exclusiveBenchmarkCell"] = False
    with pytest.raises(ValueError, match="exclusiveBenchmarkCell"):
        load(tmp_path, raw)


def test_committed_template_and_expired_identity_snapshot_are_not_runnable(tmp_path):
    template = json.loads(TEMPLATE.read_text())
    with pytest.raises(ValueError, match="templates are not runnable"):
        load(tmp_path, template)

    raw = raw_campaign()
    raw["identitySnapshot"]["expiresAtUnixSeconds"] = time.time() - 1
    with pytest.raises(ValueError, match="current and unexpired"):
        load(tmp_path, raw)


def test_immutable_images_network_and_location_identity_are_required(tmp_path):
    raw = raw_campaign()
    raw["agentImage"] = "registry/probe:latest"
    with pytest.raises(ValueError, match="immutable sha256"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["cells"]["east"]["relay"]["image"] = "mediamtx:latest"
    with pytest.raises(ValueError, match="relay.image must use an immutable"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["locations"]["east-processor"].pop("nodeUid")
    with pytest.raises(ValueError, match="nodeUid must be non-empty"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["locations"]["east-processor"]["networkIdentity"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        load(tmp_path, raw)


def test_remote_and_dedicated_policy_fail_closed_at_render_time(tmp_path):
    raw = raw_campaign()
    raw["workspacePolicies"]["shared-two-cell"]["allowRemoteExecution"] = False
    with pytest.raises(ValueError, match="workspace policy"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["workspacePolicies"]["dedicated-south"]["allowSharedFallback"] = True
    with pytest.raises(ValueError, match="fail closed"):
        load(tmp_path, raw)
    raw = raw_campaign()
    raw["pairs"][0]["arms"]["variant"]["relayCell"] = "south"
    with pytest.raises(ValueError, match="persisted source homeCell"):
        load(tmp_path, raw)

    raw = raw_campaign()
    raw["placementCases"][1]["actions"][1].update(
        {
            "expectedOutcome": "assigned",
            "expectedExecutionCell": "east",
            "expectedReasonCode": None,
        }
    )
    with pytest.raises(ValueError, match="outcome disagrees with workspace policy"):
        load(tmp_path, raw)

    raw = raw_campaign()
    raw["placementCases"][0]["actions"][1]["expectedOutcome"] = "assigned"
    with pytest.raises(ValueError, match="sticky reuse"):
        load(tmp_path, raw)


def test_paired_repetitions_and_job_cap_are_enforced(tmp_path):
    raw = raw_campaign()
    raw["repetitions"] = 2
    with pytest.raises(ValueError, match="at least three"):
        load(tmp_path, raw)
    campaign = load(tmp_path)
    campaign["maxRenderedJobs"] = 35
    with pytest.raises(ValueError, match="exceeding maxRenderedJobs"):
        render_test(campaign)


def test_long_run_ids_still_produce_distinct_role_job_names(tmp_path):
    raw = raw_campaign()
    raw["name"] = "campaign-" + "a" * 50
    raw["pairs"] = [copy.deepcopy(raw["pairs"][0])]
    raw["pairs"][0]["id"] = "pair-" + "b" * 50
    campaign = load(tmp_path, raw)
    artifacts, _manifest = render_test(campaign)
    names = [
        value["document"]["items"][0]["metadata"]["name"]
        for value in artifacts.values()
    ]

    assert len(names) == 12
    assert len(set(names)) == 12
    assert all(len(name) <= 63 for name in names)


def test_execution_id_namespaces_jobs_and_cleanup(tmp_path):
    campaign = load(tmp_path)
    artifacts_a, manifest_a = render(campaign, "execution-a-001")
    artifacts_b, manifest_b = render(campaign, "execution-b-001")

    assert set(artifacts_a).isdisjoint(artifacts_b)
    assert manifest_a["executionId"] != manifest_b["executionId"]
    assert all(
        "benchmark.roboflow.com/execution-id=execution-a-001" in item["command"]
        for item in manifest_a["cleanup"]
    )
    assert all(
        "benchmark.roboflow.com/execution-id=execution-b-001" in item["command"]
        for item in manifest_b["cleanup"]
    )


def test_renderer_refuses_to_overwrite_outputs(tmp_path):
    campaign = load(tmp_path)
    artifacts, manifest = render_test(campaign)
    output = tmp_path / "out"
    write(output, artifacts, manifest)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write(output, artifacts, manifest)


def test_evidence_template_is_complete_but_cannot_certify(tmp_path):
    campaign = load(tmp_path)
    _lists, manifest = render_test(campaign)
    template = build_evidence_template(manifest)

    assert len(template["runs"]) == len(manifest["sequence"])
    assert all(run["observationStatus"] == "not-collected" for run in template["runs"])
    with pytest.raises(EvidenceError, match="not marked observationStatus=collected"):
        analyze(manifest, template, {})


def test_complete_authoritative_evidence_certifies(tmp_path):
    campaign = load(tmp_path)
    _lists, manifest = render_test(campaign)
    report = analyze_bound(tmp_path, manifest, valid_evidence(manifest))

    assert report["certified"] is True
    assert len(report["runs"]) == 18
    assert all(run["passed"] for run in report["runs"])
    assert all(pair["repetitions"] == 3 for pair in report["pairs"])
    assert all(case["passed"] for case in report["placementCases"])


def test_raw_artifact_tamper_fails_before_certification(tmp_path):
    campaign = load(tmp_path)
    _artifacts, manifest = render_test(campaign)
    evidence = valid_evidence(manifest)
    root, _loaded = materialize_artifacts(tmp_path, evidence)
    first = evidence["artifactIndex"][0]
    (root / first["relativePath"]).write_text('{"tampered":true}\n')

    with pytest.raises(EvidenceError, match="sha256"):
        load_artifacts(evidence, root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda evidence: evidence["runs"].pop(), "run coverage"),
        (
            lambda evidence: evidence["runs"][0].update({"processor": None}),
            "processor evidence is missing",
        ),
        (
            lambda evidence: evidence["runs"][0]["reader"].update(
                {"nodeUid": "different"}
            ),
            "reader.nodeUid",
        ),
        (
            lambda evidence: evidence["runs"][0]["placement"].update(
                {"placementGeneration": 999}
            ),
            "placement.placementGeneration",
        ),
        (
            lambda evidence: evidence["runs"][0]["metrics"].pop("lbBytesIn"),
            "metrics.lbBytesIn is missing",
        ),
        (
            lambda evidence: evidence["runs"][0]["metrics"]["readerVpcRxBytes"].update(
                {"counterReset": True}
            ),
            "counterReset=false",
        ),
        (
            lambda evidence: evidence["runs"][0]["metrics"][
                "relayBytesReceived"
            ].update({"scopeSha256": "0" * 64}),
            "scopeSha256",
        ),
        (
            lambda evidence: evidence["runs"][0]["measurement"]["clockSync"].update(
                {"checkedBeforeAndAfter": False}
            ),
            "checked before and after",
        ),
    ],
)
def test_evidence_omissions_and_identity_drift_fail_closed(tmp_path, mutation, message):
    campaign = load(tmp_path)
    _lists, manifest = render_test(campaign)
    evidence = valid_evidence(manifest)
    mutation(evidence)
    with pytest.raises(EvidenceError, match=message):
        analyze_bound(tmp_path, manifest, evidence)


def test_threshold_failure_is_reported_without_certification(tmp_path):
    campaign = load(tmp_path)
    _lists, manifest = render_test(campaign)
    evidence = valid_evidence(manifest)
    evidence["runs"][0]["metrics"]["rtpPacketsLost"]["delta"] = 1
    report = analyze_bound(tmp_path, manifest, evidence)

    assert report["certified"] is False
    assert report["runs"][0]["gates"]["packetLoss"] is False


def test_stall_at_measurement_window_boundary_is_not_hidden(tmp_path):
    campaign = load(tmp_path)
    _artifacts, manifest = render_test(campaign)
    evidence = valid_evidence(manifest)
    run = evidence["runs"][0]
    run["measurement"]["decodedFrameArrivalMonotonicNs"] = [
        1_100_000_000,
        1_600_000_000,
    ]
    run["measurement"]["pixelClockLatencyMs"] = [10, 11]
    report = analyze_bound(tmp_path, manifest, evidence)

    assert report["runs"][0]["stallCount"] == 1
    assert report["runs"][0]["gates"]["stalls"] is False
    assert report["certified"] is False


def test_paired_regression_gate_uses_each_repetition(tmp_path):
    campaign = load(tmp_path)
    _lists, manifest = render_test(campaign)
    evidence = valid_evidence(manifest)
    for run in evidence["runs"]:
        if run["pairId"] == "east-vs-south-direct" and run["arm"] == "variant":
            run["measurement"]["pixelClockLatencyMs"] = [30, 31, 32, 33]
    report = analyze_bound(tmp_path, manifest, evidence)

    direct = next(
        pair for pair in report["pairs"] if pair["pairId"] == "east-vs-south-direct"
    )
    assert direct["pairedLatencyGate"] is False
    assert direct["repetitions"] == 3
    assert report["certified"] is False


def test_dedicated_rejection_must_prove_no_claim(tmp_path):
    campaign = load(tmp_path)
    _lists, manifest = render_test(campaign)
    evidence = valid_evidence(manifest)
    dedicated = next(
        case
        for case in evidence["placementCases"]
        if case["id"] == "dedicated-fail-closed"
    )
    dedicated["observations"][1]["jobClaimed"] = True

    with pytest.raises(EvidenceError, match="no execution and no claim"):
        analyze_bound(tmp_path, manifest, evidence)


def test_probe_pixel_clock_round_trip_and_cbr_command():
    numpy = pytest.importorskip("numpy")
    frame = numpy.zeros((120, 640, 3), dtype=numpy.uint8)
    value = 0xA5A55A5A

    assert decode_clock(paint_clock(frame, value)) == value
    command = build_publish_command(
        "/usr/bin/ffmpeg", 1920, 1080, 15, 5_000_000, "rtsp://host/path?token=secret"
    )
    assert "nal-hrd=cbr:force-cfr=1" in command
    assert command[command.index("-minrate") + 1] == "5000000"
    assert "token=secret" not in " ".join(redact_text(item) for item in command)


def test_probe_requires_a_stream_placeholder():
    assert expand_url("rtsp://relay/{stream}?key=secret", "safe-stream") == (
        "rtsp://relay/safe-stream?key=secret"
    )
    with pytest.raises(ValueError, match="must contain"):
        expand_url("rtsp://relay/static", "safe-stream")
