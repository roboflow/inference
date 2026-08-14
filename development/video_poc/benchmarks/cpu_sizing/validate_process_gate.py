#!/usr/bin/env python3
"""Generate or validate a CPU process-containment evidence bundle.

The bundle never accepts operator-authored booleans. Every check is derived
from five retained JSON artifacts and their hashes. Validation re-reads those
artifacts and recomputes the complete bundle.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

from verify_rollout_patch import validate_patch_document

DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")
STAGING_IMAGE = re.compile(
    r"us-central1-docker[.]pkg[.]dev/roboflow-staging/video-proc/"
    r"video-processor-cpu-telemetry@sha256:[0-9a-f]{64}\Z"
)
SENSITIVE_KEY = re.compile(
    r"(?:api.?key|authorization|access.?token|credential|password|secret)", re.I
)
SENSITIVE_VALUE = re.compile(
    r"(?:bearer\s+[A-Za-z0-9._~+/=-]+|(?:api[_-]?key|token|secret)=[^&\s]+)", re.I
)
ARTIFACT_NAMES = {
    "topologyReport",
    "processSnapshot",
    "cancellationObservation",
    "crashObservation",
    "cleanupStatus",
}


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _artifact_descriptor(path, base):
    resolved = Path(path).resolve()
    relative = resolved.relative_to(Path(base).resolve())
    if relative == Path(".") or ".." in relative.parts:
        raise ValueError("process evidence paths must be inside the bundle directory")
    return {"path": str(relative), "sha256": _sha256(resolved.read_bytes())}


def _expected_runtime(patch, catalog):
    validate_patch_document(catalog, patch)
    container = patch["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item["value"] for item in container["env"]}
    variant = environment["VIDEO_PROC_RUNTIME_VARIANT"]
    if not variant.endswith("-process"):
        raise ValueError("containment gate requires a process-topology patch")
    image = container["image"]
    revision = environment["VIDEO_PROC_GIT_SHA"]
    if STAGING_IMAGE.fullmatch(image) is None or GIT_SHA.fullmatch(revision) is None:
        raise ValueError("expected process runtime identity is not immutable")
    return {"image": image, "revision": revision, "variant": variant}


def _scan_credentials(value, location="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            if SENSITIVE_KEY.search(str(key)):
                raise ValueError(f"credential-shaped key retained at {location}.{key}")
            _scan_credentials(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_credentials(child, f"{location}[{index}]")
    elif isinstance(value, str) and SENSITIVE_VALUE.search(value):
        raise ValueError(f"credential-shaped value retained at {location}")


def _runtime_jobs(report, expected):
    if report.get("plannedConcurrency") != 2:
        raise ValueError("process containment topology report must be an exact c2 run")
    jobs = report.get("jobs") or []
    if len(jobs) != 2:
        raise ValueError("process containment topology report must retain two jobs")
    runtimes = []
    job_ids = []
    for job in jobs:
        job_id = job.get("id")
        runtime = (job.get("stats") or {}).get("runtime") or {}
        if not job_id or not runtime:
            raise ValueError("topology report jobs need IDs and runtime identity")
        for key in ("image", "revision", "variant"):
            if runtime.get(key) != expected[key]:
                raise ValueError(f"topology report runtime {key} differs from patch")
        if not runtime.get("podUid"):
            raise ValueError("topology report must retain pod UID")
        process_id = runtime.get("processId")
        if not isinstance(process_id, int) or process_id <= 0:
            raise ValueError("topology report must retain positive process IDs")
        job_ids.append(job_id)
        runtimes.append(runtime)
    if len(set(job_ids)) != 2:
        raise ValueError("topology report job IDs must be distinct")
    if len({item["processId"] for item in runtimes}) != 2:
        raise ValueError("process topology must report one distinct child PID per job")
    if len({item["podUid"] for item in runtimes}) != 1:
        raise ValueError("process containment jobs must use one pod")
    return job_ids, runtimes


def _identity(document, expected, pod_uid, location):
    identity = document.get("runtime") or {}
    required = {
        "image": expected["image"],
        "revision": expected["revision"],
        "variant": expected["variant"],
        "podUid": pod_uid,
    }
    if any(identity.get(key) != value for key, value in required.items()):
        raise ValueError(f"{location} runtime identity differs from topology report")


def _jobs_by_id(snapshot, location):
    jobs = snapshot.get("jobs") or []
    result = {}
    for job in jobs:
        job_id = job.get("id")
        frames = job.get("frames")
        if not job_id or not isinstance(frames, int) or frames < 0:
            raise ValueError(f"{location} jobs need IDs and nonnegative frame counters")
        if job_id in result:
            raise ValueError(f"{location} repeats a job ID")
        result[job_id] = job
    return result


def _process_ids(processes, location):
    if not isinstance(processes, list):
        raise ValueError(f"{location} must be a process table")
    result = set()
    for item in processes:
        if set(item) != {"pid", "ppid", "argv"}:
            raise ValueError(f"{location} process entries must contain pid, ppid, argv")
        if not isinstance(item["pid"], int) or not isinstance(item["ppid"], int):
            raise ValueError(f"{location} process IDs must be integers")
        if not isinstance(item["argv"], list) or not all(
            isinstance(token, str) for token in item["argv"]
        ):
            raise ValueError(f"{location} argv must be a sanitized string list")
        if any(
            SENSITIVE_KEY.search(token) or SENSITIVE_VALUE.search(token)
            for token in item["argv"]
        ):
            raise ValueError(
                f"credential-shaped process argument retained in {location}"
            )
        if item["pid"] in result:
            raise ValueError(f"{location} repeats a process ID")
        result.add(item["pid"])
    return result


def _process_map(processes, location):
    _process_ids(processes, location)
    return {item["pid"]: item for item in processes}


def _derive_lifecycle(
    document,
    expected,
    pod_uid,
    target_state,
    location,
    process_by_job,
    supervisor,
):
    _identity(document, expected, pod_uid, location)
    target = document.get("targetJobId")
    sibling = document.get("siblingJobId")
    if not target or not sibling or target == sibling:
        raise ValueError(f"{location} needs distinct target and sibling job IDs")
    target_pid = process_by_job.get(target)
    sibling_pid = process_by_job.get(sibling)
    action = document.get("action") or {}
    expected_action = "cancel" if target_state == "cancelled" else "signal"
    if (
        action.get("type") != expected_action
        or action.get("targetProcessId") != target_pid
    ):
        raise ValueError(f"{location} action does not bind the target child PID")
    if target_state == "error":
        if (
            action.get("signal") not in {"SIGKILL", 9}
            or action.get("deliveryExitCode") != 0
        ):
            raise ValueError(
                "crash observation must successfully inject SIGKILL into the target child"
            )
    elif action.get("responseStatus") not in {200, 202, 204}:
        raise ValueError("cancellation action must retain a successful response status")
    observed_exit = document.get("observedExit") or {}
    if (
        observed_exit.get("processId") != target_pid
        or observed_exit.get("observed") is not True
    ):
        raise ValueError(f"{location} does not retain the target child exit")
    if target_state == "error" and observed_exit.get("signal") not in {"SIGKILL", 9}:
        raise ValueError("crash observation exit does not match injected SIGKILL")
    before_processes = _process_map(
        document.get("beforeProcesses"), f"{location}.beforeProcesses"
    )
    after_processes = _process_map(
        document.get("afterProcesses"), f"{location}.afterProcesses"
    )
    if not {supervisor, target_pid, sibling_pid} <= set(before_processes):
        raise ValueError(f"{location} pre-action process table is incomplete")
    if (
        before_processes[target_pid]["ppid"] != supervisor
        or before_processes[sibling_pid]["ppid"] != supervisor
    ):
        raise ValueError(f"{location} pre-action child is not parented by supervisor")
    if target_pid in after_processes or not {supervisor, sibling_pid} <= set(
        after_processes
    ):
        raise ValueError(
            f"{location} post-action process table does not prove containment"
        )
    if after_processes[sibling_pid]["ppid"] != supervisor:
        raise ValueError(
            f"{location} sibling is not parented by supervisor after action"
        )
    before = _jobs_by_id(document.get("before") or {}, f"{location}.before")
    after = _jobs_by_id(document.get("after") or {}, f"{location}.after")
    if {target, sibling} - set(before) or {target, sibling} - set(after):
        raise ValueError(f"{location} does not retain both jobs before and after")
    if before[sibling].get("state") != "running":
        raise ValueError(f"{location} sibling was not running before the fault")
    if after[sibling].get("state") != "running":
        raise ValueError(f"{location} sibling did not survive")
    if after[sibling]["frames"] <= before[sibling]["frames"]:
        raise ValueError(f"{location} sibling frames did not advance")
    if after[target].get("state") != target_state:
        raise ValueError(f"{location} target did not reach {target_state}")
    if target_state == "error":
        failure = document.get("failure")
        if not isinstance(failure, dict) or not failure.get("message"):
            raise ValueError("crash observation must retain the sanitized failure")
    return {
        "targetJobId": target,
        "siblingJobId": sibling,
        "siblingFramesBefore": before[sibling]["frames"],
        "siblingFramesAfter": after[sibling]["frames"],
        "targetFinalState": after[target]["state"],
        "targetProcessId": target_pid,
        "siblingProcessId": sibling_pid,
        "supervisorProcessId": supervisor,
        "action": action,
        "observedExit": observed_exit,
    }


def derive(artifacts, patch, catalog):
    if set(artifacts) != ARTIFACT_NAMES:
        raise ValueError(f"process gate needs exactly {sorted(ARTIFACT_NAMES)}")
    for name, document in artifacts.items():
        _scan_credentials(document, name)
    expected = _expected_runtime(patch, catalog)
    job_ids, runtimes = _runtime_jobs(artifacts["topologyReport"], expected)
    pod_uid = runtimes[0]["podUid"]
    child_pids = sorted(item["processId"] for item in runtimes)
    process_by_job = {
        job_id: runtime["processId"] for job_id, runtime in zip(job_ids, runtimes)
    }

    snapshot = artifacts["processSnapshot"]
    _identity(snapshot, expected, pod_uid, "processSnapshot")
    supervisor = snapshot.get("supervisorProcessId")
    processes = snapshot.get("processes") or []
    if not isinstance(supervisor, int) or supervisor <= 0:
        raise ValueError("process snapshot needs a positive supervisor PID")
    process_pids = {item.get("pid") for item in processes}
    for item in processes:
        argv = item.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(token, str) for token in argv
        ):
            raise ValueError("process snapshot argv must be a sanitized string list")
        if any(
            SENSITIVE_KEY.search(token) or SENSITIVE_VALUE.search(token)
            for token in argv
        ):
            raise ValueError("credential-shaped process argument retained")
    if supervisor not in process_pids:
        raise ValueError("process snapshot does not contain the supervisor")
    observed_children = sorted(
        item.get("pid")
        for item in processes
        if item.get("ppid") == supervisor and item.get("pid") != supervisor
    )
    if observed_children != child_pids:
        raise ValueError("process snapshot children differ from job runtime PIDs")

    cancellation = _derive_lifecycle(
        artifacts["cancellationObservation"],
        expected,
        pod_uid,
        "cancelled",
        "cancellationObservation",
        process_by_job,
        supervisor,
    )
    crash = _derive_lifecycle(
        artifacts["crashObservation"],
        expected,
        pod_uid,
        "error",
        "crashObservation",
        process_by_job,
        supervisor,
    )
    if (
        cancellation["targetJobId"] not in job_ids
        or crash["targetJobId"] not in job_ids
    ):
        raise ValueError("lifecycle target jobs differ from topology report")
    cleanup = artifacts["cleanupStatus"]
    _identity(cleanup, expected, pod_uid, "cleanupStatus")
    if cleanup.get("activeJobs") != 0:
        raise ValueError("cleanup did not return activeJobs to zero")
    cleanup_processes = _process_map(
        cleanup.get("processes"), "cleanupStatus.processes"
    )
    leaked_children = {
        pid
        for pid, process in cleanup_processes.items()
        if process["ppid"] == supervisor and pid != supervisor
    }
    if (
        supervisor not in cleanup_processes
        or set(cleanup_processes) & set(child_pids)
        or leaked_children
    ):
        raise ValueError("cleanup did not preserve supervisor and remove every child")
    return {
        "environment": "staging",
        "runtime": {**expected, "podUid": pod_uid},
        "topology": {
            "plannedConcurrency": 2,
            "jobIds": sorted(job_ids),
            "supervisorProcessId": supervisor,
            "childProcessIds": child_pids,
        },
        "cancellation": cancellation,
        "crash": crash,
        "cleanupActiveJobs": 0,
    }


def generate(paths, patch_path, catalog_path, output_path):
    output_path = Path(output_path).resolve()
    base = output_path.parent
    artifacts = {name: _load(path) for name, path in paths.items()}
    derived = derive(artifacts, _load(patch_path), _load(catalog_path))
    descriptors = {
        name: _artifact_descriptor(path, base) for name, path in paths.items()
    }
    bundle = {
        "schemaVersion": 2,
        "environment": "staging",
        "expectedPatch": _artifact_descriptor(patch_path, base),
        "catalog": _artifact_descriptor(catalog_path, base),
        "artifacts": descriptors,
        "derived": derived,
    }
    output_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return bundle


def _read_descriptor(base, descriptor):
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("evidence descriptor must contain path and sha256")
    path = (base / descriptor["path"]).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError("evidence descriptor escapes the bundle directory") from error
    content = path.read_bytes()
    if _sha256(content) != descriptor["sha256"]:
        raise ValueError(f"evidence hash mismatch: {descriptor['path']}")
    return json.loads(content)


def validate(bundle_path):
    bundle_path = Path(bundle_path).resolve()
    bundle = _load(bundle_path)
    if bundle.get("schemaVersion") != 2 or bundle.get("environment") != "staging":
        raise ValueError("process gate bundle must be staging schemaVersion 2")
    if set(bundle) != {
        "schemaVersion",
        "environment",
        "expectedPatch",
        "catalog",
        "artifacts",
        "derived",
    }:
        raise ValueError("process gate bundle has unexpected fields")
    if set(bundle["artifacts"]) != ARTIFACT_NAMES:
        raise ValueError("process gate bundle artifact set is incomplete")
    base = bundle_path.parent
    patch = _read_descriptor(base, bundle["expectedPatch"])
    catalog = _read_descriptor(base, bundle["catalog"])
    artifacts = {
        name: _read_descriptor(base, descriptor)
        for name, descriptor in bundle["artifacts"].items()
    }
    recomputed = derive(artifacts, patch, catalog)
    if bundle["derived"] != recomputed:
        raise ValueError("process gate derived evidence does not match raw artifacts")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--topology-report", type=Path, required=True)
    generate_parser.add_argument("--process-snapshot", type=Path, required=True)
    generate_parser.add_argument("--cancellation-observation", type=Path, required=True)
    generate_parser.add_argument("--crash-observation", type=Path, required=True)
    generate_parser.add_argument("--cleanup-status", type=Path, required=True)
    generate_parser.add_argument("--expected-patch", type=Path, required=True)
    generate_parser.add_argument("--catalog", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("bundle", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "generate":
        paths = {
            "topologyReport": args.topology_report,
            "processSnapshot": args.process_snapshot,
            "cancellationObservation": args.cancellation_observation,
            "crashObservation": args.crash_observation,
            "cleanupStatus": args.cleanup_status,
        }
        generate(paths, args.expected_patch, args.catalog, args.output)
        print(args.output.resolve())
    else:
        validate(args.bundle)
        print(json.dumps({"valid": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
