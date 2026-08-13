#!/usr/bin/env python3
"""Run a resumable staging-only matrix of video workflow experiments.

Each scenario is executed in a fresh child process so interruption and cleanup
remain scoped to one experiment. The API key stays in the environment and is
never copied into commands, suite manifests, or result reports.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_api_multi_workspace_corpus import (
    is_multi_workspace_scenario,
    normalize_scenario as normalize_multi_workspace_scenario,
)
from run_api_workflow_corpus import (
    RunLock,
    SAFE_RUN_ID,
    SignalStop,
    parse_workload,
    validate_api_base,
)

DEFAULT_RUNNER = Path(__file__).with_name("run_api_workflow_corpus.py")
DEFAULT_MULTI_WORKSPACE_RUNNER = Path(__file__).with_name(
    "run_api_multi_workspace_corpus.py"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _positive(value, field):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return parsed


def _nonnegative(value, field):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return parsed


def load_matrix(path):
    path = Path(path).resolve()
    matrix_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as source:
        document = json.load(source)
    if document.get("schemaVersion") != 1:
        raise ValueError("matrix schemaVersion must be 1")
    if document.get("environment") != "staging":
        raise ValueError("matrix environment must be staging")

    defaults = dict(document.get("defaults") or {})
    defaults["apiBase"] = validate_api_base(
        defaults.get("apiBase")
        or "https://roboflow-api-staging.web.app"
    )
    max_jobs = int(defaults.get("maxPlannedJobs", 32))
    if max_jobs < 1:
        raise ValueError("defaults.maxPlannedJobs must be positive")

    scenarios = []
    names = set()
    for raw in document.get("scenarios") or []:
        name = str(raw.get("name") or "")
        if not SAFE_RUN_ID.fullmatch(name) or name in names:
            raise ValueError(f"invalid or duplicate scenario name: {name!r}")
        names.add(name)
        if is_multi_workspace_scenario(raw):
            normalized = normalize_multi_workspace_scenario(document, raw)
            scenarios.append(
                {
                    **normalized,
                    "multiWorkspace": True,
                    "repetitions": int(raw.get("repetitions", 1)),
                    "cooldownSeconds": max(
                        0.0,
                        float(
                            raw.get(
                                "cooldownSeconds",
                                defaults.get("cooldownSeconds", 15),
                            )
                        ),
                    ),
                    "notes": str(raw.get("notes") or ""),
                }
            )
            if scenarios[-1]["repetitions"] < 1:
                raise ValueError(f"scenario {name} repetitions must be positive")
            continue
        if not defaults.get("workspace"):
            raise ValueError("matrix defaults.workspace is required")
        source_selectors = [
            key for key in ("sourceId", "sourceName") if defaults.get(key)
        ]
        if len(source_selectors) != 1:
            raise ValueError(
                "set exactly one of defaults.sourceId or defaults.sourceName"
            )
        workloads = [parse_workload(item) for item in raw.get("workloads") or []]
        if not workloads:
            raise ValueError(f"scenario {name} must define workloads")
        planned_jobs = sum(item["count"] for item in workloads)
        if planned_jobs > max_jobs:
            raise ValueError(
                f"scenario {name} plans {planned_jobs} jobs, "
                f"above safety cap {max_jobs}"
            )
        repetitions = int(raw.get("repetitions", 1))
        if repetitions < 1:
            raise ValueError(f"scenario {name} repetitions must be positive")
        max_fps = raw.get("maxFps")
        if max_fps is not None:
            max_fps = _positive(max_fps, f"scenario {name} maxFps")
        scenarios.append(
            {
                "name": name,
                "workloads": raw["workloads"],
                "plannedJobs": planned_jobs,
                "durationSeconds": _positive(
                    raw.get("durationSeconds", defaults.get("durationSeconds", 300)),
                    f"scenario {name} durationSeconds",
                ),
                "startupTimeoutSeconds": _positive(
                    raw.get(
                        "startupTimeoutSeconds",
                        defaults.get("startupTimeoutSeconds", 300),
                    ),
                    f"scenario {name} startupTimeoutSeconds",
                ),
                "cleanupTimeoutSeconds": _positive(
                    raw.get(
                        "cleanupTimeoutSeconds",
                        defaults.get("cleanupTimeoutSeconds", 60),
                    ),
                    f"scenario {name} cleanupTimeoutSeconds",
                ),
                "recoveryTimeoutSeconds": _nonnegative(
                    raw.get(
                        "recoveryTimeoutSeconds",
                        defaults.get("recoveryTimeoutSeconds", 0),
                    ),
                    f"scenario {name} recoveryTimeoutSeconds",
                ),
                "startupFaultReadySeconds": _nonnegative(
                    raw.get(
                        "startupFaultReadySeconds",
                        defaults.get("startupFaultReadySeconds", 0),
                    ),
                    f"scenario {name} startupFaultReadySeconds",
                ),
                "pollIntervalSeconds": _positive(
                    raw.get(
                        "pollIntervalSeconds",
                        defaults.get("pollIntervalSeconds", 2),
                    ),
                    f"scenario {name} pollIntervalSeconds",
                ),
                "cooldownSeconds": max(
                    0.0,
                    float(
                        raw.get(
                            "cooldownSeconds", defaults.get("cooldownSeconds", 15)
                        )
                    ),
                ),
                "maxFps": max_fps,
                "publishOutput": bool(raw.get("publishOutput", False)),
                "requireSingleProcessor": bool(
                    raw.get("requireSingleProcessor", True)
                ),
                "repetitions": repetitions,
                "notes": str(raw.get("notes") or ""),
                "multiWorkspace": False,
                "requiredApiKeyEnvs": [
                    defaults.get("apiKeyEnv", "VIDEO_BENCHMARK_API_KEY")
                ],
            }
        )
        if scenarios[-1]["recoveryTimeoutSeconds"] > 3600:
            raise ValueError(
                f"scenario {name} recoveryTimeoutSeconds cannot exceed 3600"
            )
        if scenarios[-1]["startupFaultReadySeconds"] > 300:
            raise ValueError(
                f"scenario {name} startupFaultReadySeconds cannot exceed 300"
            )
        if 0 < scenarios[-1]["startupFaultReadySeconds"] < 60:
            raise ValueError(
                f"scenario {name} startupFaultReadySeconds must be at least 60"
            )
        if scenarios[-1]["startupFaultReadySeconds"] > 0 and planned_jobs != 1:
            raise ValueError(
                f"scenario {name} startup fault-ready mode requires one job"
            )
    if not scenarios:
        raise ValueError("matrix must contain at least one scenario")
    return {
        "path": path,
        "sha256": matrix_sha256,
        "defaults": defaults,
        "scenarios": scenarios,
    }


def scenario_run_id(suite_id, scenario_name, repetition):
    suffix = f"-{scenario_name}-r{repetition}"
    run_id = suite_id[: 64 - len(suffix)] + suffix
    if not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"generated run id is invalid: {run_id}")
    return run_id


def build_command(
    runner,
    matrix,
    scenario,
    run_id,
    output_dir,
    execute,
    multi_workspace_runner=DEFAULT_MULTI_WORKSPACE_RUNNER,
):
    defaults = matrix["defaults"]
    if scenario.get("multiWorkspace"):
        command = [
            sys.executable,
            str(multi_workspace_runner),
            "--matrix",
            str(matrix["path"]),
            "--scenario",
            scenario["name"],
            "--run-id",
            run_id,
            "--expected-matrix-sha256",
            scenario["matrixSha256"],
            "--output-dir",
            str(output_dir),
        ]
        if execute:
            command.append("--execute")
        return command
    command = [
        sys.executable,
        str(runner),
        "--api-base",
        defaults["apiBase"],
        "--workspace",
        defaults["workspace"],
        "--api-key-env",
        defaults.get("apiKeyEnv", "VIDEO_BENCHMARK_API_KEY"),
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
        "--duration-seconds",
        str(scenario["durationSeconds"]),
        "--startup-timeout-seconds",
        str(scenario["startupTimeoutSeconds"]),
        "--cleanup-timeout-seconds",
        str(scenario["cleanupTimeoutSeconds"]),
        "--recovery-timeout-seconds",
        str(scenario["recoveryTimeoutSeconds"]),
        "--startup-fault-ready-seconds",
        str(scenario["startupFaultReadySeconds"]),
        "--poll-interval-seconds",
        str(scenario["pollIntervalSeconds"]),
    ]
    selector = "--source-id" if defaults.get("sourceId") else "--source-name"
    command.extend([selector, defaults.get("sourceId") or defaults["sourceName"]])
    for workload in scenario["workloads"]:
        command.extend(["--workload", workload])
    if scenario["maxFps"] is not None:
        command.extend(["--max-fps", str(scenario["maxFps"])])
    if scenario["publishOutput"]:
        command.append("--publish-output")
    if scenario["requireSingleProcessor"]:
        command.append("--require-single-processor")
    if execute:
        command.append("--execute")
    return command


def _write_summary(path, summary):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _result_path(output_dir, scenario, run_id):
    prefix = "api-multi-workspace" if scenario.get("multiWorkspace") else "api-corpus"
    return output_dir / f"{prefix}-{run_id}.json"


def _completed_result(path, expected_run_id):
    try:
        with path.open() as source:
            report = json.load(source)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if report.get("runId") != expected_run_id:
        return None
    if report.get("checkpoint", {}).get("phase") != "complete":
        return None
    if not report.get("endedAt") or not isinstance(report.get("success"), bool):
        return None
    return report


def _load_resume_summary(
    path,
    matrix,
    suite_id,
    execute,
    selected_scenarios,
    continue_on_error,
):
    with path.open() as source:
        summary = json.load(source)
    expected = {
        "schemaVersion": 2,
        "suiteId": suite_id,
        "environment": "staging",
        "matrixSha256": matrix["sha256"],
        "execute": execute,
        "selectedScenarios": selected_scenarios,
        "continueOnError": continue_on_error,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise ValueError(f"suite resume mismatch for {field}")
    if not isinstance(summary.get("runs"), list):
        raise ValueError("suite resume document has invalid runs")
    return summary


def run_matrix(
    matrix,
    runner,
    suite_id,
    output_dir,
    execute=False,
    selected=None,
    continue_on_error=False,
    multi_workspace_runner=DEFAULT_MULTI_WORKSPACE_RUNNER,
    sleep=time.sleep,
    stop_requested=lambda: None,
    popen_factory=subprocess.Popen,
    resume_summary=None,
):
    selected = set(selected or [])
    scenarios = [
        item for item in matrix["scenarios"] if not selected or item["name"] in selected
    ]
    missing = selected - {item["name"] for item in scenarios}
    if missing:
        raise ValueError(f"unknown scenarios: {', '.join(sorted(missing))}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"suite-{suite_id}.json"
    summary = resume_summary or {
        "schemaVersion": 2,
        "suiteId": suite_id,
        "environment": "staging",
        "matrix": str(matrix["path"]),
        "matrixSha256": matrix["sha256"],
        "startedAt": utc_now(),
        "execute": execute,
        "selectedScenarios": sorted(selected),
        "continueOnError": continue_on_error,
        "runs": [],
        "success": True,
    }
    summary.pop("endedAt", None)
    summary["resumedAt"] = utc_now() if resume_summary else None
    summary["success"] = True
    summary["interrupted"] = False
    _write_summary(summary_path, summary)

    stop = False
    existing = {item.get("runId"): item for item in summary["runs"]}
    for scenario in scenarios:
        for repetition in range(1, scenario["repetitions"] + 1):
            run_id = scenario_run_id(suite_id, scenario["name"], repetition)
            prior = existing.get(run_id)
            if prior is not None:
                if prior.get("status") == "spawn-failed":
                    summary["success"] = False
                    if not continue_on_error:
                        stop = True
                    if stop:
                        break
                    continue
                if prior.get("status") == "completed":
                    if execute:
                        result_report = _completed_result(
                            _result_path(output_dir, scenario, run_id), run_id
                        )
                        if result_report is None:
                            raise ValueError(
                                f"completed run {run_id} has no valid complete result"
                            )
                        expected_return = 0 if result_report["success"] else 1
                        if prior.get("returnCode") != expected_return:
                            raise ValueError(
                                f"completed run {run_id} disagrees with its result"
                            )
                    else:
                        expected_return = prior.get("returnCode")
                        if expected_return not in {0, 1}:
                            raise ValueError(
                                f"completed dry run {run_id} has invalid return code"
                            )
                    if expected_return != 0:
                        summary["success"] = False
                        if not continue_on_error:
                            stop = True
                    if stop:
                        break
                    continue
                result_report = _completed_result(
                    _result_path(output_dir, scenario, run_id), run_id
                )
                if result_report is None:
                    raise ValueError(
                        f"run {run_id} was interrupted without a complete result; "
                        "use its exact-run cleanup tool before resuming"
                    )
                prior.update(
                    {
                        "status": "completed",
                        "endedAt": result_report["endedAt"],
                        "returnCode": 0 if result_report["success"] else 1,
                        "reconciledOnResume": True,
                    }
                )
                if not result_report["success"]:
                    summary["success"] = False
                    if not continue_on_error:
                        stop = True
                _write_summary(summary_path, summary)
                if stop:
                    break
                continue
            command = build_command(
                runner,
                matrix,
                scenario,
                run_id,
                output_dir,
                execute,
                multi_workspace_runner=multi_workspace_runner,
            )
            run = {
                "scenario": scenario["name"],
                "repetition": repetition,
                "runId": run_id,
                "plannedJobs": scenario["plannedJobs"],
                "notes": scenario["notes"],
                "startedAt": utc_now(),
                "status": "starting",
                "command": command,
            }
            summary["runs"].append(run)
            _write_summary(summary_path, summary)
            try:
                process = popen_factory(command)
            except OSError as error:
                run.update(
                    {
                        "status": "spawn-failed",
                        "endedAt": utc_now(),
                        "error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                )
                summary["success"] = False
                _write_summary(summary_path, summary)
                if not continue_on_error:
                    stop = True
                    break
                continue
            run.update({"status": "running", "pid": process.pid})
            _write_summary(summary_path, summary)
            forwarded_signal = None
            while process.poll() is None:
                forwarded_signal = stop_requested()
                if forwarded_signal:
                    process.send_signal(forwarded_signal)
                    try:
                        process.wait(
                            timeout=scenario["cleanupTimeoutSeconds"] + 30
                        )
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        run["cleanupUncertain"] = True
                    break
                sleep(0.2)
            pending_signal = forwarded_signal or stop_requested()
            run.update(
                {
                    "status": "interrupted" if forwarded_signal else "completed",
                    "endedAt": utc_now(),
                    "returnCode": process.returncode,
                }
            )
            if pending_signal:
                if forwarded_signal:
                    run["forwardedSignal"] = signal.Signals(
                        forwarded_signal
                    ).name
                else:
                    run["stopObservedAfterChildExit"] = signal.Signals(
                        pending_signal
                    ).name
                summary["interrupted"] = True
                summary["success"] = False
                stop = True
            elif process.returncode != 0:
                summary["success"] = False
                if not continue_on_error:
                    stop = True
            _write_summary(summary_path, summary)
            if stop:
                break
            if execute and scenario["cooldownSeconds"]:
                sleep(scenario["cooldownSeconds"])
        if stop:
            break
    summary["endedAt"] = utc_now()
    _write_summary(summary_path, summary)
    return summary_path, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--scenario", action="append")
    parser.add_argument("--suite-id", default=None)
    parser.add_argument("--runner", default=DEFAULT_RUNNER)
    parser.add_argument(
        "--multi-workspace-runner", default=DEFAULT_MULTI_WORKSPACE_RUNNER
    )
    parser.add_argument("--output-dir", default=Path(__file__).with_name("results"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    try:
        matrix = load_matrix(args.matrix)
        suite_id = args.suite_id or datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        if not SAFE_RUN_ID.fullmatch(suite_id):
            raise ValueError("suite id must be filesystem-safe")
        runner = Path(args.runner).resolve()
        if not runner.is_file():
            raise ValueError(f"runner does not exist: {runner}")
        selected_names = set(args.scenario or [])
        required_api_key_envs = {
            env_name
            for scenario in matrix["scenarios"]
            if not selected_names or scenario["name"] in selected_names
            for env_name in scenario["requiredApiKeyEnvs"]
        }
        missing_api_key_envs = sorted(
            env_name
            for env_name in required_api_key_envs
            if args.execute and not os.environ.get(env_name)
        )
        if missing_api_key_envs:
            raise ValueError("one or more benchmark API keys are not configured")
        multi_workspace_runner = Path(args.multi_workspace_runner).resolve()
        if any(item.get("multiWorkspace") for item in matrix["scenarios"]):
            if not multi_workspace_runner.is_file():
                raise ValueError(
                    f"multi-workspace runner does not exist: {multi_workspace_runner}"
                )
        output_dir = Path(args.output_dir).resolve()
        summary_path = output_dir / f"suite-{suite_id}.json"
        if summary_path.exists() and not args.resume:
            raise ValueError(
                f"suite result already exists for suite id {suite_id}; "
                "use --resume or a new id"
            )
        if args.resume and not summary_path.exists():
            raise ValueError("--resume requires an existing suite result")
        resume_summary = (
            _load_resume_summary(
                summary_path,
                matrix,
                suite_id,
                args.execute,
                sorted(args.scenario or []),
                args.continue_on_error,
            )
            if args.resume
            else None
        )
        with RunLock(output_dir / f".suite-{suite_id}.lock"):
            with SignalStop() as stop:
                path, summary = run_matrix(
                    matrix=matrix,
                    runner=runner,
                    suite_id=suite_id,
                    output_dir=output_dir,
                    execute=args.execute,
                    selected=args.scenario,
                    continue_on_error=args.continue_on_error,
                    multi_workspace_runner=multi_workspace_runner,
                    stop_requested=(
                        lambda: stop.signum if stop.requested.is_set() else None
                    ),
                    resume_summary=resume_summary,
                )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(path)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
