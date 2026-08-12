#!/usr/bin/env python3
"""Run a resumable staging-only matrix of video workflow experiments.

Each scenario is executed in a fresh child process so interruption and cleanup
remain scoped to one experiment. The API key stays in the environment and is
never copied into commands, suite manifests, or result reports.
"""

import argparse
import json
import os
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
    if parsed <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return parsed


def load_matrix(path):
    path = Path(path).resolve()
    with path.open() as source:
        document = json.load(source)
    if document.get("schemaVersion") != 1:
        raise ValueError("matrix schemaVersion must be 1")
    if document.get("environment") != "staging":
        raise ValueError("matrix environment must be staging")

    defaults = dict(document.get("defaults") or {})
    defaults["apiBase"] = validate_api_base(
        defaults.get("apiBase")
        or "https://us-central1-roboflow-staging.cloudfunctions.net/light-v2-device"
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
    if not scenarios:
        raise ValueError("matrix must contain at least one scenario")
    return {"path": path, "defaults": defaults, "scenarios": scenarios}


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
    temporary.replace(path)


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
    summary = {
        "schemaVersion": 1,
        "suiteId": suite_id,
        "environment": "staging",
        "matrix": str(matrix["path"]),
        "startedAt": utc_now(),
        "execute": execute,
        "runs": [],
        "success": True,
    }
    _write_summary(summary_path, summary)

    stop = False
    for scenario in scenarios:
        for repetition in range(1, scenario["repetitions"] + 1):
            run_id = scenario_run_id(suite_id, scenario["name"], repetition)
            command = build_command(
                runner,
                matrix,
                scenario,
                run_id,
                output_dir,
                execute,
                multi_workspace_runner=multi_workspace_runner,
            )
            started = utc_now()
            result = subprocess.run(command, check=False)
            run = {
                "scenario": scenario["name"],
                "repetition": repetition,
                "runId": run_id,
                "plannedJobs": scenario["plannedJobs"],
                "notes": scenario["notes"],
                "startedAt": started,
                "endedAt": utc_now(),
                "returnCode": result.returncode,
                "command": command,
            }
            summary["runs"].append(run)
            if result.returncode != 0:
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
            raise ValueError(
                "required API key environment variables are not set: "
                + ", ".join(missing_api_key_envs)
            )
        multi_workspace_runner = Path(args.multi_workspace_runner).resolve()
        if any(item.get("multiWorkspace") for item in matrix["scenarios"]):
            if not multi_workspace_runner.is_file():
                raise ValueError(
                    f"multi-workspace runner does not exist: {multi_workspace_runner}"
                )
        output_dir = Path(args.output_dir).resolve()
        summary_path = output_dir / f"suite-{suite_id}.json"
        if summary_path.exists():
            raise ValueError(
                f"suite result already exists for suite id {suite_id}; use a new id"
            )
        with RunLock(output_dir / f".suite-{suite_id}.lock"):
            path, summary = run_matrix(
                matrix=matrix,
                runner=runner,
                suite_id=suite_id,
                output_dir=output_dir,
                execute=args.execute,
                selected=args.scenario,
                continue_on_error=args.continue_on_error,
                multi_workspace_runner=multi_workspace_runner,
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(path)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
