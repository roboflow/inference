"""Private-run orchestration and offline accounting for hosted platform E2E."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from importlib.metadata import distributions
from pathlib import Path
from time import perf_counter

import click
import pytest

_SHARDS = ("A", "B", "C", "D")
_PHASES = ("setup", "call", "teardown")


def _read(path: Path):
    with path.open() as handle:
        result = json.load(handle)

    return result


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def _digest(value) -> str:
    result = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _make_plan(
    nodeids: list[str],
    *,
    context: dict,
    allowed_skips: list[str],
    durations: dict[str, float] | None = None,
    file_timings: dict[str, float] | None = None,
) -> dict:
    _require(isinstance(allowed_skips, list), "invalid skip allowlist")
    _require(durations is None or isinstance(durations, dict), "invalid durations")
    _require(
        file_timings is None or isinstance(file_timings, dict), "invalid file timings"
    )
    _require(bool(nodeids), "empty collection")
    _require(len(nodeids) == len(set(nodeids)), "duplicate collected case")
    _require(all(isinstance(n, str) and "::" in n for n in nodeids), "invalid case")
    _require(set(allowed_skips) <= set(nodeids), "stale skip allowlist")
    timings = durations or {}
    file_timings = file_timings or {}
    _require(
        all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0
            for v in [*timings.values(), *file_timings.values()]
        ),
        "invalid duration data",
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for nodeid in sorted(nodeids):
        groups[nodeid.split("::", 1)[0]].append(nodeid)

    # A file is the conservative scheduling unit, including all parametrizations.
    known = [
        file_timings[f] / len(ids) for f, ids in groups.items() if f in file_timings
    ]
    fallback = statistics.median(known) if known else 1.0
    weights = {
        name: (
            sum(timings[n] for n in ids)
            if all(n in timings for n in ids)
            else file_timings.get(name, sum(timings.get(n, fallback) for n in ids))
        )
        for name, ids in groups.items()
    }
    assignments: dict[str, list[str]] = {shard: [] for shard in _SHARDS}
    loads = dict.fromkeys(_SHARDS, 0.0)
    for name in sorted(groups, key=lambda name: (-weights[name], name)):
        shard = min(
            _SHARDS, key=lambda shard: (loads[shard], len(assignments[shard]), shard)
        )
        assignments[shard].extend(groups[name])
        loads[shard] += weights[name]

    plan = {
        "schema": 1,
        "context": context,
        "manifest": sorted(nodeids),
        "assignments": {s: sorted(ids) for s, ids in assignments.items()},
        "allowed_skips": sorted(set(allowed_skips)),
        "timings_hash": _digest({"cases": timings, "files": file_timings}),
        "estimated_seconds": loads,
        "weight_basis": {
            name: (
                "cases"
                if all(n in timings for n in ids)
                else "file-estimate" if name in file_timings else "fallback"
            )
            for name, ids in groups.items()
        },
    }
    plan["plan_hash"] = _digest(plan)
    return plan


def _validate_plan(plan: dict) -> None:
    _require(plan["schema"] == 1, "unsupported plan")
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    _require(plan["plan_hash"] == _digest(payload), "plan changed")
    manifest = plan["manifest"]
    _require(bool(manifest) and len(manifest) == len(set(manifest)), "invalid manifest")
    _require(set(plan["assignments"]) == set(_SHARDS), "invalid shard set")
    assigned = [n for ids in plan["assignments"].values() for n in ids]
    _require(len(assigned) == len(set(assigned)), "overlapping assignments")
    _require(set(assigned) == set(manifest), "incomplete assignments")
    _require(set(plan["allowed_skips"]) <= set(manifest), "invalid skip allowlist")


def _verify_identity(policy: Path) -> dict:
    # This adapter lives in the private caller, never in an untrusted test ref.
    completed = subprocess.run(
        [sys.executable, str(policy / "verify_deployment.py")],
        capture_output=True,
        timeout=120,
        check=True,
    )
    observed = json.loads(completed.stdout)
    _require(observed.get("ready") is True, "deployment is not ready")
    _require(
        observed.get("deployment_ref") == os.environ["DEPLOYMENT_REFERENCE"],
        "deployment reference mismatch",
    )
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", observed.get("candidate_sha", ""))),
        "invalid candidate identity",
    )
    _require(
        bool(re.fullmatch(r"sha256:[0-9a-f]{64}", observed.get("image_digest", ""))),
        "invalid image identity",
    )
    _require(
        observed["candidate_sha"] == os.environ["EXPECTED_CANDIDATE_SHA"],
        "candidate identity mismatch",
    )
    _require(
        observed["image_digest"] == os.environ["EXPECTED_IMAGE_DIGEST"],
        "image identity mismatch",
    )
    # Do not propagate arbitrary adapter output (URLs, tokens, payloads).
    identity = {
        k: observed[k] for k in ("deployment_ref", "candidate_sha", "image_digest")
    }
    return identity


class _Recorder:
    def __init__(
        self,
        *,
        plan: dict | None = None,
        shard: str | None = None,
        warmup: bool = False,
    ):
        self.plan = plan
        self.shard = shard
        self.warmup = warmup
        self.manifest: list[str] = []
        self.cases: dict[str, dict] = {}
        self.collection_errors = 0
        self.fixtures: dict[str, dict] = {}

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items) -> None:
        """Verify collection and select an exact shard or the warmup prerequisite.

        Args:
            session: Active pytest session.
            config: Active pytest configuration.
            items: Collected items, modified in place.
        """
        self.manifest = sorted(item.nodeid for item in items)
        if self.plan is None:
            return

        _require(self.manifest == self.plan["manifest"], "collection differs from plan")
        if self.warmup:
            _require(bool(items), "warmup requires a collected suite")
            # A function in the same module inherits the hosted suite's session
            # autouse fixtures, without running a real test twice.
            items[:] = [
                pytest.Function.from_parent(
                    items[0].parent, name="_hosted_e2e_warmup", callobj=lambda: None
                )
            ]
            return

        selected = set(self.plan["assignments"][self.shard])
        removed = [item for item in items if item.nodeid not in selected]
        items[:] = [item for item in items if item.nodeid in selected]
        config.hook.pytest_deselected(items=removed)

    def pytest_collectreport(self, report) -> None:
        """Count collection failures without copying exception text.

        Args:
            report: Pytest collection report.
        """
        self.collection_errors += int(report.failed)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_setup(self, item):
        """Observe every fixture setup, including retries.

        Args:
            item: Current test item.

        Yields:
            None: Control to pytest's setup hooks.
        """
        yield from self._fixture_report(item, phase="setup")

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_teardown(self, item):
        """Observe every teardown, including internal retry teardowns.

        Args:
            item: Current test item.

        Yields:
            None: Control to pytest's teardown hooks.
        """
        yield from self._fixture_report(item, phase="teardown")

    def _fixture_report(self, item, *, phase: str):
        started = perf_counter()
        outcome = yield
        status = "passed"
        if outcome.excinfo is not None:
            status = (
                "skipped"
                if isinstance(outcome.excinfo[1], pytest.skip.Exception)
                else "failed"
            )
        self.fixtures[item.nodeid][phase].append((status, perf_counter() - started))

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """Retain every retry outcome from the pinned pytest-retry integration.

        Args:
            item: Current test item.
            nextitem: Next item, or None.

        Yields:
            None: Control to pytest and pytest-retry.
        """
        self.fixtures[item.nodeid] = {"setup": [], "teardown": []}
        started = perf_counter()
        yield
        from pytest_retry.retry_plugin import retry_manager

        stats = retry_manager.node_stats.get(item.nodeid)
        if stats is not None:
            # pytest-retry does not emit a standard logreport for every attempt.
            # Its statistics preserve initial failures after a successful retry.
            self.cases[item.nodeid] = {
                "outcomes": {p: list(stats["outcomes"][p]) for p in _PHASES},
                "durations": {p: list(stats["durations"][p])[1:] for p in _PHASES},
                "elapsed_seconds": perf_counter() - started,
            }
            for phase, records in self.fixtures[item.nodeid].items():
                self.cases[item.nodeid]["outcomes"][phase] = [r[0] for r in records]
                self.cases[item.nodeid]["durations"][phase] = [r[1] for r in records]


def _pytest(suite: Path, *, recorder: _Recorder, collect: bool = False) -> int:
    # Disable ambient plugins/arguments that could introduce workers or silently
    # deselect cases. The existing flaky markers still control the retry policy.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    args = [
        str(suite),
        "-p",
        "pytest_retry.retry_plugin",
        "-o",
        "addopts=",
        "--durations=0",
        "--durations-min=0",
    ]
    if collect:
        args.append("--collect-only")
    result = int(pytest.main(args, plugins=[recorder]))
    return result


def _context(policy: Path) -> dict:
    context = {
        "run_key": os.environ["HOSTED_E2E_RUN_KEY"],
        "test_sha": os.environ["HOSTED_E2E_TEST_SHA"],
        "target": os.environ["HOSTED_E2E_TARGET"],
        "runner_sha": os.environ["HOSTED_E2E_RUNNER_SHA"],
        "policy_sha": os.environ["HOSTED_E2E_POLICY_SHA"],
        "identity": _verify_identity(policy),
        "dependencies_hash": _digest(
            sorted(
                (dist.metadata["Name"].lower(), dist.version)
                for dist in distributions()
            )
        ),
    }
    _require(
        bool(re.fullmatch(r"[0-9a-f]{40}", context["test_sha"])), "invalid test SHA"
    )
    return context


def _file_timings(profile: dict, *, nodeids: list[str], target: str) -> dict:
    _require(
        profile["schema"] == 1 and profile["target"] == target, "invalid timing profile"
    )
    groups = defaultdict(list)
    for nodeid in nodeids:
        groups[nodeid.split("::", 1)[0]].append(nodeid)

    result = {}
    for name, ids in groups.items():
        entry = profile["files"].get(name)
        if entry is None:
            continue

        # Changed source/collection must use fallback, not stale historical data.
        if (
            entry["cases"] == len(ids)
            and entry["source_hash"]
            == hashlib.sha256(Path(name).read_bytes()).hexdigest()
        ):
            result[name] = entry["seconds"]

    return result


def _prepare(*, suite: Path, policy: Path, output: Path) -> None:
    context = _context(policy)
    recorder = _Recorder()
    _require(_pytest(suite, recorder=recorder, collect=True) == 0, "collection failed")
    _require(recorder.collection_errors == 0, "collection failed")
    skip_path = policy / "allowed_skips.json"
    timings_path = policy / "durations.json"
    profile_path = policy / "timing_profile.json"
    plan = _make_plan(
        recorder.manifest,
        context=context,
        allowed_skips=_read(skip_path) if skip_path.exists() else [],
        durations=_read(timings_path) if timings_path.exists() else None,
        file_timings=(
            _file_timings(
                _read(profile_path), nodeids=recorder.manifest, target=context["target"]
            )
            if profile_path.exists()
            else None
        ),
    )
    _write(output / "plan.json", plan)
    os.environ["SKIP_WARMUP"] = "false"
    # Preparation and warmup run in separate CLI processes: conftest reads
    # SKIP_WARMUP at import time, so pytest must not reuse its collection import.


def _warmup(*, suite: Path, policy: Path, output: Path) -> None:
    plan = _read(output / "plan.json")
    _validate_plan(plan)
    _require(_context(policy) == plan["context"], "identity changed before warmup")
    os.environ["SKIP_WARMUP"] = "false"
    recorder = _Recorder(plan=plan, warmup=True)
    _require(_pytest(suite, recorder=recorder) == 0, "warmup failed")
    _require(recorder.collection_errors == 0, "warmup collection failed")
    _require(len(recorder.cases) == 1, "warmup did not execute")
    _require(
        all(
            case["outcomes"][phase] == ["passed"]
            for case in recorder.cases.values()
            for phase in _PHASES
        ),
        "warmup did not pass",
    )
    _require(_context(policy) == plan["context"], "identity changed during warmup")
    readiness = {
        "schema": 1,
        "plan_hash": plan["plan_hash"],
        "context": plan["context"],
        "ready": True,
    }
    _write(output / "readiness.json", readiness)
    _write(output / "warmup.json", {"cases": recorder.cases})


def _run_shard(*, suite: Path, policy: Path, output: Path, shard: str) -> int:
    plan = _read(output / "plan.json")
    _validate_plan(plan)
    report = {
        "schema": 1,
        "shard": shard,
        "plan_hash": plan["plan_hash"],
        "context": plan["context"],
        "completed": False,
        "exit_code": 1,
        "collection_errors": 0,
        "cases": {},
    }
    try:
        readiness = _read(output / "readiness.json")
        _require(
            readiness
            == {
                "schema": 1,
                "plan_hash": plan["plan_hash"],
                "context": plan["context"],
                "ready": True,
            },
            "invalid readiness record",
        )
        _require(_context(policy) == plan["context"], "identity changed before shard")
        os.environ["SKIP_WARMUP"] = "true"
        recorder = _Recorder(plan=plan, shard=shard)
        code = _pytest(suite, recorder=recorder)
        report.update(
            cases=recorder.cases, collection_errors=recorder.collection_errors
        )
        # An explicitly empty assignment is valid only after collection matches.
        if (
            not plan["assignments"][shard]
            and code == 5
            and not recorder.collection_errors
        ):
            code = 0
        report["exit_code"] = code
        _require(_context(policy) == plan["context"], "identity changed during shard")
        report["completed"] = True
    finally:
        _write(output / f"shard-{shard}.json", report)

    return report["exit_code"]


def _aggregate(plan: dict, reports: list[dict], *, readiness: dict) -> dict:
    _validate_plan(plan)
    issues = []
    failures = []
    retried = []
    unexpected_skips = []
    seen = set()
    expected_readiness = {
        "schema": 1,
        "plan_hash": plan["plan_hash"],
        "context": plan["context"],
        "ready": True,
    }
    if readiness != expected_readiness:
        issues.append("readiness")
    for report in reports:
        shard = report.get("shard")
        if shard not in _SHARDS or shard in seen:
            issues.append("duplicate-or-unknown-shard")
            continue
        seen.add(shard)
        if (
            report.get("schema") != 1
            or report.get("plan_hash") != plan["plan_hash"]
            or report.get("context") != plan["context"]
            or report.get("completed") is not True
            or type(report.get("exit_code")) is not int
            or report["exit_code"] != 0
            or report.get("collection_errors") != 0
        ):
            issues.append(f"invalid-shard-{shard}")
        cases = report.get("cases", {})
        if set(cases) != set(plan["assignments"][shard]):
            issues.append(f"coverage-{shard}")
        for nodeid, case in cases.items():
            outcomes = case["outcomes"]
            _require(set(outcomes) == set(_PHASES), "invalid phase report")
            _require(
                all(isinstance(outcomes[p], list) for p in _PHASES), "invalid outcomes"
            )
            _require(
                all(
                    o in {"passed", "failed", "skipped"}
                    for p in _PHASES
                    for o in outcomes[p]
                ),
                "invalid outcome",
            )
            durations = case["durations"]
            _require(set(durations) == set(_PHASES), "invalid timing phases")
            _require(
                all(
                    isinstance(durations[p], list)
                    and len(durations[p]) == len(outcomes[p])
                    for p in _PHASES
                ),
                "incomplete phase timings",
            )
            _require(
                all(
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                    for value in [
                        case["elapsed_seconds"],
                        *[v for p in _PHASES for v in durations[p]],
                    ]
                ),
                "invalid timings",
            )
            calls = outcomes["call"]
            if len(calls) > 1:
                retried.append(nodeid)
            skipped = (outcomes["setup"] == ["skipped"] and not calls) or (
                bool(calls) and calls[-1] == "skipped"
            )
            if skipped and nodeid not in plan["allowed_skips"]:
                unexpected_skips.append(nodeid)
            bad_fixture = (
                not outcomes["setup"]
                or not outcomes["teardown"]
                or "failed" in outcomes["setup"]
                or "failed" in outcomes["teardown"]
                or outcomes["teardown"][-1] != "passed"
                or (
                    any(o != "passed" for o in outcomes["setup"])
                    and not (outcomes["setup"] == ["skipped"] and not calls)
                )
            )
            if bad_fixture or (not skipped and (not calls or calls[-1] != "passed")):
                failures.append(nodeid)
    if seen != set(_SHARDS):
        issues.append("missing-shards")
    accepted = not (issues or failures or unexpected_skips)
    result = {
        "accepted": accepted,
        "issues": issues,
        "failures": sorted(failures),
        "unexpected_skips": sorted(unexpected_skips),
        "retried": sorted(retried),
    }
    return result


@click.command()
@click.argument(
    "operation", type=click.Choice(["prepare", "warmup", "shard", "aggregate"])
)
@click.option(
    "--suite",
    type=click.Path(path_type=Path),
    default="tests/inference/hosted_platform_tests",
)
@click.option(
    "--policy",
    type=click.Path(path_type=Path),
    default="../policy/.github/hosted-e2e",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="evidence",
)
@click.option(
    "--shard",
    type=click.Choice(_SHARDS),
)
def main(
    operation: str, suite: Path, policy: Path, output: Path, shard: str | None
) -> None:
    """Run a private acceptance stage; export only a fixed status on aggregation.

    Args:
        operation (str): Stage to execute.
        suite (Path): Hosted suite or an offline synthetic suite.
        policy (Path): Trusted private deployment adapter and optional policies.
        output (Path): Restricted evidence directory.
        shard (str | None): Shard identifier for the shard stage.
    """
    try:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            _require(
                os.environ.get("HOSTED_E2E_PRIVATE") == "true",
                "private caller required",
            )
        if operation == "prepare":
            _prepare(suite=suite, policy=policy, output=output)
        elif operation == "warmup":
            _warmup(suite=suite, policy=policy, output=output)
        elif operation == "shard":
            _require(shard in _SHARDS, "shard required")
            code = _run_shard(suite=suite, policy=policy, output=output, shard=shard)
            raise SystemExit(code)
        else:
            result = _aggregate(
                _read(output / "plan.json"),
                [_read(path) for path in sorted(output.glob("shard-*.json"))],
                readiness=_read(output / "readiness.json"),
            )
            _write(output / "acceptance.json", result)
            click.echo("accepted" if result["accepted"] else "failed")
            raise SystemExit(0 if result["accepted"] else 1)
    except (Exception, KeyboardInterrupt):
        # Error messages / adapter output can contain credentials or private
        # metadata. Detailed pytest logs belong only to the private caller.
        click.echo("Hosted E2E stage failed; inspect restricted evidence.", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
