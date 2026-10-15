"""Offline regression coverage for private hosted acceptance orchestration."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import hosted_e2e as runner
import yaml


class SchedulingTests(unittest.TestCase):
    """Exercise scheduling without collecting or contacting hosted services."""

    def test_file_weights_override_case_counts_and_cover_new_cases(self):
        """Keep parametrizations together and assign new cases exactly once."""
        nodes = [f"large.py::test_x[{i}]" for i in range(50)] + [
            "slow.py::test_x[legacy]",
            "slow.py::test_x[header]",
            "new.py::test_new",
            "other.py::test_other",
        ]
        first = runner._make_plan(
            nodes,
            context={},
            allowed_skips=[],
            file_timings={"slow.py": 180, "large.py": 10},
        )
        second = runner._make_plan(
            list(reversed(nodes)),
            context={},
            allowed_skips=[],
            file_timings={"slow.py": 180, "large.py": 10},
        )
        self.assertEqual(first, second)
        self.assertEqual(first["assignments"]["A"], sorted(nodes[50:52]))
        runner._validate_plan(first)
        self.assertEqual(first["weight_basis"]["new.py"], "fallback")

    def test_zero_weights_still_spread_files(self):
        """Avoid concentrating zero-duration measurements on one shard."""
        nodes = [f"{i}.py::test_x" for i in range(4)]
        plan = runner._make_plan(
            nodes, context={}, allowed_skips=[], durations=dict.fromkeys(nodes, 0)
        )
        self.assertEqual([len(v) for v in plan["assignments"].values()], [1] * 4)

    def test_invalid_weights_and_manifests_fail(self):
        """Reject malformed or incomplete scheduling inputs."""
        for bad in [float("nan"), float("inf"), -1, True]:
            with self.subTest(weight=bad), self.assertRaises(ValueError):
                runner._make_plan(
                    ["x.py::test_x"],
                    context={},
                    allowed_skips=[],
                    durations={"x.py::test_x": bad},
                )
        for nodes in [[], ["x.py::test_x", "x.py::test_x"]]:
            with self.assertRaises(ValueError):
                runner._make_plan(nodes, context={}, allowed_skips=[])

    def test_changed_source_or_count_invalidates_file_estimate(self):
        """Only reuse historical weights for matching source and collection."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test_x.py"
            path.write_text("def test_x(): pass\n")
            node = f"{path}::test_x"
            profile = {
                "schema": 1,
                "target": "staging",
                "files": {
                    str(path): {
                        "seconds": 10,
                        "cases": 1,
                        "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                },
            }
            self.assertEqual(
                runner._file_timings(profile, nodeids=[node], target="staging"),
                {str(path): 10},
            )
            self.assertEqual(
                runner._file_timings(
                    profile, nodeids=[node, node + "2"], target="staging"
                ),
                {},
            )
            path.write_text("def test_x(): assert True\n")
            self.assertEqual(
                runner._file_timings(profile, nodeids=[node], target="staging"), {}
            )


class AccountingTests(unittest.TestCase):
    """Reject incomplete or inconsistent acceptance evidence."""

    def setUp(self):
        """Create a complete four-shard passing record."""
        self.plan = runner._make_plan(
            [f"{s}.py::test_x" for s in "ABCD"],
            context={"run_key": "1:1"},
            allowed_skips=[],
        )
        self.ready = {
            "schema": 1,
            "plan_hash": self.plan["plan_hash"],
            "context": self.plan["context"],
            "ready": True,
        }
        self.reports = [
            {
                "schema": 1,
                "shard": s,
                "plan_hash": self.plan["plan_hash"],
                "context": self.plan["context"],
                "completed": True,
                "exit_code": 0,
                "collection_errors": 0,
                "cases": {
                    n: {
                        "outcomes": {p: ["passed"] for p in runner._PHASES},
                        "durations": {p: [0.1] for p in runner._PHASES},
                        "elapsed_seconds": 0.4,
                    }
                    for n in ids
                },
            }
            for s, ids in self.plan["assignments"].items()
        ]

    def _accepted(self, reports=None):
        result = runner._aggregate(
            self.plan,
            self.reports if reports is None else reports,
            readiness=self.ready,
        )
        return result["accepted"]

    def test_full_coverage_passes_and_retry_history_survives(self):
        """Accept a final retry pass while retaining the initial failure."""
        self.assertTrue(self._accepted())
        case = next(iter(self.reports[0]["cases"].values()))
        case["outcomes"]["call"] = ["failed", "passed"]
        case["durations"]["call"] = [0.1, 0.1]
        result = runner._aggregate(self.plan, self.reports, readiness=self.ready)
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["retried"]), 1)

    def test_missing_duplicate_cancelled_drift_and_collection_fail(self):
        """Missing, cancelled and mismatched runs cannot become acceptance."""
        self.assertFalse(self._accepted(self.reports[:-1]))
        self.assertFalse(self._accepted(self.reports + [self.reports[0]]))
        for key, value in [
            ("completed", False),
            ("exit_code", 1),
            ("context", {"run_key": "1:2"}),
            ("plan_hash", "changed"),
            ("collection_errors", 1),
            ("cases", {}),
        ]:
            with self.subTest(field=key):
                reports = copy.deepcopy(self.reports)
                reports[0][key] = value
                self.assertFalse(self._accepted(reports))

    def test_failure_and_unexpected_skip_fail(self):
        """Do not hide call or fixture failures behind skips or retries."""
        for phase, outcomes in [
            ("call", ["failed"]),
            ("call", ["skipped"]),
            ("call", ["skipped", "failed"]),
            ("setup", ["failed"]),
            ("teardown", ["failed"]),
            ("call", []),
        ]:
            reports = copy.deepcopy(self.reports)
            case = next(iter(reports[0]["cases"].values()))
            case["outcomes"][phase] = outcomes
            case["durations"][phase] = [0.1] * len(outcomes)
            self.assertFalse(self._accepted(reports))

    def test_explicit_skip_is_allowed_but_cannot_hide_a_failure(self):
        """Allow only deliberate final skips with valid fixture evidence."""
        node = self.plan["assignments"]["A"][0]
        self.plan["allowed_skips"] = [node]
        self.plan["plan_hash"] = runner._digest(
            {k: v for k, v in self.plan.items() if k != "plan_hash"}
        )
        self.ready["plan_hash"] = self.plan["plan_hash"]
        for report in self.reports:
            report["plan_hash"] = self.plan["plan_hash"]
        outcomes = self.reports[0]["cases"][node]["outcomes"]
        outcomes.update(setup=["skipped"], call=[])
        self.reports[0]["cases"][node]["durations"]["call"] = []
        self.assertTrue(self._accepted())
        outcomes.update(setup=["passed"], call=["skipped", "failed"])
        self.reports[0]["cases"][node]["durations"]["call"] = [0.1, 0.1]
        self.assertFalse(self._accepted())


class CommandTests(unittest.TestCase):
    """Run real pytest processes against synthetic suites with no network."""

    def setUp(self):
        """Prepare fake deployment verification and session warmup fixtures."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root / "suite"
        self.suite.mkdir()
        self.output = self.root / "evidence"
        self.policy = self.root / "policy"
        self.policy.mkdir()
        self.script = Path(runner.__file__).resolve()
        self.env = {
            **os.environ,
            "HOSTED_E2E_RUN_KEY": "offline:1",
            "HOSTED_E2E_RUNNER_SHA": "a" * 40,
            "HOSTED_E2E_POLICY_SHA": "d" * 40,
            "HOSTED_E2E_TEST_SHA": "a" * 40,
            "HOSTED_E2E_TARGET": "staging",
            "DEPLOYMENT_REFERENCE": "offline",
            "EXPECTED_CANDIDATE_SHA": "b" * 40,
            "EXPECTED_IMAGE_DIGEST": "sha256:" + "c" * 64,
            "GITHUB_ACTIONS": "false",
        }
        self.policy.joinpath("verify_deployment.py").write_text(
            'import json, os\nprint(json.dumps({"ready": True, '
            '"deployment_ref": os.environ["DEPLOYMENT_REFERENCE"], '
            '"candidate_sha": os.environ["EXPECTED_CANDIDATE_SHA"], '
            '"image_digest": os.environ["EXPECTED_IMAGE_DIGEST"]}))\n'
        )
        self.suite.joinpath("conftest.py").write_text(
            "import os, pytest\nfrom pathlib import Path\n"
            '@pytest.fixture(scope="session", autouse=True)\n'
            "def warmup():\n"
            '    if os.environ["SKIP_WARMUP"] == "false":\n'
            '        p=Path("warmups.txt")\n'
            '        p.write_text(p.read_text()+"1" if p.exists() else "1")\n'
        )
        for name in "abcd":
            self.suite.joinpath(f"test_{name}.py").write_text(
                "import pytest\n"
                '@pytest.mark.parametrize("auth", ["legacy", "header"])\n'
                "def test_ok(auth): assert auth\n"
            )
        self.suite.joinpath("test_a.py").write_text(
            "import pytest\ncounter=0\n@pytest.mark.flaky(retries=1, delay=0)\n"
            "def test_retry():\n    global counter\n    counter+=1\n    assert counter>1\n"
        )

    def _run(self, operation, *, shard=None, expected=0):
        args = [
            sys.executable,
            str(self.script),
            operation,
            "--suite",
            "suite",
            "--policy",
            str(self.policy),
            "--output",
            str(self.output),
        ]
        if shard:
            args.extend(["--shard", shard])
        result = subprocess.run(
            args,
            env=self.env,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_four_shards_one_warmup_and_retry_evidence(self):
        """Execute every synthetic case once, with the existing retry policy."""
        self._run("prepare")
        self._run("warmup")
        for shard in "ABCD":
            self._run("shard", shard=shard)
        result = self._run("aggregate")
        self.assertEqual(result.stdout, "accepted\n")
        self.assertEqual(self.root.joinpath("warmups.txt").read_text(), "1")
        acceptance = json.loads(self.output.joinpath("acceptance.json").read_text())
        self.assertEqual(len(acceptance["retried"]), 1)

    def test_collection_drift_fails_and_writes_incomplete_report(self):
        """A new case after planning invalidates the shard."""
        self._run("prepare")
        self._run("warmup")
        self.suite.joinpath("test_new.py").write_text("def test_new(): pass\n")
        self._run("shard", shard="A", expected=3)
        self._run("aggregate", expected=1)

    def _exercise_explicit_skip(self, *, allowlisted: bool) -> None:
        nodeid = "suite/test_skipped.py::test_disabled"
        self.suite.joinpath("test_skipped.py").write_text(
            "import pytest\n"
            '@pytest.mark.skip(reason="Deliberately disabled")\n'
            "def test_disabled():\n"
            '    raise AssertionError("Skipped test body must not execute")\n'
        )
        self.policy.joinpath("allowed_skips.json").write_text(
            json.dumps([nodeid] if allowlisted else [])
        )

        self._run("prepare")
        self._run("warmup")
        for shard in "ABCD":
            self._run("shard", shard=shard)
        result = self._run("aggregate", expected=0 if allowlisted else 1)

        reports = [
            json.loads(path.read_text()) for path in self.output.glob("shard-*.json")
        ]
        skipped_cases = [r["cases"][nodeid] for r in reports if nodeid in r["cases"]]
        self.assertEqual(len(skipped_cases), 1)
        self.assertEqual(
            skipped_cases[0]["outcomes"],
            {"setup": ["skipped"], "call": [], "teardown": ["passed"]},
        )
        self.assertEqual(skipped_cases[0]["durations"]["call"], [])
        acceptance = json.loads(self.output.joinpath("acceptance.json").read_text())
        self.assertEqual(acceptance["accepted"], allowlisted)
        self.assertEqual(
            acceptance["unexpected_skips"], [] if allowlisted else [nodeid]
        )
        self.assertEqual(acceptance["failures"], [])
        self.assertEqual(acceptance["issues"], [])
        self.assertEqual(result.stdout, "accepted\n" if allowlisted else "failed\n")
        self.assertEqual(self.root.joinpath("warmups.txt").read_text(), "1")

    def test_allowlisted_explicit_skip_is_recorded_and_accepted(self):
        """Retain an allowed setup skip in real pytest subprocess evidence."""
        self._exercise_explicit_skip(allowlisted=True)

    def test_unexpected_explicit_skip_is_recorded_and_rejected(self):
        """Reject an actual pytest skip absent from the prepared allowlist."""
        self._exercise_explicit_skip(allowlisted=False)

    def test_adapter_output_is_not_echoed_and_public_runs_are_rejected(self):
        """Keep verifier exceptions and credentialed execution private."""
        self.policy.joinpath("verify_deployment.py").write_text(
            'raise RuntimeError("synthetic-private-value")\n'
        )
        result = self._run("prepare", expected=1)
        self.assertNotIn("synthetic-private-value", result.stdout + result.stderr)
        self.env.update(GITHUB_ACTIONS="true", HOSTED_E2E_PRIVATE="false")
        result = self._run("prepare", expected=1)
        self.assertNotIn("synthetic-private-value", result.stdout + result.stderr)

    def test_identity_drift_and_failed_warmup_prevent_acceptance(self):
        """Reject changed deployments and a failing prerequisite."""
        self._run("prepare")
        self.env["EXPECTED_CANDIDATE_SHA"] = "e" * 40
        self._run("warmup", expected=1)
        self.assertFalse(self.output.joinpath("readiness.json").exists())
        self.env["EXPECTED_CANDIDATE_SHA"] = "b" * 40
        self.suite.joinpath("conftest.py").write_text(
            'import pytest\n@pytest.fixture(scope="session", autouse=True)\n'
            'def warmup(): raise RuntimeError("failed warmup")\n'
        )
        self._run("warmup", expected=1)
        self.assertFalse(self.output.joinpath("readiness.json").exists())
        self._run("shard", shard="A", expected=1)
        self._run("aggregate", expected=1)

    def test_retry_fixture_failure_cannot_be_hidden_by_passing_call(self):
        """Catch an internal retry setup failure omitted by plugin statistics."""
        self.suite.joinpath("test_a.py").write_text(
            "import pytest\ncalls=0\nsetups=0\n"
            "@pytest.fixture(autouse=True)\ndef fragile():\n"
            "    global setups\n    setups+=1\n    assert setups != 2\n"
            "@pytest.mark.flaky(retries=1, delay=0)\ndef test_retry():\n"
            "    global calls\n    calls+=1\n    assert calls>1\n"
        )
        self._run("prepare")
        self._run("warmup")
        for shard in "ABCD":
            self._run("shard", shard=shard)
        self._run("aggregate", expected=1)
        acceptance = json.loads(self.output.joinpath("acceptance.json").read_text())
        self.assertEqual(acceptance["failures"], ["suite/test_a.py::test_retry"])


class WorkflowGuardTests(unittest.TestCase):
    """Execute the real workflow guard with synthetic caller contexts."""

    def test_guard_rejects_public_untrusted_or_unvalidated_callers(self):
        """Gate secrets behind a private default-branch manual caller."""
        path = (
            Path(__file__).resolve().parents[1]
            / "workflows/hosted_inference_e2e_sharded.yml"
        )
        workflow = yaml.safe_load(path.read_text())
        script = workflow["jobs"]["guard"]["steps"][0]["run"]
        valid = {
            **os.environ,
            "HOSTED_E2E_PRIVATE": "true",
            "EVENT_NAME": "workflow_dispatch",
            "RUN_REF": "refs/heads/main",
            "DEFAULT_REF": "refs/heads/main",
            "LOAD_VALIDATED": "true",
            "HOSTED_E2E_TARGET": "staging",
            "HOSTED_E2E_TEST_SHA": "a" * 40,
            "HOSTED_E2E_RUNNER_SHA": "b" * 40,
        }
        changes = [
            {},
            {"HOSTED_E2E_PRIVATE": "false"},
            {"EVENT_NAME": "pull_request"},
            {"RUN_REF": "refs/heads/unreviewed"},
            {"LOAD_VALIDATED": "false"},
            {"HOSTED_E2E_TEST_SHA": "main; false"},
            {"HOSTED_E2E_TARGET": "other"},
        ]
        for changed in changes:
            result = subprocess.run(
                ["bash", "-e", "-c", script],
                env={**valid, **changed},
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode == 0, not changed, result.stderr)


if __name__ == "__main__":
    unittest.main()
