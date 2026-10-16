"""Regression tests for the real manual entrypoints and encrypted stage transport."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hosted_e2e_repository as repository
import yaml


class RepositoryRunTests(unittest.TestCase):
    """Execute preparation and isolated shard jobs without hosted credentials."""

    def setUp(self):
        """Create identical checkouts and synthetic credentials for every job."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = Path(repository.__file__).resolve()
        self.env = {
            **os.environ,
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "roboflow/inference",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/main",
            "HOSTED_E2E_DEFAULT_REF": "refs/heads/main",
            "GITHUB_SHA": "a" * 40,
            "HOSTED_E2E_APPROVED_REF": "refs/heads/main",
            "HOSTED_E2E_APPROVED_SHA": "a" * 40,
            "HOSTED_E2E_RUN_KEY": "123:1",
            "HOSTED_E2E_TEST_SHA": "a" * 40,
            "HOSTED_E2E_RUNNER_SHA": "a" * 40,
            "HOSTED_E2E_POLICY_SHA": "a" * 40,
            "HOSTED_E2E_TARGET": "staging",
            "HOSTED_PLATFORM_TESTS_PROJECT": "roboflow-staging-serverless",
            **{
                key: "synthetic-private-canary-123456789"
                for key in repository._CREDENTIALS
            },
        }
        for name in ["prepare", *"ABCD", "aggregate"]:
            job = self.root / name
            (job / "suite").mkdir(parents=True)
            (job / "policy").mkdir()
            (job / "incoming").mkdir()
            (job / "suite/conftest.py").write_text(
                "import os, pytest\nfrom pathlib import Path\n"
                '@pytest.fixture(scope="session", autouse=True)\n'
                "def warmup():\n"
                '    if os.environ["SKIP_WARMUP"] == "false":\n'
                '        Path("warmup-ran").write_text("yes")\n'
            )
            for letter in "abcd":
                (job / f"suite/test_{letter}.py").write_text(
                    "import os\ndef test_ok():\n"
                    '    print(os.environ["HOSTED_PLATFORM_TESTS_API_KEY"])\n'
                    "    assert True\n"
                )
            (job / "suite/test_a.py").write_text(
                "import pytest\ncalls=0\n"
                "@pytest.mark.flaky(retries=1, delay=0)\n"
                "def test_retry():\n"
                "    global calls\n    calls+=1\n    assert calls>1\n"
            )
            (job / "suite/test_skip.py").write_text(
                'import pytest\n@pytest.mark.skip(reason="deliberate")\n'
                "def test_disabled(): assert False\n"
            )
            (job / "policy/allowed_skips.json").write_text(
                json.dumps(["suite/test_skip.py::test_disabled"])
            )
        self.results = []

    def _run(self, job, stage, *, expected=0, env=None, extra=()):
        result = subprocess.run(
            [
                sys.executable,
                str(self.script),
                stage,
                "--suite",
                "suite",
                "--policy",
                "policy",
                *extra,
            ],
            cwd=self.root / job,
            env=env or self.env,
            text=True,
            capture_output=True,
            timeout=40,
        )
        self.results.append(result)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertNotIn("synthetic-private-canary", result.stdout + result.stderr)
        return result

    def _copy(self, source, destination):
        for path in (self.root / source / "encrypted").glob("*.bin"):
            shutil.copyfile(path, self.root / destination / "incoming" / path.name)

    def _complete(self, *, failing=False):
        self._run("prepare", "prepare")
        for shard in "ABCD":
            self._copy("prepare", shard)
            self._run(shard, "restore")
            self._run(
                shard, "shard", expected=1 if failing else 0, extra=("--shard", shard)
            )
            self._copy(shard, "aggregate")
        self._copy("prepare", "aggregate")
        self._run("aggregate", "restore")
        self._run("aggregate", "aggregate", expected=1 if failing else 0)

    def test_isolated_jobs_verify_full_coverage_retries_and_one_warmup(self):
        """Drive the actual wrapper and encrypted artifact boundary end to end."""
        self._complete()
        result = json.loads(
            (self.root / "aggregate/evidence/acceptance.json").read_text()
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["retried"]), 1)
        self.assertTrue((self.root / "prepare/warmup-ran").exists())
        for shard in "ABCD":
            self.assertFalse((self.root / shard / "warmup-ran").exists())
        plan = json.loads((self.root / "aggregate/evidence/plan.json").read_text())
        self.assertTrue(all(plan["assignments"].values()))
        self.assertEqual(plan["context"]["identity"]["verification"], "not-performed")
        self.assertIn("5 cases across 4 shards", self.results[-1].stdout)

    def test_approved_fast_track_runs_the_same_four_shard_pipeline(self):
        """Accept the exact ref and SHA authorized by the release PR gate."""
        self.env.update(
            GITHUB_REF="refs/heads/fast-track/post-v1.8.0-pt1",
            HOSTED_E2E_APPROVED_REF="refs/heads/fast-track/post-v1.8.0-pt1",
        )
        self._complete()

    def test_failure_output_stays_encrypted_and_cannot_pass(self):
        """Exceptions containing secrets must not reach visible output/artifacts."""
        for job in ["prepare", *"ABCD"]:
            for name in "abcd":
                (self.root / job / f"suite/test_{name}.py").write_text(
                    "import os\ndef test_failure():\n"
                    '    value = os.environ["HOSTED_PLATFORM_TESTS_API_KEY"]\n'
                    "    raise RuntimeError(value)\n"
                )
        self._complete(failing=True)
        self.assertIn(
            "synthetic-private-canary", (self.root / "A/evidence/stage.log").read_text()
        )
        for artifact in self.root.glob("*/encrypted/*.bin"):
            data = artifact.read_bytes()
            self.assertNotIn(b"synthetic-private-canary", data)
            self.assertNotIn(b"test_failure", data)
        result = json.loads(
            (self.root / "aggregate/evidence/acceptance.json").read_text()
        )
        self.assertFalse(result["accepted"])

    def test_wrong_key_attempt_target_and_tampering_fail_closed(self):
        """Bind evidence to its secret, revision, target, run and attempt."""
        self._run("prepare", "prepare")
        self._copy("prepare", "A")
        for changes in [
            {"HOSTED_PLATFORM_TESTS_API_KEY": "different-key-12345678901234567890"},
            {"HOSTED_E2E_RUN_KEY": "123:2"},
            {
                "HOSTED_E2E_TARGET": "production",
                "HOSTED_PLATFORM_TESTS_PROJECT": "roboflow-platform-serverless",
            },
            {
                "GITHUB_SHA": "b" * 40,
                "HOSTED_E2E_TEST_SHA": "b" * 40,
                "HOSTED_E2E_RUNNER_SHA": "b" * 40,
            },
        ]:
            self._run("A", "restore", expected=1, env={**self.env, **changes})
            self.assertFalse((self.root / "A/evidence").exists())
        artifact = self.root / "A/incoming/prepare.bin"
        data = artifact.read_bytes()
        artifact.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        self._run("A", "restore", expected=1)

    def test_missing_shards_and_missing_evidence_cannot_pass(self):
        """An absent shard or preparation bundle fails aggregation."""
        self._run("aggregate", "restore", expected=1)
        self._run("prepare", "prepare")
        self._copy("prepare", "aggregate")
        self._run("aggregate", "restore")
        self._run("aggregate", "aggregate", expected=1)

    def test_caller_guard_rejects_untrusted_context_before_tests(self):
        """A fork, PR, non-default branch or mismatched target cannot run tests."""
        for changes in [
            {"GITHUB_REPOSITORY": "someone/inference"},
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"GITHUB_REF": "refs/heads/feature"},
            {"HOSTED_E2E_RUNNER_SHA": "b" * 40},
            {"HOSTED_E2E_TARGET": "other"},
            {"HOSTED_PLATFORM_TESTS_PROJECT": "roboflow-platform-serverless"},
        ]:
            self._run("prepare", "prepare", expected=1, env={**self.env, **changes})
            self.assertFalse((self.root / "prepare/warmup-ran").exists())

    def test_missing_provider_and_direct_uncaptured_execution_are_rejected(self):
        """Do not silently skip a missing provider or emit plaintext test output."""
        self._run("prepare", "prepare", expected=1, env={**self.env, "OPENAI_KEY": ""})
        self.assertFalse((self.root / "prepare/evidence/plan.json").exists())
        self._run("prepare", "prepare", expected=1, extra=("--execute-hidden",))

    def test_authenticated_bundles_cannot_write_paths_or_conflicting_evidence(self):
        """Reject archive traversal and conflicting reports before extracting."""
        import base64

        with patch.dict(os.environ, self.env):
            binding = repository._binding()
            cipher = repository._cipher(binding)
            incoming = self.root / "A/incoming"
            for name in ["../escaped", "/tmp/escaped"]:
                nonce = os.urandom(12)
                payload = json.dumps({name: base64.b64encode(b"unsafe").decode()})
                (incoming / "bad.bin").write_bytes(
                    b"HE2E1" + nonce + cipher.encrypt(nonce, payload.encode(), binding)
                )
                with self.assertRaises(ValueError):
                    repository._restore(incoming, output=self.root / "A/evidence")
                self.assertFalse((self.root / "A/evidence").exists())
            (incoming / "bad.bin").unlink()
            for index in range(2):
                nonce = os.urandom(12)
                payload = json.dumps(
                    {"plan.json": base64.b64encode(str(index).encode()).decode()}
                )
                (incoming / f"{index}.bin").write_bytes(
                    b"HE2E1" + nonce + cipher.encrypt(nonce, payload.encode(), binding)
                )
            with self.assertRaises(ValueError):
                repository._restore(incoming, output=self.root / "A/evidence")
            self.assertFalse((self.root / "A/evidence").exists())

    def test_explicit_skip_warmup_preserves_existing_manual_option(self):
        """The operator can skip service warmup without skipping any test case."""
        self.env["HOSTED_E2E_SKIP_WARMUP"] = "true"
        self._complete()
        self.assertFalse((self.root / "prepare/warmup-ran").exists())
        warmup = json.loads((self.root / "prepare/evidence/warmup.json").read_text())
        self.assertTrue(warmup["services_warmup_skipped"])


class WorkflowIntegrationTests(unittest.TestCase):
    """Verify that the operator's existing buttons reach all four shard jobs."""

    def test_both_manual_entrypoints_call_four_shard_workflow(self):
        """Catch the original disconnected implementation at the user entrypoint."""
        root = Path(__file__).resolve().parents[1] / "workflows"
        for target in ["staging", "production"]:
            data = yaml.safe_load(
                (root / f"hosted_inference_e2e_test_{target}.yml").read_text()
            )
            job = data["jobs"]["hosted-tests"]
            self.assertEqual(
                job["uses"], "./.github/workflows/hosted_inference_e2e_repository.yml"
            )
            self.assertEqual(job["with"]["target"], target)
        workflow = yaml.safe_load(
            (root / "hosted_inference_e2e_repository.yml").read_text()
        )
        jobs = workflow["jobs"]
        self.assertEqual(jobs["shards"]["strategy"]["matrix"]["shard"], list("ABCD"))
        self.assertEqual(jobs["aggregate"]["needs"], ["guard", "prepare", "shards"])
        self.assertIn("always()", jobs["aggregate"]["if"])
        self.assertEqual(jobs["prepare"]["needs"], "guard")
        self.assertEqual(jobs["shards"]["needs"], ["guard", "prepare"])
        self.assertEqual(
            jobs["guard"]["steps"][-1]["run"],
            "python3 .github/scripts/hosted_e2e_gate.py",
        )
        for name in ["prepare", "shards", "aggregate"]:
            self.assertEqual(
                jobs[name]["env"]["HOSTED_E2E_APPROVED_SHA"],
                "${{ needs.guard.outputs.approved_sha }}",
            )
        for job in jobs.values():
            for step in job["steps"]:
                if step.get("uses", "").startswith("actions/upload-artifact@"):
                    self.assertEqual(step["with"]["path"], "encrypted/*.bin")
                if step.get("uses", "").startswith("actions/checkout@"):
                    self.assertEqual(step["with"]["ref"], "${{ github.sha }}")
                    self.assertFalse(step["with"]["persist-credentials"])


if __name__ == "__main__":
    unittest.main()
