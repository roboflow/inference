"""Offline checks for immutable image transport and main workflow wiring."""

import ast
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_image", Path(__file__).with_name("ci_image.py")
)
IMAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMAGE)
DIGEST = "sha256:" + "a" * 64
ARM_DIGEST = "sha256:" + "b" * 64
IMAGE_ID = "sha256:" + "c" * 64
SHA = "d" * 40
PREFIX = "test-docker.pkg.dev/example/private-ci"


class ImageTransportTests(unittest.TestCase):
    """Exercise validation and exact Docker/Depot command construction offline."""

    def setUp(self):
        """Create a private temporary output directory and synthetic configuration."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = {
            "GITHUB_REPOSITORY": "roboflow/inference",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_SHA": SHA,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "CI_IMAGE_REUSE_ENABLED": "true",
            "CI_REGISTRY_PREFIX": PREFIX,
            "IMAGE_FAMILY": "cpu",
            "MODEL_MODE": "true",
            "PRIVATE_IMAGE": "true",
            "IMAGE_DIGEST": DIGEST,
            "IMAGE_TAG": "",
            "DEPOT_PROJECT_ID": "example-builder",
            "RUNNER_TEMP": str(self.root),
            "GITHUB_ENV": str(self.root / "env"),
            "GITHUB_OUTPUT": str(self.root / "output"),
        }
        self.environment = patch.dict(os.environ, self.env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []
        self.info = {
            "Id": IMAGE_ID,
            "Os": "linux",
            "Architecture": "amd64",
            "RepoDigests": [f"{PREFIX}/inference-ci-cpu@{DIGEST}"],
            "Config": {"Labels": {"org.opencontainers.image.revision": SHA}},
        }

    def _run(self, args, **kwargs):
        self.calls.append(args)
        if args[:2] == ["git", "rev-parse"]:
            return SHA
        if args[:3] == ["docker", "image", "inspect"]:
            return json.dumps([self.info])

        return ""

    def test_private_pull_uses_exact_digest_and_immutable_local_id(self):
        """Both families consume a verified digest without build/tag/push commands."""
        for family in ("cpu", "gpu"):
            with self.subTest(family=family), patch.object(
                IMAGE, "_run", side_effect=self._run
            ):
                os.environ["IMAGE_FAMILY"] = family
                reference = f"{PREFIX}/inference-ci-{family}@{DIGEST}"
                self.info["RepoDigests"] = [reference]
                IMAGE._pull()
                self.assertIn(
                    ["docker", "pull", "--platform", "linux/amd64", reference],
                    self.calls,
                )
                self.assertIn(
                    f"INFERENCE_TEST_IMAGE={IMAGE_ID}", (self.root / "env").read_text()
                )
                self.assertFalse(
                    any(
                        command[1] in {"build", "tag", "push"} for command in self.calls
                    )
                )

    def test_invalid_input_never_reaches_docker(self):
        """Malformed and conflicting references fail before pull or build."""
        cases = [
            {"IMAGE_DIGEST": "sha256:bad"},
            {"MODEL_MODE": "unexpected"},
            {"IMAGE_TAG": "release", "IMAGE_DIGEST": DIGEST},
            {"IMAGE_DIGEST": ""},
            {
                "IMAGE_DIGEST": "",
                "PRIVATE_IMAGE": "false",
                "IMAGE_TAG": "x\nINJECT=value",
            },
            {"CI_REGISTRY_PREFIX": "docker.io/public"},
        ]
        for values in cases:
            with self.subTest(values=values), patch.dict(
                os.environ, values
            ), patch.object(IMAGE, "_run", side_effect=self._run):
                self.calls.clear()
                with self.assertRaises(ValueError):
                    IMAGE._pull()
                self.assertFalse(any(command[0] == "docker" for command in self.calls))

    def test_untrusted_contexts_are_rejected(self):
        """Forks, PR events, non-main refs and disabled rollout cannot use transport."""
        for key, value in [
            ("GITHUB_REPOSITORY", "external/inference"),
            ("GITHUB_REF", "refs/heads/feature"),
            ("GITHUB_EVENT_NAME", "pull_request_target"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("CI_IMAGE_REUSE_ENABLED", "false"),
        ]:
            with self.subTest(key=key, value=value), patch.dict(
                os.environ, {key: value}
            ), patch.object(IMAGE, "_run") as run:
                for validate in (IMAGE._preflight, IMAGE._validate):
                    with self.assertRaises(ValueError):
                        validate()
                run.assert_not_called()

    def test_wrong_checkout_fails(self):
        """Source checkout must match the frozen workflow SHA."""
        with patch.object(IMAGE, "_run", return_value="e" * 40):
            with self.assertRaises(ValueError):
                IMAGE._preflight()

    def test_wrong_image_identity_fails(self):
        """Reject mismatched architecture, digest, config ID or source label."""
        for changes in [
            {"Architecture": "arm64"},
            {"Os": "windows"},
            {"RepoDigests": []},
            {"Config": {"Labels": {}}},
            {"Id": "not-an-image-id"},
        ]:
            with self.subTest(changes=changes), patch.dict(
                self.info, changes
            ), patch.object(IMAGE, "_run", side_effect=self._run):
                with self.assertRaises(ValueError):
                    IMAGE._pull()
                self.assertFalse((self.root / "env").exists())

    def test_pull_failure_does_not_rebuild(self):
        """Transport failure cannot silently change the artifact under test."""

        def fail_pull(args, **kwargs):
            if args[:2] == ["docker", "pull"]:
                raise ValueError("private registry details")

            return self._run(args, **kwargs)

        with patch.object(IMAGE, "_run", side_effect=fail_pull):
            with self.assertRaises(ValueError):
                IMAGE._pull()
        self.assertFalse((self.root / "env").exists())
        self.assertFalse(any(command[1] == "build" for command in self.calls))

    def test_legacy_tag_and_empty_defaults(self):
        """Legacy manual and fast-track inputs retain their non-private behavior."""
        with patch.dict(
            os.environ, {"PRIVATE_IMAGE": "false", "IMAGE_DIGEST": ""}
        ), patch.object(IMAGE, "_run", side_effect=self._run):
            IMAGE._validate()
            os.environ["IMAGE_TAG"] = "1.2.3-rc1"
            IMAGE._pull()
        self.assertIn(
            [
                "docker",
                "pull",
                "--platform",
                "linux/amd64",
                "roboflow/roboflow-inference-server-cpu:1.2.3-rc1",
            ],
            self.calls,
        )

    def test_private_command_failures_do_not_print_output(self):
        """Subprocess exceptions do not expose credentials, command arguments or logs."""
        result = subprocess.CompletedProcess(
            [], 1, "sensitive stdout", "sensitive stderr"
        )
        buffer = io.StringIO()
        with patch.object(
            subprocess, "run", return_value=result
        ), contextlib.redirect_stdout(buffer):
            with self.assertRaisesRegex(ValueError, "private output withheld") as error:
                IMAGE._run(["docker", "private-argument"])
        self.assertEqual(buffer.getvalue(), "")
        self.assertNotIn("sensitive", str(error.exception))
        self.assertNotIn("private-argument", str(error.exception))

    def test_manifest_platforms(self):
        """Keep CPU arm64 distinct from the amd64 image used by server tests."""
        amd = {"digest": DIGEST, "platform": {"os": "linux", "architecture": "amd64"}}
        arm = {
            "digest": ARM_DIGEST,
            "platform": {"os": "linux", "architecture": "arm64"},
        }
        attestation = {"platform": {"os": "unknown", "architecture": "unknown"}}
        self.assertEqual(
            IMAGE._manifest_digests(
                {"manifests": [amd, arm, attestation]}, family="cpu"
            ),
            {"amd64": DIGEST, "arm64": ARM_DIGEST},
        )
        for entries in ([amd], [amd, amd, arm], [arm]):
            with self.assertRaises(ValueError):
                IMAGE._manifest_digests({"manifests": entries}, family="cpu")

    def test_build_once_from_tracked_source_and_extract_children(self):
        """One multi-platform producer builds from an archive outside credential paths."""
        manifests = {
            "manifests": [
                {
                    "digest": DIGEST,
                    "platform": {"os": "linux", "architecture": "amd64"},
                },
                {
                    "digest": ARM_DIGEST,
                    "platform": {"os": "linux", "architecture": "arm64"},
                },
            ]
        }

        def producer(args, **kwargs):
            if args[:2] == ["depot", "build"]:
                Path(args[args.index("--metadata-file") + 1]).write_text(
                    json.dumps({"containerimage.digest": IMAGE_ID})
                )
            if args[:4] == ["docker", "buildx", "imagetools", "inspect"]:
                self.calls.append(args)
                return json.dumps(manifests)

            return self._run(args, **kwargs)

        with patch.object(IMAGE, "_run", side_effect=producer):
            IMAGE._build()
        builds = [args for args in self.calls if args[:2] == ["depot", "build"]]
        self.assertEqual(len(builds), 1)
        self.assertIn("linux/amd64,linux/arm64", builds[0])
        self.assertNotEqual(builds[0][-1], ".")
        self.assertIn(f"{PREFIX}/inference-ci-cpu:ci-{SHA}-123-1", builds[0])
        self.assertTrue(any(args[:2] == ["git", "archive"] for args in self.calls))
        output = (self.root / "output").read_text()
        self.assertIn(f"amd64_digest={DIGEST}", output)
        self.assertIn(f"arm64_digest={ARM_DIGEST}", output)
        self.assertNotIn(PREFIX, output)

    def test_dynamic_port_is_loopback_only(self):
        """Do not expose test servers on a public runner interface."""
        os.environ["INFERENCE_TEST_CONTAINER"] = "test-container"
        with patch.object(IMAGE, "_run", return_value="127.0.0.1:43210\n"):
            IMAGE._port()
        self.assertIn("TEST_PORT=43210", (self.root / "env").read_text())
        with patch.object(IMAGE, "_run", return_value="0.0.0.0:43210\n"):
            with self.assertRaises(ValueError):
                IMAGE._port()


class WorkflowContractTests(unittest.TestCase):
    """Check the security boundary, coverage and completion gate as a whole."""

    def test_credential_cleanup_preserves_buildx_post_job_state(self):
        """Execute cleanup steps: credentials vanish but builder records survive."""
        for name in (
            "ci_image_build.yml",
            "integration_tests_inference_server_x86.yml",
            "test.nvidia_t4.yml",
        ):
            workflow = self._workflow(name)
            steps = workflow["jobs"]["build"]["steps"]
            cleanup_steps = [
                step for step in steps if step.get("name", "").startswith("Remove")
            ]
            self.assertTrue(cleanup_steps)
            for step in cleanup_steps:
                with self.subTest(workflow=name, step=step["name"]):
                    with TemporaryDirectory() as directory:
                        root = Path(directory)
                        config = root / "docker-config"
                        state = config / "buildx/instances/test-builder"
                        state.parent.mkdir(parents=True)
                        state.write_text("builder cleanup requires this record")
                        credentials = config / "config.json"
                        credentials.write_text('{"auths":{"example":"test-only"}}')
                        binary = root / "bin"
                        binary.mkdir()
                        docker = binary / "docker"
                        docker.write_text("#!/bin/sh\nexit 0\n")
                        docker.chmod(0o755)
                        result = subprocess.run(
                            ["bash", "-e", "-c", step["run"]],
                            env={
                                **os.environ,
                                "PATH": f"{binary}:{os.environ['PATH']}",
                                "DOCKER_CONFIG": str(config),
                                "INFERENCE_TEST_CONTAINER": "test-container",
                            },
                            capture_output=True,
                        )
                        self.assertEqual(result.returncode, 0)
                        self.assertFalse(credentials.exists())
                        self.assertTrue(state.exists())
                        self.assertEqual(
                            state.read_text(), "builder cleanup requires this record"
                        )

    def test_ci_runs_the_image_reuse_regression_suite(self):
        """Keep these checks in PR CI when other script suites are split out."""
        workflow = self._workflow("validate.github-actions.yml")
        commands = [
            step.get("run", "")
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
        ]
        self.assertTrue(
            any(
                "unittest discover -s .github/scripts -p 'test_ci_image.py'" in command
                for command in commands
            )
        )

    def test_external_actions_are_pinned_to_commit_shas(self):
        """Prevent mutable action references in the image-reuse workflow surface."""
        for name in (
            "ci_image_build.yml",
            "ci_image_reuse.yml",
            "docker.cpu.yml",
            "docker.gpu.yml",
            "fast_track_integration_tests.yml",
            "integration_tests_inference_server_x86.yml",
            "test.nvidia_t4.yml",
        ):
            workflow = self._workflow(name)
            for job in workflow["jobs"].values():
                for step in job.get("steps", []):
                    action = step.get("uses", "")
                    if action and not action.startswith("./"):
                        with self.subTest(workflow=name, action=action):
                            self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$")

    def _workflow(self, name):
        result = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        return result

    def _job_is_allowed(self, name, job, **overrides):
        # Evaluate the boolean/comparison-only subset used by these job guards.
        context = {
            "github.repository": "roboflow/inference",
            "github.ref": "refs/heads/main",
            "github.event_name": "push",
            "github.event.act": False,
            "vars.CI_IMAGE_REUSE_ENABLED": "false",
            "inputs.private_image": False,
        }
        context.update(overrides)
        expression = self._workflow(name)["jobs"][job]["if"][3:-2].strip()
        expression = re.sub(
            r"\b(?:github|vars|inputs)\.[\w.]+",
            lambda match: repr(context[match.group()]),
            expression,
        )
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression).strip()
        allowed = self._evaluate_condition(
            ast.parse(f"({expression})", mode="eval").body
        )
        return allowed

    def _evaluate_condition(self, node):
        # Deliberately reject calls, attribute access and untested syntax.
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [self._evaluate_condition(value) for value in node.values]
            result = all(values) if isinstance(node.op, ast.And) else any(values)
            return result

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            result = not self._evaluate_condition(node.operand)
            return result

        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = self._evaluate_condition(node.left)
            right = self._evaluate_condition(node.comparators[0])
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right

        raise AssertionError(f"Unsupported job-guard syntax: {type(node).__name__}")

    def test_image_jobs_reject_external_callers_and_pr_events(self):
        """Foreign callers cannot bypass the job guard by choosing legacy inputs."""
        jobs = [
            ("docker.cpu.yml", "docker"),
            ("docker.gpu.yml", "docker"),
            ("test.nvidia_t4.yml", "build"),
            ("integration_tests_inference_server_x86.yml", "build"),
            ("ci_image_build.yml", "build"),
            ("ci_image_reuse.yml", "cpu-image"),
            ("ci_image_reuse.yml", "gpu-image"),
        ]
        for name, job in jobs:
            for private in (False, True):
                for flag in ("false", "true"):
                    for event in ("push", "workflow_dispatch", "pull_request_target"):
                        with self.subTest(
                            name=name, private=private, flag=flag, event=event
                        ):
                            self.assertFalse(
                                self._job_is_allowed(
                                    name,
                                    job,
                                    **{
                                        "github.repository": "external/inference",
                                        "github.event_name": event,
                                        "inputs.private_image": private,
                                        "vars.CI_IMAGE_REUSE_ENABLED": flag,
                                    },
                                )
                            )
            for event in ("pull_request", "pull_request_target"):
                with self.subTest(name=name, event=event):
                    self.assertFalse(
                        self._job_is_allowed(
                            name,
                            job,
                            **{
                                "github.event_name": event,
                                "inputs.private_image": True,
                                "vars.CI_IMAGE_REUSE_ENABLED": "true",
                            },
                        )
                    )

    def test_expected_image_entry_points_remain_available(self):
        """Preserve legacy/manual runs and trusted main reuse without PR access."""
        for name in (
            "test.nvidia_t4.yml",
            "integration_tests_inference_server_x86.yml",
        ):
            for event, ref in (
                ("push", "refs/heads/main"),
                ("workflow_dispatch", "refs/heads/fast-track/example"),
            ):
                self.assertTrue(
                    self._job_is_allowed(
                        name,
                        "build",
                        **{
                            "github.event_name": event,
                            "github.ref": ref,
                        },
                    )
                )
            self.assertTrue(
                self._job_is_allowed(
                    name,
                    "build",
                    **{
                        "inputs.private_image": True,
                        "vars.CI_IMAGE_REUSE_ENABLED": "true",
                    },
                )
            )
            self.assertFalse(
                self._job_is_allowed(
                    name,
                    "build",
                    **{
                        "inputs.private_image": True,
                        "github.ref": "refs/heads/feature",
                    },
                )
            )
            self.assertIs(
                self._workflow(name)["jobs"]["build"]["strategy"]["fail-fast"], True
            )
        for name in ("docker.cpu.yml", "docker.gpu.yml"):
            for event in ("push", "workflow_dispatch", "release"):
                self.assertTrue(
                    self._job_is_allowed(name, "docker", **{"github.event_name": event})
                )
        for repository, expected in (
            ("roboflow/inference", True),
            ("external/inference", False),
        ):
            self.assertEqual(
                self._job_is_allowed(
                    "fast_track_integration_tests.yml",
                    "resolve",
                    **{
                        "github.repository": repository,
                        "github.event_name": "workflow_dispatch",
                    },
                ),
                expected,
            )

    def test_main_uses_all_modes_and_full_gpu_suite(self):
        """The orchestrator must not substitute the reduced fast-track preset."""
        main = self._workflow("ci_image_reuse.yml")
        for family in ("cpu", "gpu"):
            job = main["jobs"][f"{family}-tests"]
            self.assertEqual(job["with"]["use-inference-models"], "all")
            self.assertTrue(job["with"]["private_image"])
            self.assertEqual(job["needs"], f"{family}-image")
            self.assertIn("amd64_digest", job["with"]["image_digest"])
            self.assertNotIn("CI_REGISTRY_WRITER", job["secrets"])
            self.assertNotIn("DOCKERHUB_TOKEN", job["secrets"])
        self.assertEqual(main["jobs"]["gpu-tests"]["with"]["test_name"], "")
        gpu = self._workflow("test.nvidia_t4.yml")["jobs"]["build"]
        ids = {step.get("id") for step in gpu["steps"]}
        self.assertTrue(
            {
                "regression_tests_without_torch",
                "regression_tests_with_torch",
                "paligemma_tests",
                "florence_tests",
                "sam2_tests",
                "sam3_tests",
                "moondream_tests",
            }
            <= ids
        )
        self.assertTrue(gpu["strategy"]["fail-fast"])

    def test_rollout_switch_disables_old_build_paths_together(self):
        """All four independent main jobs share the same rollout switch."""
        for name, job in [
            ("docker.cpu.yml", "docker"),
            ("docker.gpu.yml", "docker"),
            ("test.nvidia_t4.yml", "build"),
            ("integration_tests_inference_server_x86.yml", "build"),
        ]:
            condition = self._workflow(name)["jobs"][job]["if"]
            self.assertIn("vars.CI_IMAGE_REUSE_ENABLED != 'true'", condition)
        for name in (
            "test.nvidia_t4.yml",
            "integration_tests_inference_server_x86.yml",
        ):
            workflow = self._workflow(name)
            for step in workflow["jobs"]["build"]["steps"]:
                if "docker build " in step.get("run", ""):
                    self.assertIn("inputs.image_digest == ''", step["if"])
                    self.assertIn("!inputs.private_image", step["if"])

    def test_no_public_artifacts_or_raw_secret_shell_interpolation(self):
        """The transport never uploads private records or embeds secrets in scripts."""
        for name in (
            "ci_image_build.yml",
            "ci_image_reuse.yml",
            "test.nvidia_t4.yml",
            "integration_tests_inference_server_x86.yml",
        ):
            workflow = self._workflow(name)
            for job in workflow["jobs"].values():
                for step in job.get("steps", []):
                    self.assertNotIn("upload-artifact", step.get("uses", ""))
                    self.assertNotIn("${{ secrets.", step.get("run", ""))

    def test_gate_fails_for_each_incomplete_outcome(self):
        """Failed, skipped, cancelled or missing work cannot pass the aggregate gate."""
        gate = self._workflow("ci_image_reuse.yml")["jobs"]["image-tests-complete"]
        script = gate["steps"][0]["run"]
        with TemporaryDirectory() as directory:
            env = {
                **os.environ,
                "GITHUB_STEP_SUMMARY": str(Path(directory) / "summary"),
                "CPU_BUILD": "success",
                "GPU_BUILD": "success",
                "CPU_TESTS": "success",
                "GPU_TESTS": "success",
                "CPU_DIGEST": DIGEST,
                "CPU_ARM64_DIGEST": ARM_DIGEST,
                "GPU_DIGEST": DIGEST,
            }
            self.assertEqual(
                subprocess.run(
                    ["bash", "-e", "-c", script], env=env, capture_output=True
                ).returncode,
                0,
            )
            for key in ("CPU_BUILD", "GPU_BUILD", "CPU_TESTS", "GPU_TESTS"):
                for outcome in ("failure", "skipped", "cancelled", ""):
                    with self.subTest(key=key, outcome=outcome):
                        result = subprocess.run(
                            ["bash", "-e", "-c", script],
                            env={**env, key: outcome},
                            capture_output=True,
                        )
                        self.assertNotEqual(result.returncode, 0)
            result = subprocess.run(
                ["bash", "-e", "-c", script],
                env={**env, "CPU_DIGEST": ""},
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
