"""Offline security and coverage checks for the opt-in Depot transport."""

import ast
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import depot_ci_image as image
import yaml

ROOT = Path(__file__).resolve().parents[2]
SHA = "d" * 40
DIGEST = "sha256:" + "a" * 64
ARM = "sha256:" + "b" * 64
IMAGE_ID = "sha256:" + "c" * 64
REF = "refs/heads/fast-track/post-v1.8.0-pt1"
WORKFLOW = "roboflow/inference/.github/workflows/fast_track_depot_images.yml"
REPOSITORY = "example.registry.depot.dev/exampleproject"


class DepotImageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = {
            "GITHUB_REPOSITORY": "roboflow/inference",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": REF,
            "GITHUB_SHA": SHA,
            "GITHUB_WORKFLOW_REF": f"{WORKFLOW}@{REF}",
            "CI_DEPOT_REUSE_ENABLED": "true",
            "CI_DEPOT_APPROVED_REF": REF,
            "CI_DEPOT_APPROVED_SHA": SHA,
            "CI_DEPOT_REGISTRY": REPOSITORY,
            "DEPOT_PROJECT_ID": "exampleproject",
            "IMAGE_FAMILY": "cpu",
            "MODEL_MODE": "true",
            "IMAGE_DIGEST": DIGEST,
            "RUNNER_TEMP": str(self.root),
            "DOCKER_CONFIG": str(self.root / "builder"),
            "GITHUB_ENV": str(self.root / "env"),
            "GITHUB_OUTPUT": str(self.root / "output"),
        }
        (self.root / "builder").mkdir()
        self.environment = patch.dict(os.environ, self.env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []
        self.info = {
            "Id": IMAGE_ID,
            "Os": "linux",
            "Architecture": "amd64",
            "RepoDigests": [f"{REPOSITORY}@{DIGEST}"],
            "Config": {
                "Labels": {
                    "org.opencontainers.image.revision": SHA,
                    "io.roboflow.ci.image-family": "cpu",
                }
            },
        }
        self.manifest = {
            "manifests": [
                {"platform": {"os": "linux", "architecture": arch}, "digest": digest}
                for arch, digest in [("amd64", DIGEST), ("arm64", ARM)]
            ]
        }
        self.metadata = {
            "containerimage.digest": DIGEST,
            "depot.build": {"buildID": "examplebuild", "projectID": "exampleproject"},
        }

    def _run(self, args, *, env=None, stdin=None):
        self.calls.append((args, env, stdin))
        if args[:2] == ["git", "rev-parse"]:
            return SHA
        if args[:2] == ["depot", "pull-token"]:
            return "synthetic-pull-token\n"
        if args[:2] == ["docker", "login"]:
            Path(env["DOCKER_CONFIG"], "config.json").write_text(stdin)
        if args[:2] == ["depot", "build"]:
            Path(args[args.index("--metadata-file") + 1]).write_text(
                json.dumps(self.metadata)
            )
        if args[:3] == ["docker", "buildx", "imagetools"]:
            return json.dumps(self.manifest)
        if args[:3] == ["docker", "image", "inspect"]:
            return json.dumps([self.info])

        return ""

    def test_approval_rejects_untrusted_or_stale_execution_before_external_commands(
        self,
    ):
        for key, value in [
            ("GITHUB_REPOSITORY", "external/inference"),
            ("GITHUB_EVENT_NAME", "pull_request"),
            ("GITHUB_EVENT_NAME", "pull_request_target"),
            ("GITHUB_EVENT_NAME", "push"),
            ("CI_DEPOT_REUSE_ENABLED", "false"),
            ("CI_DEPOT_APPROVED_REF", ""),
            ("CI_DEPOT_APPROVED_SHA", ""),
            ("CI_DEPOT_APPROVED_SHA", "e" * 40),
            ("GITHUB_REF", "refs/heads/main"),
            ("GITHUB_REF", "refs/heads/feature"),
            ("GITHUB_WORKFLOW_REF", f"wrong/workflow@{REF}"),
            ("GITHUB_SHA", "d" * 39),
        ]:
            with self.subTest(key=key, value=value), patch.dict(
                os.environ, {key: value}
            ), patch.object(image, "_run") as run:
                with self.assertRaises(ValueError):
                    image._authorize()
                run.assert_not_called()
        with patch.object(image, "_run", return_value="e" * 40):
            with self.assertRaises(ValueError):
                image._authorize()

    def test_destination_and_digest_validation_precede_login(self):
        for key, value in [
            ("CI_DEPOT_REGISTRY", "docker.io/public/image"),
            ("CI_DEPOT_REGISTRY", "example.registry.depot.dev/project/nested"),
            ("CI_DEPOT_REGISTRY", "example.registry.depot.dev/project:tag"),
            ("IMAGE_DIGEST", "latest"),
            ("IMAGE_DIGEST", ""),
            ("MODEL_MODE", "all"),
            ("IMAGE_FAMILY", "parallel"),
        ]:
            self.calls.clear()
            with self.subTest(key=key), patch.dict(
                os.environ, {key: value}
            ), patch.object(image, "_run", side_effect=self._run):
                with self.assertRaises(ValueError):
                    image._pull()
                self.assertFalse(any(args[0] == "docker" for args, _, _ in self.calls))

    def test_pull_verifies_digest_source_family_platform_and_emits_only_local_id(self):
        with patch.object(image, "_run", side_effect=self._run):
            image._pull()
        self.assertEqual(
            (self.root / "env").read_text(), f"INFERENCE_TEST_IMAGE={IMAGE_ID}\n"
        )
        commands = [args for args, _, _ in self.calls]
        self.assertIn(
            ["docker", "pull", "--platform", "linux/amd64", f"{REPOSITORY}@{DIGEST}"],
            commands,
        )
        # OIDC mints a read-only pull token; the consumer never builds anything.
        self.assertEqual(
            [args for args in commands if args[0] == "depot"],
            [["depot", "pull-token", "--project", "exampleproject"]],
        )
        self.assertFalse(any("build" in args for args in commands))
        login = next(call for call in self.calls if call[0][:2] == ["docker", "login"])
        self.assertEqual(login[2], "synthetic-pull-token")
        self.assertNotIn(login[2], str(login[0]))
        self.assertFalse(Path(login[1]["DOCKER_CONFIG"]).exists())

    def test_wrong_pulled_identity_never_reaches_tests_and_credentials_are_removed(
        self,
    ):
        variants = [
            {"Os": "windows"},
            {"Architecture": "arm64"},
            {"RepoDigests": []},
            {
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.revision": "e" * 40,
                        "io.roboflow.ci.image-family": "cpu",
                    }
                }
            },
            {
                "Config": {
                    "Labels": {
                        "org.opencontainers.image.revision": SHA,
                        "io.roboflow.ci.image-family": "gpu",
                    }
                }
            },
            {"Config": {"Labels": None}},
        ]
        for change in variants:
            with self.subTest(change=change), patch.dict(
                self.info, change
            ), patch.object(image, "_run", side_effect=self._run):
                with self.assertRaises(ValueError):
                    image._pull()
                self.assertFalse((self.root / "env").exists())
                login = self.calls[-3]
                self.assertFalse(Path(login[1]["DOCKER_CONFIG"]).exists())

    def test_empty_or_malformed_pull_token_never_logs_in(self):
        for minted in ["", "\n", "two words\n"]:

            def run(args, **kwargs):
                result = self._run(args, **kwargs)
                return minted if args[:2] == ["depot", "pull-token"] else result

            self.calls.clear()
            with self.subTest(minted=minted), patch.object(
                image, "_run", side_effect=run
            ):
                with self.assertRaises(ValueError):
                    image._pull()
                self.assertEqual([call[0][0] for call in self.calls], ["git", "depot"])

    def test_build_saves_tracked_source_without_release_publication(self):
        for family in ["cpu", "gpu"]:
            with self.subTest(family=family), patch.dict(
                os.environ, {"IMAGE_FAMILY": family}
            ), patch.object(image, "_run", side_effect=self._run):
                self.calls.clear()
                if family == "gpu":
                    self.manifest = {}
                image._build()
                commands = [call[0] for call in self.calls]
                builds = [args for args in commands if args[:2] == ["depot", "build"]]
                self.assertEqual(len(builds), 1)
                build = builds[0]
                self.assertIn("--save", build)
                self.assertNotIn("--push", build)
                self.assertNotIn("--tag", build)
                self.assertIn("--provenance=mode=min", build)
                self.assertIn(f"org.opencontainers.image.revision={SHA}", build)
                self.assertIn(f"io.roboflow.ci.image-family={family}", build)
                self.assertEqual(
                    build[build.index("--platform") + 1],
                    "linux/amd64,linux/arm64" if family == "cpu" else "linux/amd64",
                )
                archive = next(
                    args for args in commands if args[:2] == ["git", "archive"]
                )
                self.assertEqual(archive[-1], SHA)
                self.assertNotEqual(build[-1], ".")
                output = (self.root / "output").read_text()
                self.assertIn(f"amd64_digest={DIGEST}", output)
                self.assertNotIn("example", output)
                self.assertFalse((self.root / "builder/config.json").exists())

    def test_bad_saved_metadata_and_incomplete_platforms_fail_closed(self):
        for metadata, manifest in [
            ({"containerimage.digest": DIGEST}, self.manifest),
            (
                {
                    **self.metadata,
                    "depot.build": {"buildID": "id", "projectID": "other"},
                },
                self.manifest,
            ),
            ({**self.metadata, "containerimage.digest": "bad"}, self.manifest),
            (self.metadata, {}),
            (self.metadata, {"manifests": self.manifest["manifests"][:1]}),
        ]:
            with self.subTest(metadata=metadata, manifest=manifest), patch.object(
                image, "_run", side_effect=self._run
            ):
                self.metadata, self.manifest = metadata, manifest
                with self.assertRaises((ValueError, KeyError)):
                    image._build()
                self.assertFalse((self.root / "output").exists())
                self.assertFalse((self.root / "builder/config.json").exists())

    def test_build_requires_matching_project_and_uses_a_minted_pull_token(self):
        for change in [{"DEPOT_PROJECT_ID": ""}, {"DEPOT_PROJECT_ID": "wrong"}]:
            self.calls.clear()
            with patch.dict(os.environ, change), patch.object(
                image, "_run", side_effect=self._run
            ):
                with self.assertRaises(ValueError):
                    image._build()
            self.assertFalse(any(args[0] == "depot" for args, _, _ in self.calls))
        self.calls.clear()
        with patch.object(image, "_run", side_effect=self._run):
            image._build()
        depot = [args for args, _, _ in self.calls if args[0] == "depot"]
        self.assertEqual(depot[0][:2], ["depot", "build"])
        self.assertEqual(
            depot[1], ["depot", "pull-token", "--project", "exampleproject"]
        )
        login = next(call for call in self.calls if call[0][:2] == ["docker", "login"])
        self.assertEqual(login[2], "synthetic-pull-token")
        self.assertNotIn("DEPOT_TOKEN", os.environ)

    def test_failure_removes_only_builder_credentials(self):
        (self.root / "builder/config.json").write_text("synthetic-secret")
        (self.root / "builder/buildx").mkdir()
        (self.root / "builder/buildx/state").write_text("keep")

        def failed(args, **kwargs):
            if args[0] == "depot":
                raise ValueError("synthetic-secret")
            return self._run(args, **kwargs)

        with patch.object(image, "_run", side_effect=failed):
            with self.assertRaises(ValueError):
                image._build()
        self.assertFalse((self.root / "builder/config.json").exists())
        self.assertEqual((self.root / "builder/buildx/state").read_text(), "keep")

    def test_real_subprocess_failures_do_not_disclose_output_or_credentials(self):
        # Real executable boundary: the fake Docker writes credential-bearing
        # output and config before failing. Nothing may reach public stdout.
        bindir = self.root / "bin"
        bindir.mkdir()
        git = bindir / "git"
        git.write_text(f"#!/bin/sh\nprintf '%s' '{SHA}'\n")
        depot = bindir / "depot"
        depot.write_text("#!/bin/sh\nprintf '%s' 'synthetic-pull-token'\n")
        docker = bindir / "docker"
        docker.write_text("""#!/bin/sh
cat > "$DOCKER_CONFIG/config.json"
echo "synthetic-private-host synthetic-pull-token"
echo "synthetic-private-host synthetic-pull-token" >&2
exit 17
""")
        for path in [git, depot, docker]:
            path.chmod(0o700)
        env = dict(os.environ, PATH=f"{bindir}:/usr/bin:/bin", CI_DEPOT_ACTION="pull")
        result = subprocess.run(
            [sys.executable, str(ROOT / ".github/scripts/depot_ci_image.py")],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        self.assertNotIn("synthetic", result.stdout)
        self.assertEqual(result.stdout.count("::error::"), 1)
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), ["bin", "builder"]
        )

    def test_private_test_shell_preserves_failure_and_environment_outputs(self):
        script = self.root / "script.sh"
        script.write_text(
            'echo "private-token"\necho "private-token" >&2\necho "TEST_PORT=1234" >> "$GITHUB_ENV"\nexit 17\n'
        )
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / ".github/scripts/ci_private_shell.py"),
                str(script),
            ],
            env=dict(os.environ, PATH="/usr/bin:/bin"),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 17)
        self.assertNotIn("private-token", result.stdout + result.stderr)
        self.assertEqual((self.root / "env").read_text(), "TEST_PORT=1234\n")


def _load(name):
    result = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    return result


class WorkflowTests(unittest.TestCase):
    def test_depot_selection_excludes_legacy_jobs_even_for_resume_inputs(self):
        workflow = _load("fast_track_integration_tests.yml")
        # YAML 1.1 parses GitHub's `on` key as True.
        self.assertFalse(
            workflow[True]["workflow_dispatch"]["inputs"]["use_depot_images"]["default"]
        )
        for name, job in workflow["jobs"].items():
            if name == "depot-trial":
                self.assertIn("&& inputs.use_depot_images", job["if"])
                self.assertEqual(
                    job["uses"], "./.github/workflows/fast_track_depot_images.yml"
                )
            else:
                self.assertIn("!inputs.use_depot_images", job["if"], name)
        nested = _load("fast_track_depot_images.yml")
        self.assertNotEqual(
            workflow["concurrency"]["group"], nested["concurrency"]["group"]
        )

    def test_oidc_auth_without_stored_depot_tokens_and_separated_permissions(self):
        # Org convention: GitHub OIDC through the org-level Depot connection,
        # no stored Depot token anywhere, project and registry as variables.
        oidc = {"contents": "read", "id-token": "write"}
        for name in [
            "fast_track_depot_images.yml",
            "depot_ci_image_build.yml",
            "depot_ci_image_tests.yml",
        ]:
            workflow = _load(name)
            self.assertEqual(workflow["permissions"], {"contents": "read"})
            self.assertNotRegex(json.dumps(workflow), r"(CI_)?DEPOT_[A-Z_]*TOKEN")
            self.assertNotIn("secrets.CI_DEPOT", json.dumps(workflow))
            for job_id, job in workflow["jobs"].items():
                self.assertNotEqual(job.get("secrets"), "inherit")
                if job_id in {"authorize", "complete"}:
                    self.assertNotIn("id-token", job.get("permissions", {}))
                else:
                    self.assertEqual(job["permissions"], oidc, f"{name}:{job_id}")
                for step in job.get("steps", []):
                    self.assertNotIn("${{ secrets.", step.get("run", ""))
                    action = step.get("uses", "")
                    self.assertNotIn("upload-artifact", action)
                    if action and not action.startswith("./"):
                        self.assertRegex(action, r"@[0-9a-f]{40}$")
        caller = _load("fast_track_integration_tests.yml")["jobs"]["depot-trial"]
        self.assertEqual(caller["permissions"], oidc)
        self.assertNotIn("CI_DEPOT", json.dumps(caller))
        tests = _load("depot_ci_image_tests.yml")
        build = _load("depot_ci_image_build.yml")["jobs"]["build"]
        self.assertEqual(build["environment"], "ci-depot-builder")
        for job in [build, *tests["jobs"].values()]:
            self.assertEqual(
                job["env"]["CI_DEPOT_REGISTRY"], "${{ vars.CI_DEPOT_REGISTRY }}"
            )
            self.assertEqual(
                job["env"]["DEPOT_PROJECT_ID"], "${{ vars.CI_DEPOT_PROJECT }}"
            )
            setup = [
                index
                for index, step in enumerate(job["steps"])
                if "depot/setup-action" in step.get("uses", "")
            ]
            self.assertEqual(len(setup), 1)
            depot_steps = [
                index
                for index, step in enumerate(job["steps"])
                if step.get("env", {}).get("CI_DEPOT_ACTION") in {"build", "pull"}
            ]
            self.assertEqual(len(depot_steps), 1)
            self.assertLess(setup[0], depot_steps[0])
            self.assertEqual(
                set(job["steps"][depot_steps[0]].get("env", {})), {"CI_DEPOT_ACTION"}
            )
        for job in tests["jobs"].values():
            self.assertEqual(job["environment"], "ci-depot-reader")

    def test_all_modes_and_shared_stages_are_required(self):
        tests = _load("depot_ci_image_tests.yml")
        for job in tests["jobs"].values():
            self.assertTrue(job["strategy"]["fail-fast"])
            self.assertEqual(job["strategy"]["matrix"]["model_mode"], ["true", "false"])
            for step in job["steps"]:
                if "server-tests-" in step.get("uses", ""):
                    self.assertEqual(step["with"]["private_logs"], "true")
                    if step["uses"].endswith("gpu"):
                        self.assertEqual(step["with"]["test_name"], "")
        gpu = yaml.safe_load(
            (ROOT / ".github/actions/server-tests-gpu/action.yml").read_text()
        )
        self.assertEqual(
            {step["id"] for step in gpu["runs"]["steps"] if "id" in step},
            {
                "regression_tests_without_torch",
                "regression_tests_with_torch",
                "paligemma_tests",
                "florence_tests",
                "sam2_tests",
                "sam3_tests",
                "moondream_tests",
            },
        )
        for family, filename in [
            ("cpu", "integration_tests_inference_server_x86.yml"),
            ("gpu", "test.nvidia_t4.yml"),
        ]:
            self.assertIn(
                f"./.github/actions/server-tests-{family}",
                [
                    step.get("uses")
                    for step in _load(filename)["jobs"]["build"]["steps"]
                ],
            )

    def test_job_guards_reject_external_callers_and_unapproved_revisions(self):
        context = {
            "github.repository": "roboflow/inference",
            "github.event_name": "workflow_dispatch",
            "github.ref": REF,
            "github.sha": SHA,
            "github.workflow_ref": f"{WORKFLOW}@{REF}",
            "vars.CI_DEPOT_REUSE_ENABLED": "true",
            "vars.CI_DEPOT_APPROVED_REF": REF,
            "vars.CI_DEPOT_APPROVED_SHA": SHA,
            "inputs.family": "cpu",
        }

        def evaluate(node):
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
                values = [evaluate(value) for value in node.values]
                return all(values) if isinstance(node.op, ast.And) else any(values)
            if (
                isinstance(node, ast.Compare)
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
            ):
                return evaluate(node.left) == evaluate(node.comparators[0])
            raise AssertionError("Unexpected guard syntax")

        def allows(condition, values):
            expression = condition[3:-2].replace(
                "startsWith(github.ref, 'refs/heads/fast-track/post-v')",
                repr(values["github.ref"].startswith("refs/heads/fast-track/post-v")),
            )
            expression = expression.replace(
                "format('roboflow/inference/.github/workflows/fast_track_depot_images.yml@{0}', github.ref)",
                repr(f'{WORKFLOW}@{values["github.ref"]}'),
            )
            expression = expression.replace(
                "format('roboflow/inference/.github/workflows/fast_track_integration_tests.yml@{0}', github.ref)",
                repr(
                    f'roboflow/inference/.github/workflows/fast_track_integration_tests.yml@{values["github.ref"]}'
                ),
            )
            expression = re.sub(
                r"\b(?:github|vars|inputs)\.[\w.]+",
                lambda match: repr(values[match.group()]),
                expression,
            )
            return evaluate(
                ast.parse(
                    expression.replace("&&", " and ").replace("||", " or ").strip(),
                    mode="eval",
                ).body
            )

        for name in ["depot_ci_image_build.yml", "depot_ci_image_tests.yml"]:
            for job_id, job in _load(name)["jobs"].items():
                values = dict(
                    context,
                    **{"inputs.family": job_id if job_id in {"cpu", "gpu"} else "cpu"},
                )
                self.assertTrue(allows(job["if"], values))
                for key, value in [
                    ("github.repository", "external/inference"),
                    ("github.event_name", "pull_request"),
                    ("github.event_name", "pull_request_target"),
                    ("github.event_name", "push"),
                    ("vars.CI_DEPOT_REUSE_ENABLED", "false"),
                    ("github.sha", "e" * 40),
                    ("github.ref", "refs/heads/main"),
                    ("github.workflow_ref", "another-workflow"),
                ]:
                    with self.subTest(name=name, job=job_id, key=key):
                        self.assertFalse(allows(job["if"], {**values, key: value}))

    def test_aggregate_rejects_skipped_cancelled_failed_or_wrong_image(self):
        script = _load("fast_track_depot_images.yml")["jobs"]["complete"]["steps"][0][
            "run"
        ]
        with TemporaryDirectory() as directory:
            env = {
                "PATH": "/usr/bin:/bin",
                "RESULTS": "success success success success success",
                "CPU_DIGEST": DIGEST,
                "CPU_ARM_DIGEST": ARM,
                "GPU_DIGEST": DIGEST,
                "CPU_TESTED": DIGEST,
                "GPU_TESTED": DIGEST,
                "GITHUB_STEP_SUMMARY": f"{directory}/summary",
            }
            self.assertEqual(
                subprocess.run(
                    ["bash", "-c", script], env=env, capture_output=True
                ).returncode,
                0,
            )
            for result in ["failure", "cancelled", "skipped", ""]:
                for index in range(5):
                    outcomes = ["success"] * 5
                    outcomes[index] = result
                    self.assertNotEqual(
                        subprocess.run(
                            ["bash", "-c", script],
                            env={**env, "RESULTS": " ".join(outcomes)},
                            capture_output=True,
                        ).returncode,
                        0,
                    )
            for key in [
                "CPU_DIGEST",
                "CPU_ARM_DIGEST",
                "GPU_DIGEST",
                "CPU_TESTED",
                "GPU_TESTED",
            ]:
                self.assertNotEqual(
                    subprocess.run(
                        ["bash", "-c", script],
                        env={**env, key: ""},
                        capture_output=True,
                    ).returncode,
                    0,
                )
