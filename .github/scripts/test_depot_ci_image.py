"""Offline security and coverage checks for the Depot Registry CI image transport."""

import io
import json
import os
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
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
INDEX = "sha256:" + "e" * 64
REF = "refs/heads/main"
ORCHESTRATOR = "roboflow/inference/.github/workflows/ci_image_reuse.yml"
REPOSITORY = "registry.depot.dev/exampleproject"
CONSUMERS = ("integration_tests_inference_server_x86.yml", "test.nvidia_t4.yml")


def _load(name):
    result = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
    return result


class DepotImageTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = {
            "GITHUB_REPOSITORY": "roboflow/inference",
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": REF,
            "GITHUB_SHA": SHA,
            "GITHUB_WORKFLOW_REF": f"{ORCHESTRATOR}@{REF}",
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

    def test_authorization_trusts_main_pushes_and_manual_branch_dispatches_only(self):
        accepted = [
            {},
            {
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/feature",
                "GITHUB_WORKFLOW_REF": f"{ORCHESTRATOR}@refs/heads/feature",
            },
        ]
        for changed in accepted:
            with self.subTest(accepted=changed), patch.dict(
                os.environ, changed
            ), patch.object(image, "_run", return_value=SHA):
                image._authorize()
        rejected = [
            {"GITHUB_REF": "refs/heads/feature"},
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"GITHUB_EVENT_NAME": "pull_request_target"},
            {"GITHUB_REPOSITORY": "external/inference"},
            {"GITHUB_WORKFLOW_REF": f"{ORCHESTRATOR}@refs/heads/feature"},
            {
                "GITHUB_WORKFLOW_REF": (
                    "roboflow/inference/.github/workflows/"
                    f"fast_track_integration_tests.yml@{REF}"
                )
            },
            {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": "refs/tags/v1"},
            {"GITHUB_SHA": "d" * 39},
        ]
        for changed in rejected:
            with self.subTest(rejected=changed), patch.dict(
                os.environ, changed
            ), patch.object(image, "_run") as run, self.assertRaises(ValueError):
                image._authorize()
            run.assert_not_called()
        with patch.object(image, "_run", return_value="e" * 40):
            with self.assertRaises(ValueError):
                image._authorize()

    def test_protected_projects_are_rejected_unless_pushed_to_main(self):
        manual = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/feature",
            "GITHUB_WORKFLOW_REF": f"{ORCHESTRATOR}@refs/heads/feature",
        }
        for project in ["9x3fbbxlnm", "grl7ffzxd7"]:
            self.calls.clear()
            with self.subTest(project=project), patch.dict(
                os.environ, {**manual, "DEPOT_PROJECT_ID": project}
            ), patch.object(image, "_run", side_effect=self._run):
                with self.assertRaises(ValueError):
                    image._validate()
                self.assertFalse(any(args[0] == "depot" for args, _, _ in self.calls))
        with patch.dict(
            os.environ, {**manual, "DEPOT_PROJECT_ID": "7m01vtml7h"}
        ), patch.object(image, "_run", side_effect=self._run):
            image._validate()
        with patch.dict(os.environ, {"DEPOT_PROJECT_ID": "9x3fbbxlnm"}), patch.object(
            image, "_run", side_effect=self._run
        ):
            image._validate()

    def test_specialized_workflows_allow_manual_sandbox_runs_but_not_direct_pushes(
        self,
    ):
        for filename in ["test.colab.gpu.yml", "test.nvidia_t4_parallel_server.yml"]:
            workflow = f"roboflow/inference/.github/workflows/{filename}@{REF}"
            with self.subTest(workflow=workflow), patch.dict(
                os.environ,
                {
                    "GITHUB_WORKFLOW_REF": workflow,
                    "GITHUB_EVENT_NAME": "workflow_dispatch",
                },
            ), patch.object(image, "_run", side_effect=self._run):
                image._validate()
                with patch.dict(os.environ, {"DEPOT_PROJECT_ID": "9x3fbbxlnm"}):
                    with self.assertRaises(ValueError):
                        image._validate()
                with patch.dict(os.environ, {"GITHUB_EVENT_NAME": "push"}):
                    with self.assertRaises(ValueError):
                        image._authorize()

    def test_parent_digest_is_required_only_for_the_parallel_test_image(self):
        cases = [
            {"IMAGE_FAMILY": "parallel-tests", "SERVER_IMAGE_DIGEST": value}
            for value in ["", "latest", "sha256:bad", f"{REPOSITORY}@{DIGEST}"]
        ] + [
            {"IMAGE_FAMILY": family, "SERVER_IMAGE_DIGEST": DIGEST}
            for family in ["cpu", "gpu", "colab-gpu", "parallel-gpu"]
        ]
        for env in cases:
            self.calls.clear()
            with self.subTest(env=env), patch.dict(os.environ, env), patch.object(
                image, "_run", side_effect=self._run
            ), self.assertRaises(ValueError):
                image._build()
            self.assertFalse(
                any(args[0] in {"depot", "docker"} for args, _, _ in self.calls)
            )

    def test_specialized_consumers_verify_identity_and_keep_separate_local_ids(self):
        for family in ["colab-gpu", "parallel-gpu", "parallel-tests"]:
            labels = self.info["Config"]["Labels"]
            labels["io.roboflow.ci.image-family"] = family
            labels["io.roboflow.ci.server-image-digest"] = INDEX
            output = (
                "INFERENCE_PARALLEL_TEST_IMAGE"
                if family == "parallel-tests"
                else "INFERENCE_TEST_IMAGE"
            )
            with self.subTest(family=family), patch.dict(
                os.environ,
                {
                    "IMAGE_FAMILY": family,
                    "MODEL_MODE": "",
                    "SERVER_IMAGE_DIGEST": INDEX if family == "parallel-tests" else "",
                },
            ), patch.object(image, "_run", side_effect=self._run):
                (self.root / "env").unlink(missing_ok=True)
                image._pull()
                self.assertEqual(
                    (self.root / "env").read_text(), f"{output}={IMAGE_ID}\n"
                )
                (self.root / "env").unlink()
                labels["io.roboflow.ci.image-family"] = "gpu"
                with self.assertRaises(ValueError):
                    image._pull()
                self.assertFalse((self.root / "env").exists())
                labels["io.roboflow.ci.image-family"] = family
                if family == "parallel-tests":
                    for bad_parent in [None, DIGEST]:
                        labels["io.roboflow.ci.server-image-digest"] = bad_parent
                        with self.assertRaises(ValueError):
                            image._pull()
                        self.assertFalse((self.root / "env").exists())
                        login = self.calls[-3]
                        self.assertFalse(Path(login[1]["DOCKER_CONFIG"]).exists())

    def test_destination_and_digest_validation_precede_login(self):
        for key, value in [
            ("DEPOT_PROJECT_ID", ""),
            ("DEPOT_PROJECT_ID", "Example/Project"),
            ("DEPOT_PROJECT_ID", "project:tag"),
            ("DEPOT_PROJECT_ID", "registry.depot.dev/project"),
            ("IMAGE_DIGEST", "latest"),
            ("IMAGE_DIGEST", ""),
            ("MODEL_MODE", "all"),
            ("IMAGE_FAMILY", "parallel"),
        ]:
            self.calls.clear()
            with self.subTest(key=key, value=value), patch.dict(
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
        self.assertEqual(login[0][2], "registry.depot.dev")
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
        amd = {"platform": {"os": "linux", "architecture": "amd64"}, "digest": DIGEST}
        arm = {"platform": {"os": "linux", "architecture": "arm64"}, "digest": ARM}
        cases = [
            (
                "cpu",
                {"manifests": [amd, arm]},
                {"amd64_digest": DIGEST, "arm64_digest": ARM},
            ),
            ("gpu", {"manifests": [amd]}, {"amd64_digest": DIGEST}),
            ("gpu", {"schemaVersion": 2}, {"amd64_digest": INDEX}),
            ("colab-gpu", {"manifests": [amd]}, {"amd64_digest": DIGEST}),
            ("parallel-gpu", {"schemaVersion": 2}, {"amd64_digest": INDEX}),
            ("parallel-tests", {"manifests": [amd]}, {"amd64_digest": DIGEST}),
        ]
        for family, manifest, children in cases:
            with self.subTest(
                family=family, indexed="manifests" in manifest
            ), patch.dict(
                os.environ,
                {
                    "IMAGE_FAMILY": family,
                    "SERVER_IMAGE_DIGEST": DIGEST if family == "parallel-tests" else "",
                },
            ), patch.object(
                image, "_run", side_effect=self._run
            ):
                self.calls.clear()
                (self.root / "output").unlink(missing_ok=True)
                self.manifest = manifest
                self.metadata = {**self.metadata, "containerimage.digest": INDEX}
                image._build()
                commands = [call[0] for call in self.calls]
                builds = [args for args in commands if args[:2] == ["depot", "build"]]
                self.assertEqual(len(builds), 1)
                build = builds[0]
                self.assertIn("--save", build)
                self.assertNotIn("--push", build)
                self.assertNotIn("--tag", build)
                self.assertIn("--provenance=mode=min", build)
                if family in {"colab-gpu", "parallel-gpu", "parallel-tests"}:
                    self.assertEqual(
                        build[build.index("--output") + 1],
                        "type=image,oci-mediatypes=true,compression=zstd,"
                        "compression-level=3,force-compression=true",
                    )
                else:
                    self.assertNotIn("--output", build)
                self.assertEqual(build[build.index("--project") + 1], "exampleproject")
                self.assertEqual(
                    build[build.index("--save-tag") + 1], f"ci-{family}-{SHA}"
                )
                self.assertIn(f"org.opencontainers.image.revision={SHA}", build)
                self.assertIn(f"io.roboflow.ci.image-family={family}", build)
                dockerfile = {
                    "cpu": "Dockerfile.onnx.cpu",
                    "gpu": "Dockerfile.onnx.gpu",
                    "colab-gpu": "Dockerfile.google.colab.gpu",
                    "parallel-gpu": "Dockerfile.onnx.gpu.parallel",
                    "parallel-tests": "Dockerfile.parallel.tests",
                }[family]
                self.assertEqual(
                    Path(build[build.index("--file") + 1]).name, dockerfile
                )
                if family == "parallel-tests":
                    self.assertEqual(
                        build[build.index("--build-arg") + 1],
                        f"SERVER_IMAGE={REPOSITORY}@{DIGEST}",
                    )
                    self.assertIn(f"io.roboflow.ci.server-image-digest={DIGEST}", build)
                    login_index = next(
                        i
                        for i, args in enumerate(commands)
                        if args[:2] == ["docker", "login"]
                    )
                    self.assertLess(login_index, commands.index(build))
                else:
                    self.assertNotIn("--build-arg", build)
                self.assertEqual(
                    build[build.index("--platform") + 1],
                    "linux/amd64,linux/arm64" if family == "cpu" else "linux/amd64",
                )
                archive = next(
                    args for args in commands if args[:2] == ["git", "archive"]
                )
                self.assertEqual(archive[-1], SHA)
                self.assertNotEqual(build[-1], ".")
                outputs = dict(
                    line.split("=", 1)
                    for line in (self.root / "output").read_text().splitlines()
                )
                self.assertEqual(outputs, {"index_digest": INDEX, **children})
                self.assertNotIn("example", (self.root / "output").read_text())
                self.assertFalse((self.root / "builder/config.json").exists())

    def test_pull_reports_only_fixed_numeric_metrics_without_private_metadata(self):
        output = io.StringIO()
        with patch.object(image, "_run", side_effect=self._run), patch.object(
            image.time, "monotonic", side_effect=[10.0, 10.5, 11.0, 81.25]
        ), redirect_stdout(output):
            image._pull()
        metrics = [
            json.loads(line.removeprefix("CI_IMAGE_METRICS "))
            for line in output.getvalue().splitlines()
        ]
        self.assertEqual(
            metrics,
            [
                {"family": "cpu", "phase": "pull-auth", "seconds": 0.5},
                {"family": "cpu", "phase": "pull", "seconds": 70.25},
            ],
        )
        for private in [REPOSITORY, DIGEST, SHA, "synthetic-pull-token"]:
            self.assertNotIn(private, output.getvalue())

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

    def test_build_requires_a_valid_project_and_uses_a_minted_pull_token(self):
        for change in [{"DEPOT_PROJECT_ID": ""}, {"DEPOT_PROJECT_ID": "Bad-Id"}]:
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
            'echo "private-token"\necho "private-token" >&2\n'
            'echo "TEST_PORT=1234" >> "$GITHUB_ENV"\nexit 17\n'
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


class WorkflowTests(unittest.TestCase):
    def test_gpu_test_commands_consume_verified_ids_including_make_targets(self):
        # Execute the real workflow shell and Make recipes with a recording
        # Docker executable. This checks image handoff, not CUDA execution.
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "commands.jsonl"
            docker = root / "docker"
            docker.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['DOCKER_COMMAND_RECORD'], 'a') as handle:\n"
                "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            )
            docker.chmod(0o700)
            env = {
                **os.environ,
                "PATH": f"{root}:{os.environ.get('PATH', '/usr/bin:/bin')}",
                "DOCKER_COMMAND_RECORD": str(record),
                "GITHUB_WORKSPACE": str(ROOT),
                "RUNNER_TEMP": str(root),
                "INFERENCE_TEST_IMAGE": IMAGE_ID,
                "INFERENCE_PARALLEL_TEST_IMAGE": INDEX,
            }
            for filename in [
                "test.colab.gpu.yml",
                "test.nvidia_t4_parallel_server.yml",
            ]:
                for step in _load(filename)["jobs"]["build"]["steps"]:
                    script = step.get("run", "")
                    if (
                        "docker run" not in script
                        and "make start_test_docker_gpu" not in script
                    ):
                        continue

                    with self.subTest(workflow=filename, step=step["name"]):
                        record.unlink(missing_ok=True)
                        result = subprocess.run(
                            ["bash", "-e", "-o", "pipefail", "-c", script],
                            cwd=ROOT,
                            env=env,
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        runs = [
                            json.loads(line)
                            for line in record.read_text().splitlines()
                            if json.loads(line)[0] == "run"
                        ]
                        self.assertTrue(runs)
                        expected = (
                            IMAGE_ID
                            if filename == "test.colab.gpu.yml"
                            or "make start_test_docker_gpu" in script
                            else INDEX
                        )
                        for args in runs:
                            self.assertIn(expected, args)
                            self.assertNotIn("inference-parallel-tests:local", args)
                            self.assertFalse(any(arg.endswith(":test") for arg in args))

    def test_gpu_builds_use_depot_and_only_tests_allocate_gpu_runners(self):
        orchestrator = _load("ci_image_reuse.yml")
        for job_id, filename, family in [
            ("colab-gpu", "test.colab.gpu.yml", "colab-gpu"),
            ("parallel-gpu", "test.nvidia_t4_parallel_server.yml", "parallel-gpu"),
        ]:
            with self.subTest(workflow=filename):
                workflow = _load(filename)
                triggers = workflow.get("on", workflow.get(True))
                self.assertEqual(set(triggers), {"workflow_dispatch", "workflow_call"})
                self.assertEqual(
                    triggers["workflow_dispatch"]["inputs"]["depot_project"]["default"],
                    "7m01vtml7h",
                )
                caller = orchestrator["jobs"][job_id]
                self.assertEqual(caller["uses"], f"./.github/workflows/{filename}")
                self.assertEqual(
                    caller["with"]["depot_project"],
                    orchestrator["jobs"]["gpu-image"]["with"]["depot_project"],
                )
                self.assertIn(
                    job_id, orchestrator["jobs"]["image-tests-complete"]["needs"]
                )
                self.assertNotEqual(
                    workflow["concurrency"]["group"],
                    orchestrator["concurrency"]["group"],
                )
                producer = workflow["jobs"]["image"]
                self.assertEqual(
                    producer["uses"], "./.github/workflows/ci_image_build.yml"
                )
                self.assertEqual(producer["with"]["family"], family)
                self.assertIn(
                    "github.repository == 'roboflow/inference'", producer["if"]
                )
                consumer = workflow["jobs"]["build"]
                self.assertEqual(consumer["runs-on"], "Roboflow-GPU-VM-Runner")
                steps = consumer["steps"]
                scripts = "\n".join(step.get("run", "") for step in steps)
                self.assertNotRegex(scripts, r"(?:docker|depot) build")
                self.assertIn(
                    (
                        "$INFERENCE_TEST_IMAGE"
                        if family == "colab-gpu"
                        else "$INFERENCE_PARALLEL_TEST_IMAGE"
                    ),
                    scripts,
                )
                self.assertEqual(consumer["permissions"]["id-token"], "write")
                self.assertTrue(
                    any(
                        step.get("env", {}).get("CI_DEPOT_ACTION") == "pull"
                        for step in steps
                    )
                )
        parallel = _load("test.nvidia_t4_parallel_server.yml")["jobs"]
        self.assertEqual(set(parallel["build"]["needs"]), {"image", "test-image"})
        self.assertEqual(
            parallel["test-image"]["with"]["server_image_digest"],
            "${{ needs.image.outputs.amd64_digest }}",
        )
        steps = parallel["build"]["steps"]
        test_pull = next(
            step
            for step in steps
            if step.get("env", {}).get("IMAGE_FAMILY") == "parallel-tests"
        )
        self.assertEqual(
            test_pull["env"]["SERVER_IMAGE_DIGEST"],
            "${{ needs.image.outputs.amd64_digest }}",
        )
        self.assertEqual(
            test_pull["env"]["IMAGE_DIGEST"],
            "${{ needs.test-image.outputs.amd64_digest }}",
        )

    def test_one_project_from_the_orchestrator_and_no_stored_depot_tokens(self):
        orchestrator = _load("ci_image_reuse.yml")
        projects = {
            orchestrator["jobs"][job]["with"]["depot_project"]
            for job in ["cpu-image", "gpu-image", "cpu-tests", "gpu-tests"]
        }
        self.assertEqual(len(projects), 1)
        # Pushes use the CI project; a manual dispatch may pick another one.
        expression = projects.pop()
        self.assertRegex(
            expression,
            r"^\$\{\{ github\.event_name == 'push' && '[a-z0-9]+' "
            r"\|\| inputs\.depot_project \|\| '[a-z0-9]+' \}\}$",
        )
        push_project, manual_project = [
            value
            for value in re.findall(r"'([a-z0-9]+)'", expression)
            if value != "push"
        ]
        self.assertIn(push_project, image._PROTECTED_PROJECTS)
        self.assertNotIn(manual_project, image._PROTECTED_PROJECTS)
        triggers = orchestrator.get("on", orchestrator.get(True))
        manual = triggers["workflow_dispatch"]["inputs"]["depot_project"]["default"]
        self.assertEqual(manual, manual_project)
        for name in ["ci_image_build.yml", *CONSUMERS]:
            workflow = _load(name)
            text = (ROOT / ".github/workflows" / name).read_text()
            self.assertNotRegex(
                text, r"DEPOT_[A-Z_]*TOKEN|CI_DEPOT_(?!ACTION)|CI_REGISTRY|CI_WIF"
            )
            triggers = workflow.get("on", workflow.get(True))
            self.assertIn("depot_project", triggers["workflow_call"]["inputs"])
            job = workflow["jobs"]["build"]
            self.assertEqual(
                job["env"]["DEPOT_PROJECT_ID"], "${{ inputs.depot_project }}"
            )
            self.assertEqual(
                job["permissions"], {"contents": "read", "id-token": "write"}
            )
            self.assertNotEqual(job.get("secrets"), "inherit")
            steps = job["steps"]
            setup = [
                index
                for index, step in enumerate(steps)
                if "depot/setup-action" in step.get("uses", "")
            ]
            self.assertEqual(len(setup), 1)
            depot_steps = [
                index
                for index, step in enumerate(steps)
                if step.get("env", {}).get("CI_DEPOT_ACTION") in {"build", "pull"}
            ]
            self.assertEqual(len(depot_steps), 1)
            self.assertLess(setup[0], depot_steps[0])
            self.assertEqual(
                set(steps[depot_steps[0]].get("env", {})), {"CI_DEPOT_ACTION"}
            )
            for step in steps:
                self.assertNotIn("upload-artifact", step.get("uses", ""))
                self.assertNotIn("${{ secrets.", step.get("run", ""))
                action = step.get("uses", "")
                if action and not action.startswith("./"):
                    self.assertRegex(action, r"@[0-9a-f]{40}$")

    def test_fast_track_trial_is_superseded(self):
        for name in [
            "fast_track_depot_images.yml",
            "depot_ci_image_build.yml",
            "depot_ci_image_tests.yml",
        ]:
            self.assertFalse((ROOT / ".github/workflows" / name).exists(), name)
        fast_track = ROOT / ".github/workflows/fast_track_integration_tests.yml"
        self.assertNotIn("use_depot_images", fast_track.read_text())
        self.assertNotRegex(
            "".join(
                path.read_text() for path in (ROOT / ".github/workflows").glob("*.yml")
            ),
            r"CI_DEPOT_(REUSE_ENABLED|APPROVED_REF|APPROVED_SHA|PROJECT|REGISTRY)",
        )

    def test_all_gpu_stages_and_shared_actions_remain_in_use(self):
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


if __name__ == "__main__":
    unittest.main()
