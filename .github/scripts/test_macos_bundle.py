"""Exercise recovery selection and bundle failure gates without Apple credentials."""

import importlib.util
import json
import os
import runpy
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/build-osx-bundle.yml").read_text())
STEPS = {step["name"]: step for step in WORKFLOW["jobs"]["build_osx"]["steps"]}
TOOLING = ROOT / "app_bundles/osx"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.output.touch()
        self.env = {
            **os.environ,
            "EVENT_NAME": "workflow_dispatch",
            "EVENT_TAG": "",
            "RELEASE_TAG": "",
            "MANUAL_VERSION": "",
            "PUBLISH_RELEASE": "false",
            "GITHUB_SHA": "a" * 40,
            "GITHUB_OUTPUT": str(self.output),
            "BUILD_VERSION": "1.8.6",
            "GH_LOG": str(self.root / "gh.json"),
            "RELEASE_METADATA": json.dumps({"isDraft": False, "assets": []}),
        }
        gh = self.root / "gh"
        gh.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['GH_LOG']).write_text(json.dumps(sys.argv[1:]))\n"
            "if os.environ.get('GH_FAIL'): sys.exit(1)\n"
            "print(os.environ['RELEASE_METADATA'])\n"
        )
        gh.chmod(0o755)
        self.env["PATH"] = f"{self.root}:{os.environ['PATH']}"

    def _run(self, step, **env):
        result = subprocess.run(
            ["bash", "-e", "-o", "pipefail", "-c", STEPS[step]["run"]],
            cwd=self.root,
            env={**self.env, **env},
            capture_output=True,
            text=True,
        )
        return result

    def _target(self, **env):
        result = self._run("Select source and release target", **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(
            line.split("=", 1) for line in self.output.read_text().splitlines()
        )
        return outputs

    def test_recovery_selects_tag_not_workflow_commit(self):
        outputs = self._target(RELEASE_TAG="v1.8.6", PUBLISH_RELEASE="true")
        self.assertEqual(outputs["source_ref"], "refs/tags/v1.8.6")
        self.assertEqual(outputs["version"], "1.8.6")
        self.assertEqual(outputs["release_tag"], "v1.8.6")
        self.assertEqual(outputs["publish"], "true")
        # The checkout consumes the resolver output; tooling stays on the
        # selected workflow commit, rather than the old tag's broken scripts.
        self.assertEqual(
            STEPS["Checkout application source"]["with"]["ref"],
            "${{ steps.target.outputs.source_ref }}",
        )
        self.assertEqual(
            STEPS["Checkout build tooling"]["with"]["ref"], "${{ github.sha }}"
        )

    def test_manual_release_build_does_not_publish_by_default(self):
        outputs = self._target(RELEASE_TAG="v1.8.6")
        self.assertEqual(outputs["publish"], "false")

    def test_published_release_preserves_automatic_upload(self):
        outputs = self._target(EVENT_NAME="release", EVENT_TAG="v1.8.6")
        self.assertEqual(outputs["source_ref"], "refs/tags/v1.8.6")
        self.assertEqual(outputs["publish"], "true")

    def test_tag_without_v_keeps_exact_release_name(self):
        outputs = self._target(RELEASE_TAG="1.8.6")
        self.assertEqual(outputs["source_ref"], "refs/tags/1.8.6")
        self.assertEqual(outputs["release_tag"], "1.8.6")

    def test_existing_manual_version_is_artifact_only(self):
        outputs = self._target(MANUAL_VERSION="v1.8.6")
        self.assertEqual(outputs["source_ref"], self.env["GITHUB_SHA"])
        self.assertEqual(outputs["version"], "1.8.6")
        self.assertEqual(outputs["publish"], "false")

    def test_empty_manual_version_preserves_dev_default(self):
        outputs = self._target()
        self.assertEqual(outputs["version"], "0.0.0-manual-dev")
        self.assertEqual(outputs["publish"], "false")

    def test_invalid_target_combinations_fail_before_checkout(self):
        for env in (
            {"PUBLISH_RELEASE": "true"},
            {"EVENT_NAME": "release"},
            {"RELEASE_TAG": "v1.8.6", "MANUAL_VERSION": "1.9.0"},
            {"RELEASE_TAG": "main"},
            {"RELEASE_TAG": "v1.8.6\npublish=true"},
            {"RELEASE_TAG": "v1.8.6$(touch injected)"},
            {"MANUAL_VERSION": "1.8.6; touch injected"},
        ):
            with self.subTest(env=env):
                result = self._run("Select source and release target", **env)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.output.read_text(), "")
                self.assertFalse((self.root / "injected").exists())

    def test_preflight_accepts_existing_release_with_windows_assets(self):
        result = self._run(
            "Check existing release before building",
            RELEASE_TAG="v1.8.6",
            RELEASE_METADATA=json.dumps(
                {"isDraft": False, "assets": [{"name": "inference-1.8.6.exe"}]}
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.root / "gh.json").read_text())
        self.assertEqual(args[:3], ["release", "view", "v1.8.6"])

    def test_preflight_rejects_missing_draft_and_existing_dmg(self):
        for env in (
            {"GH_FAIL": "1"},
            {"RELEASE_METADATA": '{"isDraft":true,"assets":[]}'},
            {
                "RELEASE_METADATA": json.dumps(
                    {
                        "isDraft": False,
                        "assets": [{"name": "Roboflow-Inference-1.8.6.dmg"}],
                    }
                )
            },
        ):
            with self.subTest(env=env):
                result = self._run("Check existing release before building", **env)
                self.assertNotEqual(result.returncode, 0)

    def test_upload_uses_exact_tag_and_never_clobbers(self):
        result = self._run("Upload DMG to Release", RELEASE_TAG="v1.8.6")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads((self.root / "gh.json").read_text()),
            [
                "release",
                "upload",
                "v1.8.6",
                "app_bundles/osx/Roboflow-Inference-1.8.6.dmg",
            ],
        )
        result = self._run("Upload DMG to Release", GH_FAIL="1")
        self.assertNotEqual(result.returncode, 0)

    def test_upload_is_gated_after_smoke_test(self):
        names = list(STEPS)
        self.assertLess(
            names.index("Verify and smoke test the DMG"),
            names.index("Upload DMG to Release"),
        )
        self.assertEqual(
            STEPS["Upload DMG to Release"]["if"],
            "steps.target.outputs.publish == 'true'",
        )
        self.assertNotIn("continue-on-error", STEPS["Verify and smoke test the DMG"])

    def test_notary_credentials_are_stored_in_explicit_keychain(self):
        xcrun = self.root / "xcrun"
        xcrun.write_text((self.root / "gh").read_text())
        xcrun.chmod(0o755)
        result = self._run(
            "Configure Notarytool",
            NOTARY_PROFILE_NAME="roboflow-notary",
            KEYCHAIN_NAME="/tmp/build.keychain",
            NOTARY_APPLE_ID_SECRET="example@example.com",
            NOTARY_TEAM_ID_SECRET="test-team",
            NOTARY_APP_SPECIFIC_PASSWORD_SECRET="test-password",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        command = json.loads((self.root / "gh.json").read_text())
        self.assertEqual(
            command[:3], ["notarytool", "store-credentials", "roboflow-notary"]
        )
        self.assertEqual(
            command[command.index("--keychain") + 1], "/tmp/build.keychain"
        )


class PackagingTests(unittest.TestCase):
    def test_notary_submission_uses_configured_keychain(self):
        build = _load("macos_build", TOOLING / "build.py")
        with patch.dict(os.environ, {"NOTARY_KEYCHAIN_PATH": "/tmp/build.keychain"}):
            with patch.object(build.subprocess, "run") as run:
                build.notarize_app("app.zip")
        command = run.call_args.args[0]
        self.assertEqual(
            command[command.index("--keychain") + 1], "/tmp/build.keychain"
        )
        self.assertEqual(
            command[command.index("--keychain-profile") + 1], "roboflow-notary"
        )
        self.assertTrue(run.call_args.kwargs["check"])

    def test_local_notary_default_is_preserved(self):
        build = _load("macos_build_local", TOOLING / "build.py")
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(build.subprocess, "run") as run:
                build.notarize_app("app.zip")
        self.assertNotIn("--keychain", run.call_args.args[0])

    def test_notary_failure_propagates(self):
        build = _load("macos_build_failure", TOOLING / "build.py")
        with patch.object(
            build.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(69, "notarytool"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                build.notarize_app("app.zip")

    def test_hook_collects_stable_abi_libraries_at_package_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "torchvision"
            package.mkdir()
            (package / "__init__.py").touch()
            (package / ".dylibs").mkdir()
            for relative in (
                "_C_stable.so",
                "image_stable.so",
                ".dylibs/libjpeg.dylib",
            ):
                (package / relative).touch()
            with patch.object(sys, "path", [directory, *sys.path]):
                result = runpy.run_path(str(TOOLING / "hooks/hook-torchvision.py"))
            actual = {
                (Path(source).relative_to(package).as_posix(), destination)
                for source, destination in result["binaries"]
            }
            self.assertEqual(
                actual,
                {
                    ("_C_stable.so", "torchvision"),
                    ("image_stable.so", "torchvision"),
                    (".dylibs/libjpeg.dylib", "torchvision/.dylibs"),
                },
            )
            self.assertEqual(result["module_collection_mode"], "pyz+py")


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.smoke = _load("macos_smoke", TOOLING / "smoke_test.py")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def _server(self, code):
        executable = self.root / "server"
        executable.write_text(f"#!{sys.executable}\n" + code)
        executable.chmod(0o755)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        result = self.smoke._run_server(
            executable, directory=self.root, port=port, timeout=2
        )
        return result

    def test_real_ready_process_passes_and_is_stopped(self):
        elapsed = self._server(
            "import http.server, os\n"
            "from pathlib import Path\n"
            "Path('pid').write_text(str(os.getpid()))\n"
            "class Handler(http.server.BaseHTTPRequestHandler):\n"
            " def do_GET(self):\n"
            "  self.send_response(200 if self.path == '/readiness' else 404)\n"
            "  self.end_headers()\n"
            "http.server.HTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()\n"
        )
        self.assertLess(elapsed, 2)
        with self.assertRaises(ProcessLookupError):
            os.kill(int((self.root / "pid").read_text()), 0)

    def test_real_crashed_process_fails_with_log(self):
        with patch("builtins.print") as output:
            with self.assertRaisesRegex(RuntimeError, "exited"):
                self._server(
                    "print('torchvision native library missing', flush=True)\nraise SystemExit(1)\n"
                )
        self.assertIn("torchvision native library missing", output.call_args.args[0])

    def test_real_unready_process_times_out(self):
        with self.assertRaisesRegex(RuntimeError, "did not become ready"):
            self._server("import time\ntime.sleep(60)\n")

    def test_signature_or_ticket_failure_detaches_without_starting_server(self):
        dmg = self.root / "test.dmg"
        dmg.touch()
        for failing_command in ("codesign", "xcrun"):
            with self.subTest(failing_command=failing_command):

                def run(command, **kwargs):
                    if command[0] == failing_command:
                        raise subprocess.CalledProcessError(1, command)

                with patch.object(
                    self.smoke.subprocess, "run", side_effect=run
                ) as native:
                    with patch.object(self.smoke, "_run_server") as server:
                        result = CliRunner().invoke(
                            self.smoke.main, ["--dmg", str(dmg)]
                        )
                self.assertNotEqual(result.exit_code, 0)
                server.assert_not_called()
                self.assertEqual(native.call_args.args[0][:2], ["hdiutil", "detach"])


if __name__ == "__main__":
    unittest.main()
