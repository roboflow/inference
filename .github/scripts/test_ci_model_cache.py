"""Offline checks for the persistent GPU runner model cache and its wiring."""

import importlib.util
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_model_cache", Path(__file__).with_name("ci_model_cache.py")
)
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)
IMAGE_ID = "sha256:" + "c" * 64
DAY = 86400


class ModelCacheTests(unittest.TestCase):
    """Exercise invalidation and exact Docker command construction offline."""

    def setUp(self):
        """Use a private HOME and GITHUB_ENV file with a reusable cache."""
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.env = {
            "HOME": str(self.root),
            "GITHUB_ENV": str(self.root / "env"),
            "MODEL_MODE": "true",
            "MODEL_CACHE_KEY": "v1",
            "INFERENCE_TEST_IMAGE": IMAGE_ID,
        }
        self.environment = patch.dict(os.environ, self.env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []
        self.runner = patch.object(CACHE, "_run", side_effect=self._run)
        self.runner.start()
        self.addCleanup(self.runner.stop)
        self.printer = patch.object(CACHE, "print", create=True)
        self.printer.start()
        self.addCleanup(self.printer.stop)
        self.cache_dir = (
            self.root
            / ".cache"
            / "roboflow-inference-ci"
            / "model-weights"
            / "use-inference-models-true"
        )

    def _run(self, args):
        self.calls.append(args)
        return ""

    def _populate(self, *, key="v1", created=None):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "weights.onnx").write_bytes(b"x" * 10)
        (self.cache_dir / CACHE.KEY_FILE).write_text(f"{key}\n")
        created = int(time.time()) if created is None else created
        (self.cache_dir / CACHE.CREATED_FILE).write_text(f"{created}\n")

    def _wipe_command(self):
        return [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "find",
            "-v",
            f"{self.cache_dir}:/cache",
            IMAGE_ID,
            "/cache",
            "-mindepth",
            "1",
            "-delete",
        ]

    def test_fresh_cache_gets_markers_and_a_per_leg_mount(self):
        """A new cache needs no Docker call and mounts the leg's own directory."""
        CACHE._prepare()

        self.assertEqual(self.calls, [])
        self.assertEqual((self.cache_dir / CACHE.KEY_FILE).read_text(), "v1\n")
        self.assertTrue((self.cache_dir / CACHE.CREATED_FILE).is_file())
        self.assertEqual(
            (self.root / "env").read_text(),
            f"MODEL_CACHE_DOCKER_ARGS=-v {self.cache_dir}:/tmp/cache "
            "-e DISK_CACHE_CLEANUP=False\n",
        )

    def test_valid_cache_is_reused_without_touching_docker(self):
        """A populated cache with the current key and age keeps its files."""
        self._populate(created=int(time.time()) - 13 * DAY)

        CACHE._prepare()

        self.assertEqual(self.calls, [])
        self.assertTrue((self.cache_dir / "weights.onnx").is_file())

    def test_changed_key_wipes_through_the_test_image(self):
        """A bumped key deletes the contents from inside the test image."""
        self._populate(key="v0")

        CACHE._prepare()

        self.assertEqual(self.calls, [self._wipe_command()])
        self.assertEqual((self.cache_dir / CACHE.KEY_FILE).read_text(), "v1\n")

    def test_stale_cache_is_wiped_only_past_the_maximum_age(self):
        """The default age is 14 days and MODEL_CACHE_MAX_AGE_DAYS overrides it."""
        created = int(time.time()) - 15 * DAY
        for days, wiped in [(None, True), ("30", False), ("7", True)]:
            with self.subTest(days=days):
                self.calls.clear()
                self._populate(created=created)
                values = {} if days is None else {"MODEL_CACHE_MAX_AGE_DAYS": days}
                with patch.dict(os.environ, values):
                    CACHE._prepare()
                self.assertEqual(self.calls, [self._wipe_command()] if wiped else [])

    def test_reset_wipes_a_populated_cache_but_not_an_empty_one(self):
        """An explicit reset deletes files and never runs Docker on nothing."""
        with patch.dict(os.environ, {"MODEL_CACHE_RESET": "true"}):
            CACHE._prepare()
            self.assertEqual(self.calls, [])

            self._populate()
            CACHE._prepare()
            self.assertEqual(self.calls, [self._wipe_command()])

    def test_wipe_falls_back_to_the_make_image_name(self):
        """Without a pulled image id the Makefile's :test alias is used."""
        self._populate(key="v0")
        for repository, image in [
            ("", "roboflow/roboflow-inference-server-gpu:test"),
            (
                "inference-server-experimental",
                "roboflow/inference-server-experimental:test",
            ),
        ]:
            with self.subTest(repository=repository), patch.dict(
                os.environ,
                {"INFERENCE_TEST_IMAGE": "", "INFERENCE_SERVER_REPO": repository},
            ):
                self.calls.clear()
                self._populate(key="v0")
                CACHE._prepare()
                self.assertEqual(self.calls[0][7], image)

    def test_invalid_inputs_never_reach_docker(self):
        """Malformed mode, key or age fail before any Docker command."""
        self._populate(key="v0")
        cases = [
            {"MODEL_MODE": "unexpected"},
            {"MODEL_CACHE_KEY": ""},
            {"MODEL_CACHE_KEY": "v 1"},
            {"MODEL_CACHE_KEY": "x\nINJECT=value"},
            {"MODEL_CACHE_MAX_AGE_DAYS": "two"},
        ]
        for values in cases:
            with self.subTest(values=values), patch.dict(os.environ, values):
                self.calls.clear()
                with self.assertRaises(ValueError):
                    CACHE._prepare()
                self.assertEqual(self.calls, [])

    def test_every_gpu_stage_mounts_the_cache(self):
        """Each server start in the shared GPU action passes the cache arguments."""
        action = yaml.safe_load(
            (
                ROOT / ".github" / "actions" / "server-tests-gpu" / "action.yml"
            ).read_text()
        )
        steps = action["runs"]["steps"]
        self.assertEqual(
            steps[0]["run"].strip(), "python3 .github/scripts/ci_model_cache.py"
        )
        starts = [
            step
            for step in steps
            if "make start_test_docker_gpu" in step.get("run", "")
        ]
        self.assertGreater(len(starts), 0)
        for step in starts:
            self.assertIn(
                'INFERENCE_TEST_DOCKER_ARGS="$MODEL_CACHE_DOCKER_ARGS', step["run"]
            )


if __name__ == "__main__":
    unittest.main()
