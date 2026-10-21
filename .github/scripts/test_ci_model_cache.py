"""Offline checks for the persistent GPU runner model cache and its wiring."""

import importlib.util
import os
import time
import unittest
from collections import namedtuple
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
GIB = 2**30
Usage = namedtuple("Usage", "total used free")


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
            "GITHUB_RUN_ID": "100",
            "GITHUB_RUN_ATTEMPT": "1",
            "MODEL_MODE": "true",
            "MODEL_CACHE_KEY": "v1",
            "INFERENCE_TEST_IMAGE": IMAGE_ID,
        }
        self.environment = patch.dict(os.environ, self.env, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []
        for target, value in [
            ("_run", self._run),
            ("print", lambda *args, **kwargs: None),
        ]:
            patcher = patch.object(CACHE, target, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.free = 500 * GIB
        self.freed_by_wipe = 0
        usage = patch.object(
            CACHE.shutil, "disk_usage", lambda path: Usage(0, 0, self.free)
        )
        usage.start()
        self.addCleanup(usage.stop)
        self.cache_dir = (
            self.root
            / ".cache"
            / "roboflow-inference-ci"
            / "model-weights"
            / "use-inference-models-true"
        )

    def _run(self, args):
        self.calls.append(args)
        if args[:2] == ["docker", "run"]:
            self.free += self.freed_by_wipe

        return ""

    def _populate(self, *, key="v1", created=None):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "weights.onnx").write_bytes(b"x" * 10)
        (self.cache_dir / CACHE.KEY_FILE).write_text(f"{key}\n")
        created = int(time.time()) if created is None else created
        (self.cache_dir / CACHE.CREATED_FILE).write_text(f"{created}\n")

    def _mark_in_use(self, *, identity, since=None):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        since = int(time.time()) if since is None else since
        (self.cache_dir / CACHE.IN_USE_FILE).write_text(f"{identity} {since}\n")

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

    def _mount_args(self):
        return (
            f"MODEL_CACHE_DOCKER_ARGS=-v {self.cache_dir}:/tmp/cache "
            "-e DISK_CACHE_CLEANUP=False -e ATOMIC_CACHE_WRITES_ENABLED=True\n"
        )

    def test_fresh_cache_gets_markers_and_a_per_leg_mount(self):
        """A new cache needs no Docker call and mounts the leg's own directory."""
        CACHE._prepare()

        self.assertEqual(self.calls, [])
        self.assertEqual((self.cache_dir / CACHE.KEY_FILE).read_text(), "v1\n")
        self.assertTrue((self.cache_dir / CACHE.CREATED_FILE).is_file())
        self.assertTrue(
            (self.cache_dir / CACHE.IN_USE_FILE).read_text().startswith("100-1 ")
        )
        self.assertEqual((self.root / "env").read_text(), self._mount_args())

    def test_valid_cache_is_reused_without_touching_docker(self):
        """A populated cache with the current key and age keeps its files."""
        self._populate(created=int(time.time()) - 13 * DAY)
        for reset in ["false", ""]:
            with self.subTest(reset=reset), patch.dict(
                os.environ, {"MODEL_CACHE_RESET": reset}
            ):
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

    def test_low_free_disk_wipes_then_fails_when_still_low(self):
        """A full disk discards the cache first and fails only if that is not enough."""
        self._populate()
        self.free = 10 * GIB
        with self.assertRaises(ValueError):
            CACHE._prepare()
        self.assertEqual(self.calls, [self._wipe_command()])
        self.assertFalse((self.cache_dir / CACHE.IN_USE_FILE).exists())

        self.calls.clear()
        self.free = 25 * GIB
        self._populate()
        CACHE._prepare()
        self.assertEqual(self.calls, [])

        self.free = 25 * GIB
        self.freed_by_wipe = 20 * GIB
        (self.root / "env").write_text("")
        with patch.dict(os.environ, {"MODEL_CACHE_MIN_FREE_GIB": "30"}):
            CACHE._prepare()
        self.assertEqual(self.calls, [self._wipe_command()])
        self.assertEqual((self.root / "env").read_text(), self._mount_args())

    def test_cache_held_by_a_live_run_is_not_mounted(self):
        """Another run's fresh marker leaves its files alone and skips the mount."""
        self._populate(key="v0")
        self._mark_in_use(identity="99-1")

        CACHE._prepare()

        self.assertEqual(self.calls, [])
        self.assertEqual((self.root / "env").read_text(), "MODEL_CACHE_DOCKER_ARGS=\n")
        self.assertEqual((self.cache_dir / CACHE.KEY_FILE).read_text(), "v0\n")
        self.assertTrue(
            (self.cache_dir / CACHE.IN_USE_FILE).read_text().startswith("99-1 ")
        )

    def test_marker_from_a_lost_run_or_this_run_is_taken_over(self):
        """A marker past the job timeout, or our own, does not block the cache."""
        for identity, since in [("99-1", int(time.time()) - 121 * 60), ("100-1", None)]:
            with self.subTest(identity=identity):
                self._populate()
                self._mark_in_use(identity=identity, since=since)
                (self.root / "env").write_text("")
                CACHE._prepare()
                self.assertEqual((self.root / "env").read_text(), self._mount_args())
                self.assertTrue(
                    (self.cache_dir / CACHE.IN_USE_FILE)
                    .read_text()
                    .startswith("100-1 ")
                )

    def test_release_removes_only_this_runs_marker(self):
        """Releasing drops our marker and never another run's."""
        self._mark_in_use(identity="100-1")
        CACHE._release()
        self.assertFalse((self.cache_dir / CACHE.IN_USE_FILE).exists())

        self._mark_in_use(identity="99-1")
        CACHE._release()
        self.assertTrue((self.cache_dir / CACHE.IN_USE_FILE).exists())

        CACHE._release()

    def test_wipe_falls_back_to_the_make_image_name(self):
        """Without a pulled image id the Makefile's :test alias is used."""
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
        """Malformed mode, key, numbers or run identity never reach Docker."""
        self._populate(key="v0")
        cases = [
            {"MODEL_MODE": "unexpected"},
            {"MODEL_CACHE_KEY": ""},
            {"MODEL_CACHE_KEY": "v 1"},
            {"MODEL_CACHE_KEY": "x\nINJECT=value"},
            {"MODEL_CACHE_MAX_AGE_DAYS": "two"},
            {"MODEL_CACHE_MIN_FREE_GIB": "-1"},
            {"MODEL_CACHE_BUSY_TIMEOUT_MINUTES": ""},
            {"GITHUB_RUN_ID": "abc"},
            {"GITHUB_RUN_ATTEMPT": ""},
        ]
        for values in cases:
            with self.subTest(values=values), patch.dict(os.environ, values):
                self.calls.clear()
                with self.assertRaises(ValueError):
                    CACHE._prepare()
                self.assertEqual(self.calls, [])

    def test_every_gpu_stage_mounts_the_cache_and_the_job_releases_it(self):
        """The action prepares first, mounts on every server start and releases last."""
        action = yaml.safe_load(
            (
                ROOT / ".github" / "actions" / "server-tests-gpu" / "action.yml"
            ).read_text()
        )
        steps = action["runs"]["steps"]
        self.assertEqual(steps[0]["env"]["CI_MODEL_CACHE_ACTION"], "prepare")
        self.assertEqual(steps[-1]["env"]["CI_MODEL_CACHE_ACTION"], "release")
        self.assertEqual(steps[-1]["if"], "${{ always() }}")
        for step in (steps[0], steps[-1]):
            self.assertEqual(
                step["run"].strip(), "python3 .github/scripts/ci_model_cache.py"
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
