import json
import os
import threading
import time
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest
from filelock import FileLock, Timeout

from inference.core.cache import model_artifacts
from inference.core.exceptions import ModelArtefactError
from inference.core.models import roboflow
from inference.core.models.roboflow import (
    acquire_model_download_lock,
    class_mapping_not_available_in_environment,
    color_mapping_available_in_environment,
    get_class_names_from_environment_file,
    get_color_mapping_from_environment,
    get_model_download_lock_path,
    is_model_artefacts_bucket_available,
)


@mock.patch.object(roboflow, "AWS_ACCESS_KEY_ID", None)
def test_is_model_artefacts_bucket_available_when_access_key_not_set() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "AWS_SECRET_ACCESS_KEY", None)
def test_is_model_artefacts_bucket_available_when_secret_not_set() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "LAMBDA", False)
def test_is_model_artefacts_bucket_available_when_not_in_lambda_mode() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "S3_CLIENT", None)
def test_is_model_artefacts_bucket_available_when_s3_client_not_initialised() -> None:
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is False


@mock.patch.object(roboflow, "AWS_ACCESS_KEY_ID", "some")
@mock.patch.object(roboflow, "AWS_SECRET_ACCESS_KEY", "other")
@mock.patch.object(roboflow, "LAMBDA", True)
@mock.patch.object(roboflow, "S3_CLIENT", MagicMock())
def test_is_model_artefacts_bucket_available_when_availability_check_should_pass() -> (
    None
):
    # when
    result = is_model_artefacts_bucket_available()

    # then
    assert result is True


@pytest.mark.parametrize(
    "environment, expected_result",
    [
        (None, False),
        ({}, False),
        ({"COLORS": json.dumps({"class_a": "#ffffff"})}, False),
        ({"COLORS": {"class_a": "#ffffff"}}, True),
    ],
)
def test_color_mapping_available_in_environment_when_environment(
    environment: Optional[dict], expected_result: bool
) -> None:
    # when
    result = color_mapping_available_in_environment(environment=environment)

    # then
    assert result is expected_result


def test_get_color_mapping_from_environment_when_color_mapping_in_environment() -> None:
    # given
    environment = {"COLORS": {"class_a": "#ffffff"}}

    # when
    result = get_color_mapping_from_environment(
        environment=environment, class_names=["class_a"]
    )

    # then
    assert result == {"class_a": "#ffffff"}


def test_get_color_mapping_from_environment_when_color_mapping_in_environment_as_json_string() -> (
    None
):
    # given
    environment = {"COLORS": json.dumps({"class_a": "#ffffff"})}

    # when
    result = get_color_mapping_from_environment(
        environment=environment, class_names=["class_a"]
    )

    # then
    assert result == {"class_a": "#4892EA"}


def test_get_color_mapping_from_environment_when_color_mapping_not_in_environment() -> (
    None
):
    # given
    environment = {}

    # when
    result = get_color_mapping_from_environment(
        environment=environment, class_names=["class_a", "class_b"]
    )

    # then
    assert result == {"class_a": "#4892EA", "class_b": "#00EEC3"}


@pytest.mark.parametrize(
    "environment, expected_result",
    [
        ({}, True),
        ({"CLASS_MAP": json.dumps({"0": "class_a"})}, True),
        ({"CLASS_MAP": {"0": "class_1"}}, False),
    ],
)
def test_class_mapping_not_available_in_environment(
    environment: dict, expected_result: bool
) -> None:
    # when
    result = class_mapping_not_available_in_environment(environment=environment)

    # then
    assert result is expected_result


@pytest.mark.parametrize(
    "environment", [None, {}, {"CLASS_MAP": json.dumps({"0": "class_a"})}]
)
def test_get_class_names_from_environment_file_when_procedure_should_fail(
    environment: Optional[dict],
) -> None:
    # when
    with pytest.raises(ModelArtefactError):
        _ = get_class_names_from_environment_file(environment=environment)


def test_get_class_names_from_environment_file() -> None:
    # given
    environment = {
        "CLASS_MAP": {
            "0": "class_a",
            "1": "class_b",
            "2": "class_c",
            "3": "class_d",
            "4": "class_e",
            "5": "class_f",
            "6": "class_g",
            "7": "class_h",
            "8": "class_i",
            "9": "class_j",
            "10": "class_k",
            "11": "class_l",
        }
    }

    # when
    result = get_class_names_from_environment_file(environment=environment)

    # then
    assert result == [
        "class_a",
        "class_b",
        "class_c",
        "class_d",
        "class_e",
        "class_f",
        "class_g",
        "class_h",
        "class_i",
        "class_j",
        "class_k",
        "class_l",
    ]


def test_get_model_download_lock_path_is_stable_for_same_model(tmp_path) -> None:
    # given
    with mock.patch.object(roboflow, "MODEL_CACHE_DIR", str(tmp_path)):
        cache_dir = str(tmp_path / "dataset-a" / "1")

        # when - called twice, including via a non-normalized path
        first = get_model_download_lock_path(cache_dir=cache_dir)
        second = get_model_download_lock_path(cache_dir=cache_dir + "/")

    # then - siblings loading the same model must agree on one lock file
    assert first == second


@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_MAX_ATTEMPTS", 3)
@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_TIMEOUT", 600.0)
@mock.patch.object(roboflow, "FileLock")
def test_acquire_model_download_lock_returns_held_lock_on_first_success(
    file_lock_mock: MagicMock,
) -> None:
    # given
    lock = MagicMock()
    file_lock_mock.return_value = lock

    # when
    result = acquire_model_download_lock("/tmp/model.lock", model_id="m/1")

    # then
    assert result is lock
    file_lock_mock.assert_called_once_with("/tmp/model.lock", timeout=600.0)
    lock.acquire.assert_called_once()


@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_MAX_ATTEMPTS", 3)
@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_TIMEOUT", 600.0)
@mock.patch.object(roboflow, "FileLock")
def test_acquire_model_download_lock_retries_on_timeout_then_succeeds(
    file_lock_mock: MagicMock,
) -> None:
    # given - first lock times out on acquire (holder still downloading), second succeeds
    timing_out_lock = MagicMock()
    timing_out_lock.acquire.side_effect = Timeout("/tmp/model.lock")
    succeeding_lock = MagicMock()
    file_lock_mock.side_effect = [timing_out_lock, succeeding_lock]

    # when
    result = acquire_model_download_lock("/tmp/model.lock", model_id="m/1")

    # then
    assert result is succeeding_lock
    assert file_lock_mock.call_count == 2


@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_MAX_ATTEMPTS", 3)
@mock.patch.object(roboflow, "MODEL_WEIGHTS_DOWNLOAD_LOCK_TIMEOUT", 600.0)
@mock.patch.object(roboflow, "FileLock")
def test_acquire_model_download_lock_raises_after_exhausting_attempts(
    file_lock_mock: MagicMock,
) -> None:
    # given - every attempt times out
    always_timing_out_lock = MagicMock()
    always_timing_out_lock.acquire.side_effect = Timeout("/tmp/model.lock")
    file_lock_mock.return_value = always_timing_out_lock

    # when / then
    with pytest.raises(Timeout):
        acquire_model_download_lock("/tmp/model.lock", model_id="m/1")
    assert file_lock_mock.call_count == 3


# --- End-to-end lock behavior, exercising the REAL FileLock across threads. ---
# The mocked tests above cover the helpers in isolation; these cover the contract
# the four download implementations actually depend on: exactly one downloader
# runs, waiters reuse the warmed cache instead of re-downloading, and the lock is
# released even when the guarded block raises.


class _FakeDownloader:
    """Minimal stand-in for a model performing a guarded weight download.

    Mirrors the production shape: acquire the per-model lock, re-check the cache,
    then download only if still cold - writing artifacts the way a real download
    would so a concurrent waiter observes a warm cache.
    """

    def __init__(self, cache_dir: str, endpoint: str, download_seconds: float = 0.0):
        self.cache_dir = cache_dir
        self.endpoint = endpoint
        self.download_seconds = download_seconds
        self.downloads_performed = 0
        self.skipped_because_warm = False

    def required_files(self):
        return ["weights.onnx", "environment.json"]

    def download(self, raise_inside_lock: bool = False) -> None:
        lock_file = get_model_download_lock_path(cache_dir=self.cache_dir)
        lock = acquire_model_download_lock(lock_file, model_id=self.endpoint)
        try:
            if model_artifacts.are_all_files_cached(
                files=self.required_files(), model_id=self.endpoint
            ):
                self.skipped_because_warm = True
                return
            if raise_inside_lock:
                raise ModelArtefactError("boom")
            self.downloads_performed += 1
            time.sleep(self.download_seconds)
            os.makedirs(self.cache_dir, exist_ok=True)
            for file_name in self.required_files():
                with open(os.path.join(self.cache_dir, file_name), "wb") as f:
                    f.write(b"payload")
        finally:
            lock.release()


@pytest.fixture
def isolated_model_cache(tmp_path):
    """Point both the lock helper and the artifact cache at a temp directory."""
    cache_root = str(tmp_path / "cache")
    os.makedirs(cache_root, exist_ok=True)
    with mock.patch.object(roboflow, "MODEL_CACHE_DIR", cache_root), mock.patch.object(
        model_artifacts, "MODEL_CACHE_DIR", cache_root
    ):
        yield cache_root


def test_concurrent_downloads_serialize_and_waiter_reuses_warm_cache(
    isolated_model_cache,
) -> None:
    # given - two "pipelines" racing to load the same model on a cold cache
    endpoint = "dataset/1"
    model_cache_dir = os.path.join(isolated_model_cache, endpoint)
    downloaders = [
        _FakeDownloader(model_cache_dir, endpoint, download_seconds=0.3),
        _FakeDownloader(model_cache_dir, endpoint, download_seconds=0.3),
    ]
    errors = []

    def run(downloader):
        try:
            downloader.download()
        except Exception as error:  # pragma: no cover - surfaced via assertion
            errors.append(error)

    # when - both start at the same time
    threads = [threading.Thread(target=run, args=(d,)) for d in downloaders]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    # then - the lock serialized them: exactly one downloaded, the other found a
    # warm cache after waiting and skipped the redundant fetch entirely.
    assert not errors, f"unexpected errors: {errors}"
    assert not any(t.is_alive() for t in threads), "threads deadlocked on the lock"
    assert sum(d.downloads_performed for d in downloaders) == 1
    assert sum(1 for d in downloaders if d.skipped_because_warm) == 1
    for file_name in downloaders[0].required_files():
        assert os.path.isfile(os.path.join(model_cache_dir, file_name))


def test_download_lock_is_released_when_guarded_block_raises(
    isolated_model_cache,
) -> None:
    # given - a download that fails partway through while holding the lock
    endpoint = "dataset/1"
    model_cache_dir = os.path.join(isolated_model_cache, endpoint)
    failing = _FakeDownloader(model_cache_dir, endpoint)

    # when
    with pytest.raises(ModelArtefactError):
        failing.download(raise_inside_lock=True)

    # then - a later attempt must not inherit a stuck lock
    lock_file = get_model_download_lock_path(cache_dir=model_cache_dir)
    probe = FileLock(lock_file, timeout=1)
    probe.acquire()
    probe.release()

    # and the retry genuinely succeeds through the normal path
    recovering = _FakeDownloader(model_cache_dir, endpoint)
    recovering.download()
    assert recovering.downloads_performed == 1


def test_waiter_blocks_until_holder_releases_the_lock(isolated_model_cache) -> None:
    # given - the lock is already held, as it would be during a cold download
    endpoint = "dataset/1"
    model_cache_dir = os.path.join(isolated_model_cache, endpoint)
    lock_file = get_model_download_lock_path(cache_dir=model_cache_dir)
    holder = FileLock(lock_file, timeout=5)
    holder.acquire()
    acquired_at = []

    def waiter():
        lock = acquire_model_download_lock(lock_file, model_id=endpoint)
        acquired_at.append(time.monotonic())
        lock.release()

    # when - a sibling tries to acquire while the holder still has it
    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.4)
    assert not acquired_at, "waiter acquired the lock while it was held"
    released_at = time.monotonic()
    holder.release()
    thread.join(timeout=30)

    # then - it waited rather than erroring, and proceeded once the lock freed
    assert not thread.is_alive(), "waiter never acquired the lock"
    assert len(acquired_at) == 1
    assert acquired_at[0] >= released_at


def test_real_download_method_skips_network_when_cache_warmed_while_waiting(
    isolated_model_cache,
) -> None:
    """Drive the production method, not a stand-in.

    The threaded tests above prove the lock primitive behaves; this proves
    `RoboflowInferenceModel.download_model_artifacts_from_roboflow_api` actually
    honors the post-acquisition cache check - i.e. a waiter that wakes up to a
    cache another pipeline just warmed performs no API call or download at all.
    """
    # given - a model whose artifacts a sibling already finished downloading
    endpoint = "dataset/1"
    model_cache_dir = os.path.join(isolated_model_cache, endpoint)
    os.makedirs(model_cache_dir, exist_ok=True)
    required = ["weights.onnx", "environment.json"]
    for file_name in required:
        with open(os.path.join(model_cache_dir, file_name), "wb") as f:
            f.write(b"payload")

    model = mock.Mock()
    model.endpoint = endpoint
    model.cache_dir = model_cache_dir
    model.version_id = "1"
    model.api_key = "key"
    model.device_id = "device"
    model.weights_file = "weights.onnx"
    model.get_all_required_infer_bucket_file.return_value = list(required)

    # when
    with mock.patch.object(
        roboflow, "get_roboflow_model_data"
    ) as api_mock, mock.patch.object(
        roboflow, "get_from_url"
    ) as url_mock, mock.patch.object(
        roboflow, "save_bytes_in_cache"
    ) as save_mock:
        roboflow.RoboflowInferenceModel.download_model_artifacts_from_roboflow_api(
            model
        )

    # then - the warm cache short-circuits before any network or disk write
    api_mock.assert_not_called()
    url_mock.assert_not_called()
    save_mock.assert_not_called()


def test_real_download_method_releases_lock_when_download_fails(
    isolated_model_cache,
) -> None:
    """A failed download must not leave the per-model lock held.

    Without this, one transient API failure would wedge every sibling pipeline
    for the full retry budget and then fail them too.
    """
    # given - a cold cache and an API call that blows up mid-download
    endpoint = "dataset/1"
    model_cache_dir = os.path.join(isolated_model_cache, endpoint)

    model = mock.Mock()
    model.endpoint = endpoint
    model.cache_dir = model_cache_dir
    model.version_id = "1"
    model.api_key = "key"
    model.device_id = "device"
    model.weights_file = "weights.onnx"
    model.get_all_required_infer_bucket_file.return_value = ["weights.onnx"]

    # when
    with mock.patch.object(
        roboflow, "get_roboflow_model_data", side_effect=ModelArtefactError("boom")
    ):
        with pytest.raises(ModelArtefactError):
            roboflow.RoboflowInferenceModel.download_model_artifacts_from_roboflow_api(
                model
            )

    # then - the lock is free for the next attempt
    lock_file = get_model_download_lock_path(cache_dir=model_cache_dir)
    probe = FileLock(lock_file, timeout=1)
    probe.acquire()  # raises filelock.Timeout if the failed download leaked it
    probe.release()


def test_get_model_download_lock_path_keeps_the_legacy_lock_name(tmp_path) -> None:
    """Guard the rolling-deploy contract.

    `FileLock` only serializes processes opening the identical pathname, so this
    name must stay byte-for-byte what already-deployed versions use. Changing it
    would let an old and a new process download the same model concurrently
    during a rollout. See the helper docstring before touching this.
    """
    # given
    with mock.patch.object(roboflow, "MODEL_CACHE_DIR", str(tmp_path)):
        # when
        lock_path = get_model_download_lock_path(
            cache_dir=str(tmp_path / "dataset" / "1")
        )

    # then
    assert lock_path == os.path.join(str(tmp_path), "_file_locks", "1.lock")
    assert os.path.isdir(os.path.dirname(lock_path))
