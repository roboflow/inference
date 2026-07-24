"""Behavioral tests for the process-wide OFFLINE_MODE startup latch."""

import importlib
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[4])
_INFERENCE_MODELS_ROOT = str(Path(_REPO_ROOT) / "inference_models")
_OFFLINE_MODE_LATCH = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
_CONTROLLED_VARIABLES = [
    "OFFLINE_MODE",
    _OFFLINE_MODE_LATCH,
    "USE_INFERENCE_MODELS",
    "MODELS_CACHE_AUTH_ENABLED",
    "ALLOW_OFFLINE_MODEL_CACHE_AUTH_BYPASS",
    "LAMBDA",
    "GCP_SERVERLESS",
    "DEDICATED_DEPLOYMENT_WORKSPACE_URL",
    "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT",
    "SAM3_EXEC_MODE",
    "SAM3_FINE_TUNED_MODELS_ENABLED",
    "WORKFLOWS_STEP_EXECUTION_MODE",
    "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE",
    "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS",
    "WEBRTC_MODAL_TOKEN_ID",
    "WEBRTC_MODAL_TOKEN_SECRET",
    "WEBRTC_MODAL_PUBLIC_STUN_SERVERS",
    "OTEL_TRACING_ENABLED",
    "OTEL_METRICS_ENABLED",
    "METRICS_ENABLED",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "YOLO_OFFLINE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_MODULES_CACHE",
    "YOLO_CONFIG_DIR",
    "VLLM_PROXY_ENABLED",
    "DISABLE_VERSION_CHECK",
    "VERSION_CHECK_MODE",
    "INFERENCE_HOME",
    "MODEL_CACHE_DIR",
    "SECURE_GATEWAY",
    "LICENSE_SERVER",
    "ELASTICACHE_ENDPOINT",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_SSL",
    "REDIS_TIMEOUT",
    "INFERENCE_WARNINGS_DISABLED",
    "PYTHONWARNINGS",
]


def _run_with_env(code: str, extra_env: dict) -> None:
    env = {**os.environ}
    for variable in _CONTROLLED_VARIABLES:
        env.pop(variable, None)
    env.update(
        {
            "DISABLE_VERSION_CHECK": "True",
            "USE_INFERENCE_MODELS": "False",
        }
    )
    env.update(extra_env)
    env["PYTHONPATH"] = (
        _INFERENCE_MODELS_ROOT
        + os.pathsep
        + _REPO_ROOT
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def _assert_multiprocessing_child_retains_offline_mode_latch(
    expected_mode: bool,
) -> None:
    """Assert a forked or spawned child cannot consume a runtime env mutation."""

    import inference
    from inference.core import env as core_env
    from inference_models import _offline as inference_models_offline
    from inference_models import configuration as inference_models_configuration

    importlib.reload(inference)
    importlib.reload(core_env)
    importlib.reload(inference_models_offline)
    importlib.reload(inference_models_configuration)

    process_state = sys.modules["_roboflow_inference_process_state"]
    assert process_state.offline_mode is expected_mode
    assert inference._LATCHED_OFFLINE_MODE is expected_mode
    assert core_env.OFFLINE_MODE is expected_mode
    assert inference_models_offline.OFFLINE_MODE is expected_mode
    assert inference_models_configuration.OFFLINE_MODE is expected_mode
    assert os.environ[_OFFLINE_MODE_LATCH] == str(expected_mode)
    assert os.environ["OFFLINE_MODE"] == str(not expected_mode)


def test_top_level_import_latches_mode_before_first_core_import() -> None:
    _run_with_env(
        """
import os
import sys

import inference

state = sys.modules["_roboflow_inference_process_state"]
assert state.offline_mode is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"

os.environ["OFFLINE_MODE"] = "True"
from inference.core import env as core_env

assert core_env.OFFLINE_MODE is False
assert state.offline_mode is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
""",
        {"OFFLINE_MODE": "False"},
    )


def test_top_level_import_rejects_offline_mode_without_dotenv_value() -> None:
    _run_with_env(
        """
import os
import sys
import types

dotenv = types.ModuleType("dotenv")
dotenv.dotenv_values = lambda _: {"OFFLINE_MODE": None}
dotenv.load_dotenv = lambda _: None
sys.modules["dotenv"] = dotenv

try:
    import inference
except ValueError as error:
    assert "Expected OFFLINE_MODE to be a boolean" in str(error)
else:
    raise AssertionError("OFFLINE_MODE without a value was accepted")

os.environ["OFFLINE_MODE"] = "False"
try:
    import inference_models
except ValueError as error:
    assert "Expected OFFLINE_MODE to be a boolean" in str(error)
else:
    raise AssertionError("A competing import bypassed the malformed startup value")
""",
        {},
    )


def test_dotenv_offline_mode_is_import_order_independent() -> None:
    for first_package, second_package in (
        ("inference_models", "inference"),
        ("inference", "inference_models"),
    ):
        _run_with_env(
            f"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temporary_directory:
    expected_cache_directory = str(Path(temporary_directory, "model-cache"))
    Path(temporary_directory, ".env").write_text(
        "OFFLINE_MODE=True\\n"
        f"MODEL_CACHE_DIR={{expected_cache_directory}}\\n"
    )
    os.chdir(temporary_directory)

    importlib.import_module({first_package!r})
    state = sys.modules["_roboflow_inference_process_state"]
    assert state.offline_mode is True
    assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "True"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["YOLO_OFFLINE"] == "True"
    assert os.environ["HF_HOME"] == str(Path(expected_cache_directory, "hf_home"))
    assert os.environ["HF_HUB_CACHE"] == str(
        Path(expected_cache_directory, "hf_home", "hub")
    )
    assert os.environ["HF_MODULES_CACHE"] == str(
        Path(expected_cache_directory, "hf_home", "modules")
    )
    assert os.environ["YOLO_CONFIG_DIR"] == str(
        Path(
            tempfile.gettempdir(),
            "roboflow-inference",
            "ultralytics",
        )
    )

    importlib.import_module({second_package!r})
    models_configuration = importlib.import_module(
        "inference_models.configuration"
    )
    from inference.core import env as core_env

    assert state.offline_mode is True
    assert models_configuration.OFFLINE_MODE is True
    assert models_configuration.INFERENCE_HOME == expected_cache_directory
    assert os.environ["MODEL_CACHE_DIR"] == expected_cache_directory
    assert core_env.OFFLINE_MODE is True
""",
            {},
        )


def test_process_environment_takes_precedence_over_dotenv_in_both_import_orders() -> (
    None
):
    for first_package, second_package in (
        ("inference_models", "inference"),
        ("inference", "inference_models"),
    ):
        _run_with_env(
            f"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temporary_directory:
    Path(temporary_directory, ".env").write_text("OFFLINE_MODE=True\\n")
    os.chdir(temporary_directory)

    importlib.import_module({first_package!r})
    importlib.import_module({second_package!r})

state = sys.modules["_roboflow_inference_process_state"]
assert state.offline_mode is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
assert "HF_HUB_OFFLINE" not in os.environ
assert "TRANSFORMERS_OFFLINE" not in os.environ
assert "YOLO_OFFLINE" not in os.environ
""",
            {"OFFLINE_MODE": "False"},
        )


def test_invalid_dotenv_offline_mode_fails_closed_in_both_import_orders() -> None:
    for dotenv_contents in (
        "OFFLINE_MODE\n",
        "OFFLINE_MODE=not-a-boolean\n",
    ):
        for first_package, second_package in (
            ("inference_models", "inference"),
            ("inference", "inference_models"),
        ):
            _run_with_env(
                f"""
import importlib
import os
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as temporary_directory:
    Path(temporary_directory, ".env").write_text({dotenv_contents!r})
    os.chdir(temporary_directory)

    try:
        importlib.import_module({first_package!r})
    except Exception as error:
        first_error = error
        assert "OFFLINE_MODE" in str(error)
    else:
        raise AssertionError("Invalid .env OFFLINE_MODE was accepted")

    os.environ["OFFLINE_MODE"] = "False"
    try:
        importlib.import_module({second_package!r})
    except Exception as error:
        assert type(error) is type(first_error)
        assert str(error) == str(first_error)
    else:
        raise AssertionError(
            "A competing import bypassed the invalid startup value"
        )
""",
                {},
            )


def test_malformed_inference_models_latch_cannot_be_bypassed_by_root() -> None:
    _run_with_env(
        """
import os

try:
    import inference_models
except Exception as error:
    first_error = error
    assert "true or false" in str(error)
else:
    raise AssertionError("Malformed OFFLINE_MODE was accepted")

os.environ["OFFLINE_MODE"] = "False"
try:
    import inference
except Exception as error:
    assert type(error) is type(first_error)
    assert str(error) == str(first_error)
else:
    raise AssertionError("A competing import bypassed the malformed startup value")
""",
        {"OFFLINE_MODE": "not-a-boolean"},
    )


def test_root_reserves_process_latch_before_reading_startup_environment() -> None:
    _run_with_env(
        """
import _thread
import os
import sys
import threading

root_snapshot_started = threading.Event()
resume_root_snapshot = threading.Event()
models_lock_attempted = threading.Event()
models_snapshot_attempted = threading.Event()
original_environ_get = type(os.environ).get
original_rlock = _thread.RLock
root_import_thread = None
models_import_thread = None
root_snapshot_intercepted = False

class TrackingRLock:
    def __init__(self):
        self._lock = original_rlock()

    def acquire(self, *args, **kwargs):
        if threading.current_thread() is models_import_thread:
            models_lock_attempted.set()
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        return self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

def tracking_rlock():
    return TrackingRLock()

def blocking_environ_get(self, key, default=None):
    global root_snapshot_intercepted
    if (
        key == "OFFLINE_MODE"
        and threading.current_thread() is root_import_thread
        and not root_snapshot_intercepted
    ):
        root_snapshot_intercepted = True
        startup_value = original_environ_get(self, key, default)
        root_snapshot_started.set()
        assert resume_root_snapshot.wait(timeout=30)
        return startup_value
    if (
        key == "OFFLINE_MODE"
        and threading.current_thread() is models_import_thread
    ):
        models_snapshot_attempted.set()
    return original_environ_get(self, key, default)

_thread.RLock = tracking_rlock
type(os.environ).get = blocking_environ_get
root_import_error = []
models_import_error = []

def import_inference():
    try:
        import inference
    except BaseException as error:
        root_import_error.append(error)

def import_inference_models():
    try:
        import inference_models
    except BaseException as error:
        models_import_error.append(error)

root_import_thread = threading.Thread(target=import_inference)
root_import_thread.start()
assert root_snapshot_started.wait(timeout=30)

os.environ["OFFLINE_MODE"] = "True"
models_import_thread = threading.Thread(target=import_inference_models)
models_import_thread.start()
assert models_lock_attempted.wait(timeout=30)
assert not models_snapshot_attempted.is_set()

resume_root_snapshot.set()
root_import_thread.join(timeout=90)
models_import_thread.join(timeout=90)

assert not root_import_thread.is_alive()
assert not models_import_thread.is_alive()
assert not root_import_error, root_import_error
assert not models_import_error, models_import_error
state = sys.modules["_roboflow_inference_process_state"]
assert state.offline_mode_startup_snapshot == ("False", None)
assert state.offline_mode is False
assert sys.modules["inference_models.configuration"].OFFLINE_MODE is False
""",
        {"OFFLINE_MODE": "False"},
    )


def test_inference_models_first_contender_controls_competing_inference_import() -> None:
    _run_with_env(
        """
import importlib.metadata
import os
import sys
import threading

metadata_lookup_started = threading.Event()
resume_metadata_lookup = threading.Event()
original_version = importlib.metadata.version

def blocking_version(*args, **kwargs):
    metadata_lookup_started.set()
    assert resume_metadata_lookup.wait(timeout=30)
    return original_version(*args, **kwargs)

importlib.metadata.version = blocking_version
import_error = []

def import_inference_models():
    try:
        import inference_models
    except BaseException as error:
        import_error.append(error)

import_thread = threading.Thread(target=import_inference_models)
import_thread.start()
assert metadata_lookup_started.wait(timeout=30)
os.environ["OFFLINE_MODE"] = "True"
inference_import_error = []

def import_inference():
    try:
        import inference
    except BaseException as error:
        inference_import_error.append(error)

inference_import_thread = threading.Thread(target=import_inference)
inference_import_thread.start()
inference_import_thread.join(timeout=30)
assert not inference_import_thread.is_alive()
assert not inference_import_error, inference_import_error
resume_metadata_lookup.set()
import_thread.join(timeout=90)

assert not import_thread.is_alive()
assert not import_error, import_error
state = sys.modules["_roboflow_inference_process_state"]
assert state.offline_mode is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
assert sys.modules["inference_models.configuration"].OFFLINE_MODE is False
""",
        {"OFFLINE_MODE": "False"},
    )


def test_inference_first_contender_controls_competing_inference_models_import() -> None:
    _run_with_env(
        """
import importlib.metadata
import os
import sys
import tempfile
import threading
from pathlib import Path

import dotenv

with tempfile.TemporaryDirectory() as temporary_directory:
    Path(temporary_directory, ".env").write_text("OFFLINE_MODE=False\\n")
    os.chdir(temporary_directory)

    dotenv_load_started = threading.Event()
    resume_dotenv_load = threading.Event()
    metadata_lookup_started = threading.Event()
    original_load_dotenv = dotenv.load_dotenv
    original_version = importlib.metadata.version

    def blocking_load_dotenv(*args, **kwargs):
        dotenv_load_started.set()
        assert resume_dotenv_load.wait(timeout=30)
        return original_load_dotenv(*args, **kwargs)

    def tracking_version(*args, **kwargs):
        metadata_lookup_started.set()
        return original_version(*args, **kwargs)

    dotenv.load_dotenv = blocking_load_dotenv
    importlib.metadata.version = tracking_version
    import_error = []

    def import_inference():
        try:
            import inference
        except BaseException as error:
            import_error.append(error)

    import_thread = threading.Thread(target=import_inference)
    import_thread.start()
    assert dotenv_load_started.wait(timeout=30)
    os.environ["OFFLINE_MODE"] = "True"
    inference_models_import_error = []
    inference_models_import_started = threading.Event()

    def import_inference_models():
        inference_models_import_started.set()
        try:
            import inference_models
        except BaseException as error:
            inference_models_import_error.append(error)

    inference_models_import_thread = threading.Thread(
        target=import_inference_models
    )
    inference_models_import_thread.start()
    assert inference_models_import_started.wait(timeout=30)
    assert not metadata_lookup_started.wait(timeout=0.5)
    resume_dotenv_load.set()
    import_thread.join(timeout=90)
    inference_models_import_thread.join(timeout=90)

assert not import_thread.is_alive()
assert not inference_models_import_thread.is_alive()
assert not import_error, import_error
assert not inference_models_import_error, inference_models_import_error
state = sys.modules["_roboflow_inference_process_state"]
assert state.offline_mode is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
""",
        {},
    )


def test_reloading_inference_models_helper_preserves_process_latch() -> None:
    code = """
import importlib
import os

from inference_models import _offline
from inference_models import configuration

expected_mode = os.environ["OFFLINE_MODE"] == "True"
assert _offline.OFFLINE_MODE is expected_mode
assert configuration.OFFLINE_MODE is expected_mode

opposite_mode = str(not expected_mode)
os.environ["OFFLINE_MODE"] = opposite_mode
os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] = opposite_mode
importlib.reload(_offline)
importlib.reload(configuration)

assert _offline.OFFLINE_MODE is expected_mode
assert configuration.OFFLINE_MODE is expected_mode
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == str(
    expected_mode
)
"""
    _run_with_env(code, {"OFFLINE_MODE": "False"})
    _run_with_env(code, {"OFFLINE_MODE": "True"})


def test_isolated_usage_payload_helper_rejects_malformed_offline_mode() -> None:
    payload_helpers_path = str(
        Path(_REPO_ROOT) / "inference" / "usage_tracking" / "payload_helpers.py"
    )
    _run_with_env(
        f"""
import runpy
import sys

assert "inference" not in sys.modules
try:
    runpy.run_path({payload_helpers_path!r})
except ValueError as error:
    assert "Expected OFFLINE_MODE to be a boolean" in str(error)
else:
    raise AssertionError("Malformed OFFLINE_MODE was accepted")
assert "inference" not in sys.modules
""",
        {"OFFLINE_MODE": "not-a-boolean"},
    )


def test_isolated_usage_payload_helper_prefers_shared_process_latch() -> None:
    payload_helpers_path = str(
        Path(_REPO_ROOT) / "inference" / "usage_tracking" / "payload_helpers.py"
    )
    _run_with_env(
        f"""
import os
import runpy
import sys
import types

assert "inference" not in sys.modules
state = types.ModuleType("_roboflow_inference_process_state")
state.offline_mode = False
sys.modules[state.__name__] = state
module_globals = runpy.run_path({payload_helpers_path!r})
assert module_globals["OFFLINE_MODE"] is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
assert "inference" not in sys.modules
""",
        {
            "OFFLINE_MODE": "not-a-boolean",
            _OFFLINE_MODE_LATCH: "also-not-a-boolean",
        },
    )


def test_isolated_usage_payload_helper_latches_reexecution_and_descendants() -> None:
    payload_helpers_path = str(
        Path(_REPO_ROOT) / "inference" / "usage_tracking" / "payload_helpers.py"
    )
    _run_with_env(
        f"""
import os
import runpy
import subprocess
import sys

assert "inference" not in sys.modules
first_globals = runpy.run_path({payload_helpers_path!r})
assert first_globals["OFFLINE_MODE"] is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"

os.environ["OFFLINE_MODE"] = "True"
second_globals = runpy.run_path({payload_helpers_path!r})
assert second_globals["OFFLINE_MODE"] is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"

child = subprocess.run(
    [
        sys.executable,
        "-c",
        '''
import os
import runpy

module_globals = runpy.run_path({payload_helpers_path!r})
assert module_globals["OFFLINE_MODE"] is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
''',
    ],
    env=os.environ.copy(),
    capture_output=True,
    text=True,
    timeout=60,
)
assert child.returncode == 0, child.stderr
assert "inference" not in sys.modules
""",
        {"OFFLINE_MODE": "False"},
    )


def test_runtime_environment_change_cannot_enable_offline_mode() -> None:
    _run_with_env(
        """
import importlib
import os

from inference.core import env as core_env

assert core_env.OFFLINE_MODE is False
os.environ["OFFLINE_MODE"] = "True"
os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] = "True"
importlib.reload(core_env)
models_config = importlib.import_module("inference_models.configuration")
assert core_env.OFFLINE_MODE is False
assert models_config.OFFLINE_MODE is False
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "False"
assert "HF_HUB_OFFLINE" not in os.environ
assert "TRANSFORMERS_OFFLINE" not in os.environ
assert "YOLO_OFFLINE" not in os.environ
""",
        {
            "OFFLINE_MODE": "False",
            "DISABLE_VERSION_CHECK": "True",
        },
    )


def test_inference_root_publishes_dependency_offline_environment() -> None:
    _run_with_env(
        """
import os
import tempfile

import inference

assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
assert os.environ["YOLO_OFFLINE"] == "True"
assert os.environ["HF_HOME"] == "/tmp/cache/hf_home"
assert os.environ["HF_HUB_CACHE"] == "/tmp/cache/hf_home/hub"
assert os.environ["HF_MODULES_CACHE"] == "/tmp/cache/hf_home/modules"
assert os.environ["YOLO_CONFIG_DIR"] == os.path.join(
    tempfile.gettempdir(),
    "roboflow-inference",
    "ultralytics",
)
""",
        {"OFFLINE_MODE": "True"},
    )


def test_inference_core_env_republishes_dependency_offline_environment() -> None:
    _run_with_env(
        """
import importlib
import os

import inference
from inference.core import env as core_env

for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE"):
    os.environ.pop(variable)
importlib.reload(core_env)

assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
assert os.environ["YOLO_OFFLINE"] == "True"
""",
        {"OFFLINE_MODE": "True"},
    )


def test_inference_models_republishes_dependency_offline_environment() -> None:
    _run_with_env(
        """
import importlib
import os

import inference_models
from inference_models import _offline

for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE"):
    os.environ.pop(variable)
importlib.reload(_offline)

assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
assert os.environ["YOLO_OFFLINE"] == "True"
""",
        {"OFFLINE_MODE": "True"},
    )


def test_dependency_cache_root_is_import_order_independent_and_explicit_hf_home_wins() -> (
    None
):
    for first_package, second_package in (
        ("inference_models", "inference"),
        ("inference", "inference_models"),
    ):
        _run_with_env(
            f"""
import importlib
import os

importlib.import_module({first_package!r})
importlib.import_module({second_package!r})
assert os.environ["HF_HOME"] == "/mounted/inference-cache/hf_home"
assert os.environ["HF_HUB_CACHE"] == "/mounted/inference-cache/hf_home/hub"
assert os.environ["HF_MODULES_CACHE"] == "/mounted/inference-cache/hf_home/modules"
""",
            {
                "OFFLINE_MODE": "False",
                "MODEL_CACHE_DIR": "/mounted/inference-cache",
            },
        )
        _run_with_env(
            f"""
import importlib
import os

importlib.import_module({first_package!r})
importlib.import_module({second_package!r})
assert os.environ["HF_HOME"] == "/explicit/hugging-face-cache"
assert os.environ["HF_HUB_CACHE"] == "/explicit/hugging-face-cache/hub"
assert os.environ["HF_MODULES_CACHE"] == "/explicit/hugging-face-cache/modules"
""",
            {
                "OFFLINE_MODE": "True",
                "MODEL_CACHE_DIR": "/mounted/inference-cache",
                "HF_HOME": "/explicit/hugging-face-cache",
            },
        )


def test_explicit_hugging_face_cache_paths_win_in_both_import_orders() -> None:
    for first_package, second_package in (
        ("inference_models", "inference"),
        ("inference", "inference_models"),
    ):
        _run_with_env(
            f"""
import importlib
import os

importlib.import_module({first_package!r})
importlib.import_module({second_package!r})
assert os.environ["HF_HOME"] == "/explicit/home"
assert os.environ["HF_HUB_CACHE"] == "/explicit/hub"
assert os.environ["HF_MODULES_CACHE"] == "/explicit/modules"
""",
            {
                "OFFLINE_MODE": "False",
                "MODEL_CACHE_DIR": "/mounted/inference-cache",
                "HF_HOME": "/explicit/home",
                "HF_HUB_CACHE": "/explicit/hub",
                "HUGGINGFACE_HUB_CACHE": "/ignored/legacy-hub",
                "HF_MODULES_CACHE": "/explicit/modules",
            },
        )
        _run_with_env(
            f"""
import importlib
import os

importlib.import_module({first_package!r})
importlib.import_module({second_package!r})
assert os.environ["HF_HUB_CACHE"] == "/explicit/legacy-hub"
""",
            {
                "OFFLINE_MODE": "False",
                "HUGGINGFACE_HUB_CACHE": "/explicit/legacy-hub",
            },
        )
        _run_with_env(
            f"""
import importlib
import os

importlib.import_module({first_package!r})
importlib.import_module({second_package!r})
assert os.environ["HF_HUB_CACHE"] == "/explicit/transformers-cache"
""",
            {
                "OFFLINE_MODE": "False",
                "TRANSFORMERS_CACHE": "/explicit/transformers-cache",
            },
        )


def test_preimported_hugging_face_hub_cannot_change_loader_cache_path() -> None:
    _run_with_env(
        """
import os

from huggingface_hub import constants as hub_constants

snapshot_before_inference = hub_constants.HF_HUB_CACHE
assert snapshot_before_inference != "/mounted/inference-cache/hf_home/hub"

import inference
from inference.core import env as core_env

# huggingface_hub has already snapshotted its old default. Inference therefore
# cannot rely on changing the process environment alone; built-in repository-ID
# loaders use this explicit path.
assert hub_constants.HF_HUB_CACHE == snapshot_before_inference
assert os.environ["HF_HUB_CACHE"] == "/mounted/inference-cache/hf_home/hub"
assert core_env.HF_HUB_CACHE == "/mounted/inference-cache/hf_home/hub"
""",
        {
            "OFFLINE_MODE": "False",
            "MODEL_CACHE_DIR": "/mounted/inference-cache",
        },
    )


def test_ultralytics_runtime_config_is_outside_read_only_model_cache() -> None:
    _run_with_env(
        """
import os
import tempfile

import inference

expected = os.path.join(
    tempfile.gettempdir(),
    "roboflow-inference",
    "ultralytics",
)
assert os.environ["YOLO_CONFIG_DIR"] == expected
assert not expected.startswith("/mounted/read-only-model-cache/")
""",
        {
            "OFFLINE_MODE": "True",
            "MODEL_CACHE_DIR": "/mounted/read-only-model-cache",
        },
    )

    _run_with_env(
        """
import os

import inference

assert os.environ["YOLO_CONFIG_DIR"] == "/explicit/yolo-runtime"
""",
        {
            "OFFLINE_MODE": "True",
            "YOLO_CONFIG_DIR": "/explicit/yolo-runtime",
        },
    )


def test_offline_mode_rejects_ultralytics_imported_before_inference() -> None:
    for package_name in ("inference", "inference_models"):
        _run_with_env(
            f"""
import importlib
import sys
import types

sys.modules["ultralytics"] = types.ModuleType("ultralytics")
try:
    importlib.import_module({package_name!r})
except RuntimeError as error:
    assert "Ultralytics was imported before" in str(error)
else:
    raise AssertionError("offline mode accepted stale Ultralytics online state")
""",
            {"OFFLINE_MODE": "True"},
        )


def test_offline_mode_rejects_vllm_http_proxy_at_startup() -> None:
    for enabled_value in ("true", "1", "yes", "y", "on"):
        _run_with_env(
            """
try:
    from inference.core import env
except ValueError as error:
    assert "VLLM_PROXY_ENABLED is not supported" in str(error)
else:
    raise AssertionError("offline mode accepted an HTTP vLLM proxy")
""",
            {
                "OFFLINE_MODE": "True",
                "VLLM_PROXY_ENABLED": enabled_value,
            },
        )


def test_both_packages_share_latch_and_offline_forces_local_execution() -> None:
    _run_with_env(
        """
import importlib
import os
import sys

models_config = importlib.import_module("inference_models.configuration")
assert models_config.OFFLINE_MODE is True
assert models_config.OFFLINE_MODE_CONTRACT_VERSION >= 3
assert os.environ["YOLO_OFFLINE"] == "True"
os.environ["OFFLINE_MODE"] = "False"
os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] = "False"
from inference.core import env as core_env
import inference

assert core_env.OFFLINE_MODE is True
assert core_env.SAM3_EXEC_MODE == "local"
assert core_env.SAM3_FINE_TUNED_MODELS_ENABLED is True
assert core_env.WORKFLOWS_STEP_EXECUTION_MODE == "local"
assert core_env.WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE == "local"
assert core_env.USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS is True
assert core_env.WEBRTC_MODAL_TOKEN_ID is None
assert core_env.WEBRTC_MODAL_TOKEN_SECRET is None
assert core_env.WEBRTC_MODAL_PUBLIC_STUN_SERVERS == ""
assert core_env.OTEL_TRACING_ENABLED is False
assert core_env.OTEL_METRICS_ENABLED is False
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
assert os.environ["YOLO_OFFLINE"] == "True"

importlib.reload(inference)
importlib.reload(core_env)
assert sys.modules["_roboflow_inference_process_state"].offline_mode is True
assert core_env.OFFLINE_MODE is True
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "True"
assert os.environ["YOLO_OFFLINE"] == "True"

os.environ["OFFLINE_MODE"] = "not-a-boolean"
importlib.reload(inference)
importlib.reload(core_env)
importlib.reload(models_config)
assert core_env.OFFLINE_MODE is True
assert models_config.OFFLINE_MODE is True
assert os.environ["_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"] == "True"
assert os.environ["YOLO_OFFLINE"] == "True"
""",
        {
            "OFFLINE_MODE": "True",
            "USE_INFERENCE_MODELS": "True",
            "SAM3_EXEC_MODE": "remote",
            "WORKFLOWS_STEP_EXECUTION_MODE": "remote",
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": "modal",
            "USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS": "False",
            "WEBRTC_MODAL_TOKEN_ID": "runtime-token-id",
            "WEBRTC_MODAL_TOKEN_SECRET": "runtime-token-secret",
            "WEBRTC_MODAL_PUBLIC_STUN_SERVERS": "stun:stun.example.com:19302",
            "OTEL_TRACING_ENABLED": "True",
            "OTEL_METRICS_ENABLED": "True",
        },
    )


def test_offline_startup_rejects_incompatible_inference_models_contract() -> None:
    incompatible_contracts = [
        ("", True),
        ("configuration.OFFLINE_MODE_CONTRACT_VERSION = 1", True),
        ("configuration.OFFLINE_MODE_CONTRACT_VERSION = True", True),
        ('configuration.OFFLINE_MODE_CONTRACT_VERSION = "2"', True),
        ("configuration.OFFLINE_MODE_CONTRACT_VERSION = 2", True),
        ("configuration.OFFLINE_MODE_CONTRACT_VERSION = 3", False),
    ]
    for contract_declaration, inference_models_offline_mode in incompatible_contracts:
        _run_with_env(
            f"""
import sys
import types

import inference

inference_models = types.ModuleType("inference_models")
configuration = types.ModuleType("inference_models.configuration")
configuration.OFFLINE_MODE = {inference_models_offline_mode!r}
{contract_declaration}
inference_models.configuration = configuration
sys.modules["inference_models"] = inference_models
sys.modules["inference_models.configuration"] = configuration

try:
    from inference.core import env as core_env
except RuntimeError as error:
    assert "does not support the required process-wide OFFLINE_MODE" in str(error)
else:
    raise AssertionError(
        f"Expected incompatible inference-models contract to fail, got {{core_env!r}}"
    )
""",
            {
                "OFFLINE_MODE": "True",
                "USE_INFERENCE_MODELS": "True",
            },
        )


def test_offline_startup_accepts_inference_models_contract_v3() -> None:
    _run_with_env(
        """
import sys
import types

import inference

inference_models = types.ModuleType("inference_models")
configuration = types.ModuleType("inference_models.configuration")
configuration.OFFLINE_MODE = True
configuration.OFFLINE_MODE_CONTRACT_VERSION = 3
inference_models.configuration = configuration
sys.modules["inference_models"] = inference_models
sys.modules["inference_models.configuration"] = configuration

from inference.core import env as core_env

assert core_env.OFFLINE_MODE is True
""",
        {
            "OFFLINE_MODE": "True",
            "USE_INFERENCE_MODELS": "True",
        },
    )


def test_inference_models_contract_is_not_required_outside_new_offline_stack() -> None:
    _run_with_env(
        """
from inference.core import env as core_env

assert core_env.OFFLINE_MODE is False
assert core_env.USE_INFERENCE_MODELS is True
""",
        {
            "OFFLINE_MODE": "False",
            "USE_INFERENCE_MODELS": "True",
        },
    )
    _run_with_env(
        """
from inference.core import env as core_env

assert core_env.OFFLINE_MODE is True
assert core_env.USE_INFERENCE_MODELS is False
""",
        {
            "OFFLINE_MODE": "True",
            "USE_INFERENCE_MODELS": "False",
        },
    )


@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(),
    reason="multiprocessing spawn is not available on this platform",
)
@pytest.mark.parametrize(
    ("startup_mode", "runtime_mode"),
    [(False, True), (True, False)],
)
def test_runtime_change_is_ignored_by_multiprocessing_spawned_child(
    startup_mode,
    runtime_mode,
) -> None:
    _run_with_env(
        f"""
import multiprocessing
import os
import sys

import inference
from tests.inference.unit_tests.core.test_offline_mode_configuration import (
    _assert_multiprocessing_child_retains_offline_mode_latch,
)

assert sys.modules["_roboflow_inference_process_state"].offline_mode is {startup_mode}
assert os.environ[
    "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
] == {str(startup_mode)!r}
os.environ["OFFLINE_MODE"] = {str(runtime_mode)!r}

child = multiprocessing.get_context("spawn").Process(
    target=_assert_multiprocessing_child_retains_offline_mode_latch,
    args=({startup_mode},),
)
child.start()
child.join(timeout=60)
if child.is_alive():
    child.terminate()
    child.join(timeout=10)
    raise AssertionError("multiprocessing spawn child did not terminate")
assert child.exitcode == 0, child.exitcode
""",
        {"OFFLINE_MODE": str(startup_mode)},
    )


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="multiprocessing fork is not available on this platform",
)
@pytest.mark.parametrize(
    ("startup_mode", "runtime_mode"),
    [(False, True), (True, False)],
)
def test_runtime_change_is_ignored_by_multiprocessing_forked_child(
    startup_mode,
    runtime_mode,
) -> None:
    _run_with_env(
        f"""
import multiprocessing
import os
import sys

import inference
from tests.inference.unit_tests.core.test_offline_mode_configuration import (
    _assert_multiprocessing_child_retains_offline_mode_latch,
)

assert sys.modules["_roboflow_inference_process_state"].offline_mode is {startup_mode}
assert os.environ[
    "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
] == {str(startup_mode)!r}
os.environ["OFFLINE_MODE"] = {str(runtime_mode)!r}

child = multiprocessing.get_context("fork").Process(
    target=_assert_multiprocessing_child_retains_offline_mode_latch,
    args=({startup_mode},),
)
child.start()
child.join(timeout=60)
if child.is_alive():
    child.terminate()
    child.join(timeout=10)
    raise AssertionError("multiprocessing fork child did not terminate")
assert child.exitcode == 0, child.exitcode
""",
        {"OFFLINE_MODE": str(startup_mode)},
    )


def test_forked_child_resets_inherited_latch_lock() -> None:
    _run_with_env(
        """
import importlib
import os
import signal
import sys

import inference

if not hasattr(os, "fork"):
    raise SystemExit(0)

state = sys.modules["_roboflow_inference_process_state"]
state.lock.acquire()
pid = os.fork()
if pid == 0:
    signal.alarm(10)
    importlib.reload(inference)
    assert state.offline_mode is False
    os._exit(0)

try:
    _, status = os.waitpid(pid, 0)
finally:
    state.lock.release()

assert os.WIFEXITED(status)
assert os.WEXITSTATUS(status) == 0
""",
        {"OFFLINE_MODE": "False"},
    )


def test_offline_model_cache_auth_requires_explicit_bypass() -> None:
    _run_with_env(
        """
try:
    from inference.core import env as core_env
except ValueError as error:
    assert "ALLOW_OFFLINE_MODEL_CACHE_AUTH_BYPASS=True" in str(error)
else:
    raise AssertionError(
        f"Expected unsafe auth configuration to fail, got {core_env!r}"
    )
""",
        {
            "OFFLINE_MODE": "True",
            "MODELS_CACHE_AUTH_ENABLED": "True",
        },
    )


def test_offline_model_cache_auth_allows_explicit_single_tenant_bypass() -> None:
    _run_with_env(
        """
from inference.core import env as core_env

assert core_env.OFFLINE_MODE is True
assert core_env.MODELS_CACHE_AUTH_ENABLED is True
assert core_env.ALLOW_OFFLINE_MODEL_CACHE_AUTH_BYPASS is True
""",
        {
            "OFFLINE_MODE": "True",
            "MODELS_CACHE_AUTH_ENABLED": "True",
            "ALLOW_OFFLINE_MODEL_CACHE_AUTH_BYPASS": "True",
        },
    )


def test_offline_lambda_http_import_does_not_load_legacy_usage_module() -> None:
    _run_with_env(
        """
import sys

try:
    import inference.core.interfaces.http.http_api
except ValueError as error:
    assert "OFFLINE_MODE is not supported together with LAMBDA" in str(error)
else:
    raise AssertionError("Expected offline Lambda startup to fail closed")

assert "inference.core.usage" not in sys.modules
""",
        {
            "OFFLINE_MODE": "True",
            "LAMBDA": "True",
        },
    )


def test_direct_usage_import_does_not_discover_elasticache_offline() -> None:
    _run_with_env(
        """
import sys
import types
from unittest.mock import MagicMock

discover = MagicMock(name="discover")
elasticache_module = types.ModuleType("elasticache_auto_discovery")
elasticache_module.discover = discover
hash_client = MagicMock(name="HashClient")
pymemcache_module = types.ModuleType("pymemcache")
pymemcache_client_module = types.ModuleType("pymemcache.client")
pymemcache_hash_module = types.ModuleType("pymemcache.client.hash")
pymemcache_hash_module.HashClient = hash_client
sys.modules["elasticache_auto_discovery"] = elasticache_module
sys.modules["pymemcache"] = pymemcache_module
sys.modules["pymemcache.client"] = pymemcache_client_module
sys.modules["pymemcache.client.hash"] = pymemcache_hash_module

from inference.core import usage

discover.assert_not_called()
hash_client.assert_not_called()
assert usage.memcache_client is None
assert usage.trackUsage("endpoint", "actor") is None
""",
        {"OFFLINE_MODE": "True"},
    )
