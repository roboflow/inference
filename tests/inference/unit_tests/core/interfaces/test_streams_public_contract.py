"""Public-contract freeze for the stream package's user-facing surface.

`InferencePipeline.init*`, `Stream.__init__`, and the `sinks` module are the
call paths external callers actually use; a future extraction that
accidentally changes a default, drops a parameter, or edits the published
docs for one of them would otherwise go unnoticed by the decontamination
tests (which only check imports, not the contract). This freezes each
callable's `inspect.signature()` string and docstring by hash - the
docstrings are large, and hashing avoids dumping multi-KB fixture text into
this file while still failing loudly (with the actual value) on any drift.

Several of these signatures embed a literal default (`predictions_queue_size`,
`decoding_buffer_size`, `Stream`'s post-processing/API-key defaults) that
`inference.core.env` binds from environment variables at import time. Freezing
them by importing `inference` directly in this pytest process would make the
comparison depend on whatever the ambient shell happens to export, rather than
on an actual contract change. `_capture_contracts` sidesteps that by running
the capture in a fresh subprocess (`_stream_contract_probe.py`) whose
environment this module fully controls: `_ISOLATED_ENV_KEYS` are stripped so
the canonical capture always sees documented defaults, and the override tests
below set specific keys to prove the same signatures track real configuration
changes.
"""

import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

from inference.core.interfaces.stream import sinks

from ._stream_contract_probe import _stable_signature
from .conftest import require_git_baseline_history

# parents[5] is the repo root; used as the subprocess cwd so imports match the harness.
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_PROBE_SCRIPT = Path(__file__).resolve().parent / "_stream_contract_probe.py"

# runpy.run_path avoids a sys.path insertion that would shadow stdlib http.
_RUN_PROBE_SNIPPET = (
    "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')"
)

# Stripped from the child before capture; env_overrides layers specific ones back.
_ISOLATED_ENV_KEYS = (
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE",
    "VIDEO_SOURCE_BUFFER_SIZE",
    "ENABLE_TENSOR_DATA_REPRESENTATION",
    "ROBOFLOW_API_KEY",
    "API_KEY",
    "CLASS_AGNOSTIC_NMS",
    "CONFIDENCE",
    "ENFORCE_FPS",
    "IOU_THRESHOLD",
    "MAX_CANDIDATES",
    "MAX_DETECTIONS",
    "MODEL_ID",
    "STREAM_ID",
    "ENABLE_BYTE_TRACK",
)

# Baseline public names, frozen at the SHA test_streams_decontamination.py pins.
_INVENTORY_PATH = (
    _PROJECT_ROOT
    / "tests"
    / "inference"
    / "unit_tests"
    / "streams_compat_inventory.json"
)
_INVENTORY = json.loads(_INVENTORY_PATH.read_text())

# Dropped by decontamination; restoring them would re-import host modules.
_APPROVED_REMOVALS = {
    "inference.core.interfaces.camera.collection_policy": {"core_env"},
    "inference.core.interfaces.stream_manager.manager_app.app": {"Tuple"},
}


def _redact_api_key(text: str, secret: str) -> str:
    """Scrub a configured API-key value out of diagnostic text.

    Only ever applied to messages that might be printed on assertion or
    subprocess failure; the values a test actually asserts on are never
    redacted, so this cannot mask a real drift.
    """
    if not secret:
        return text

    return text.replace(secret, "<redacted-api-key>")


def _capture_contracts(env_overrides: Optional[Dict[str, str]] = None) -> dict:
    """Capture every frozen contract's signature/docstring in a subprocess.

    Args:
        env_overrides: Environment variables applied after `_ISOLATED_ENV_KEYS`
            are stripped from the child environment. Only ever pass a
            synthetic sentinel for an API-key override - never the parent
            process's real key, which is stripped and never forwarded.

    Returns:
        Parsed JSON from `_stream_contract_probe.py`:
        `{"contracts": {name: {"signature", "sig_hash", "doc_hash"}},
        "config": {...}}`.
    """
    child_env = os.environ.copy()
    for key in _ISOLATED_ENV_KEYS:
        child_env.pop(key, None)
    child_env.update(env_overrides or {})

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _RUN_PROBE_SNIPPET, str(_PROBE_SCRIPT)],
            env=child_env,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"contract probe subprocess timed out: {error}")

    if completed.returncode != 0:
        secret = (env_overrides or {}).get("ROBOFLOW_API_KEY") or (
            env_overrides or {}
        ).get("API_KEY", "")
        stderr = _redact_api_key(completed.stderr, secret)
        pytest.fail(f"contract probe subprocess failed:\n{stderr}")

    captured = json.loads(completed.stdout)

    return captured


# hashes are sha256[:16] - a change-detector, not a security control.
_FROZEN_CONTRACTS = [
    (
        "InferencePipeline.init",
        "0f58a2833d3d32a9",
        "845dcfd5f8738052",
    ),
    (
        "InferencePipeline.init_with_yolo_world",
        "d7f01d6301bf7b92",
        "939ff0ce0262eaa1",
    ),
    (
        "InferencePipeline.init_with_workflow",
        "3dc51ebead0b32bb",
        "0d730ff4251e66cb",
    ),
    (
        "InferencePipeline.init_with_custom_logic",
        "f6a2999c313d244f",
        "75c4342a83037a1d",
    ),
    ("Stream.__init__", "ad88b06bfd0ef5aa", "a3785ad7cc10ce9a"),
    (
        "sinks.display_image",
        "f8e554f57455543c",
        "e3b0c44298fc1c14",
    ),
    ("sinks.render_boxes", "bd087e986582eec5", "141c9f6f8746df93"),
    (
        "sinks.render_statistics",
        "12aa6eb0579b10d5",
        "e3b0c44298fc1c14",
    ),
    ("sinks.multi_sink", "f665e4f385adb432", "53ffb292a998199d"),
    # Only the middleware annotation changed; rest is pinned by the shape test below.
    (
        "sinks.active_learning_sink",
        "4c5b2c7e4b2c1f03",
        "6c08e7c5e5752f95",
    ),
]


@pytest.fixture(scope="module")
def canonical_contracts() -> dict:
    """Frozen-contract capture with every configuration-backing env var removed.

    Session-shared (module-scoped) across the parametrized frozen-contract
    assertions below, so the subprocess runs once per test session rather
    than once per contract.
    """
    return _capture_contracts()


@pytest.mark.parametrize(
    "name, expected_sig_hash, expected_doc_hash", _FROZEN_CONTRACTS
)
def test_public_signature_and_docstring_are_frozen(
    name: str,
    expected_sig_hash: str,
    expected_doc_hash: str,
    canonical_contracts: dict,
) -> None:
    captured = canonical_contracts["contracts"][name]

    assert captured["sig_hash"] == expected_sig_hash, (
        f"{name} signature changed - if the change is intentional, refreeze "
        f"this hash:\n{captured['signature']}"
    )
    assert captured["doc_hash"] == expected_doc_hash, (
        f"{name} docstring changed - refreeze this hash if intentional "
        f"(actual hash: {captured['doc_hash']})"
    )


def test_canonical_environment_yields_documented_defaults(
    canonical_contracts: dict,
) -> None:
    assert canonical_contracts["config"] == {
        "predictions_queue_size": 512,
        "predictions_queue_size_explicit": False,
        "decoding_buffer_size": 64,
        "enable_tensor_data_representation": False,
    }


_CONFIGURATION_OVERRIDE_CASES = [
    pytest.param(
        {"INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE": "7"},
        {
            "predictions_queue_size": 7,
            "predictions_queue_size_explicit": True,
            "decoding_buffer_size": 64,
            "enable_tensor_data_representation": False,
        },
        id="nondefault_queue_size",
    ),
    pytest.param(
        {"VIDEO_SOURCE_BUFFER_SIZE": "3"},
        {
            "predictions_queue_size": 512,
            "predictions_queue_size_explicit": False,
            "decoding_buffer_size": 3,
            "enable_tensor_data_representation": False,
        },
        id="nondefault_buffer_size",
    ),
    pytest.param(
        {"ENABLE_TENSOR_DATA_REPRESENTATION": "True", "USE_INFERENCE_MODELS": "True"},
        {
            "predictions_queue_size": 512,
            "predictions_queue_size_explicit": False,
            # Implicit buffer is 8, not numpy's 64, when unset (env.py:1763).
            "decoding_buffer_size": 8,
            "enable_tensor_data_representation": True,
        },
        id="tensor_mode_on_implicit_buffer",
    ),
    pytest.param(
        {"ENABLE_TENSOR_DATA_REPRESENTATION": "False", "USE_INFERENCE_MODELS": "True"},
        {
            "predictions_queue_size": 512,
            "predictions_queue_size_explicit": False,
            "decoding_buffer_size": 64,
            "enable_tensor_data_representation": False,
        },
        id="tensor_mode_off_explicit",
    ),
    pytest.param(
        {"INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE": "512"},
        {
            "predictions_queue_size": 512,
            # Records whether the host set it, not whether the value differs.
            "predictions_queue_size_explicit": True,
            "decoding_buffer_size": 64,
            "enable_tensor_data_representation": False,
        },
        id="explicit_queue_size_equal_to_default",
    ),
]


@pytest.mark.parametrize(
    "env_overrides, expected_config", _CONFIGURATION_OVERRIDE_CASES
)
def test_configuration_overrides_change_factory_defaults(
    env_overrides: Dict[str, str], expected_config: dict
) -> None:
    captured = _capture_contracts(env_overrides)

    assert captured["config"] == expected_config

    for factory_name in (
        "InferencePipeline.init",
        "InferencePipeline.init_with_yolo_world",
        "InferencePipeline.init_with_workflow",
        "InferencePipeline.init_with_custom_logic",
    ):
        signature = captured["contracts"][factory_name]["signature"]
        assert (
            "predictions_queue_size: <class 'int'> = "
            f"{expected_config['predictions_queue_size']}" in signature
        )
        assert (
            "decoding_buffer_size: <class 'int'> = "
            f"{expected_config['decoding_buffer_size']}" in signature
        )


def test_api_key_override_reflects_only_the_synthetic_sentinel() -> None:
    # Only a synthetic sentinel is used; the real API key is always stripped first.
    sentinel = "sentinel-do-not-use-0f2c9b"

    captured = _capture_contracts({"ROBOFLOW_API_KEY": sentinel})
    signature = captured["contracts"]["Stream.__init__"]["signature"]

    # A plain bool, not inline - pytest rewriting would print the raw secret on failure.
    found = f"api_key: <class 'str'> = {sentinel!r}" in signature
    assert found, _redact_api_key(signature, sentinel)


def test_redact_api_key_scrubs_configured_secret() -> None:
    secret = "super-secret-roboflow-key"

    assert _redact_api_key(f"api_key: str = {secret!r}", secret) == (
        "api_key: str = '<redacted-api-key>'"
    )
    assert _redact_api_key("no secret present", secret) == "no secret present"


def test_active_learning_sink_signature_shape_is_unchanged() -> None:
    # Only the middleware annotation is expected to differ from the baseline shape.
    import ast
    import subprocess
    from pathlib import Path

    baseline_sha = "65ad2beaaca0825bffc2fbbe99199d3a40994324"
    project_root = Path(__file__).resolve().parents[5]
    require_git_baseline_history(baseline_sha, project_root=project_root)

    baseline_source = subprocess.run(
        [
            "git",
            "show",
            f"{baseline_sha}:inference/core/interfaces/stream/sinks.py",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    current_source = inspect.getsource(sinks)

    def _shape(source: str) -> list:
        function = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "active_learning_sink"
        )
        arguments = function.args
        defaults = [None] * (
            len(arguments.args) - len(arguments.defaults)
        ) + arguments.defaults
        shape = [
            (
                argument.arg,
                (
                    None
                    if argument.arg == "active_learning_middleware"
                    else ast.unparse(argument.annotation)
                ),
                None if default is None else ast.unparse(default),
            )
            for argument, default in zip(arguments.args, defaults)
        ]
        assert not arguments.posonlyargs and not arguments.kwonlyargs
        assert arguments.vararg is None and arguments.kwarg is None
        return shape + [ast.unparse(function.returns)]

    assert _shape(current_source) == _shape(baseline_source)
    annotation = (
        inspect.signature(sinks.active_learning_sink)
        .parameters["active_learning_middleware"]
        .annotation
    )
    assert annotation is sinks.ActiveLearningBatchRegistrar


def test_stable_signature_catches_positional_only_to_normal_change() -> None:
    def before(a, /, b): ...
    def after(a, b): ...

    assert _stable_signature(before) != _stable_signature(after)


def test_stable_signature_catches_positional_or_keyword_to_keyword_only_change() -> (
    None
):
    def before(a, b): ...
    def after(a, *, b): ...

    assert _stable_signature(before) != _stable_signature(after)


def test_stable_signature_catches_return_type_change() -> None:
    def before(a) -> int: ...
    def after(a) -> str: ...

    assert _stable_signature(before) != _stable_signature(after)


def test_stable_signature_catches_dropped_return_annotation() -> None:
    def before(a) -> int: ...
    def after(a): ...

    assert _stable_signature(before) != _stable_signature(after)


def test_stable_signature_catches_var_positional_and_var_keyword_kind_changes() -> None:
    def before(a, *args): ...
    def after(a, *, args): ...

    assert _stable_signature(before) != _stable_signature(after)

    def before_kw(**kwargs): ...
    def after_kw(kwargs): ...

    assert _stable_signature(before_kw) != _stable_signature(after_kw)


def test_stable_signature_catches_enum_member_changes() -> None:
    from inference.core.interfaces.stream.inference_pipeline import SinkMode

    def before(mode: SinkMode = SinkMode.ADAPTIVE) -> None: ...
    def after(mode: SinkMode = SinkMode.SEQUENTIAL) -> None: ...

    assert _stable_signature(before) != _stable_signature(after)


def test_stable_signature_catches_callback_changes() -> None:
    def callback_a() -> None: ...
    def callback_b() -> None: ...

    def before(cb=callback_a) -> None: ...
    def after(cb=callback_b) -> None: ...

    assert _stable_signature(before) != _stable_signature(after)


@pytest.mark.parametrize("entry", _INVENTORY["modules"], ids=lambda e: e["module"])
def test_every_baseline_public_name_resolves_at_its_historical_path(
    entry: dict, stub_ultralytics_if_missing
) -> None:
    # The decontamination lint checks imports, not that historical names still resolve.
    module = importlib.import_module(entry["module"])

    expected = set(entry["public_top_level_names"]) - _APPROVED_REMOVALS.get(
        entry["module"], set()
    )
    missing = sorted(name for name in expected if not hasattr(module, name))

    assert not missing, f"{entry['module']} lost {missing}"


def test_stream_session_names_resolve_at_their_historical_path() -> None:
    # Not in the inventory (outside the three trees); its incidental names were dropped.
    import inference.core.interfaces.stream.session as session_module
    import inference.usage_tracking.stream_session as stream_session_module

    assert hasattr(stream_session_module, "mint_stream_session_id")
    assert hasattr(stream_session_module, "stream_session_id")
    assert stream_session_module.stream_session_id is session_module.stream_session_id


def test_invalid_environment_variable_error_identity_is_the_historical_object() -> None:
    # Re-export, not a facade: both paths must yield one class.
    from inference.core.exceptions import (
        InvalidEnvironmentVariableError as from_exceptions,
    )
    from inference.core.utils.environment import (
        InvalidEnvironmentVariableError as from_environment,
    )

    assert from_environment is from_exceptions


def test_stream_entities_str2bool_and_safe_env_to_type_are_intentional_copies() -> None:
    # Intentional copies: a host-neutral module must not import core.utils.
    from inference.core.interfaces.stream.entities import (
        safe_env_to_type as entities_safe_env_to_type,
    )
    from inference.core.interfaces.stream.entities import str2bool as entities_str2bool
    from inference.core.utils.environment import (
        safe_env_to_type as environment_safe_env_to_type,
    )
    from inference.core.utils.environment import str2bool as environment_str2bool

    assert entities_str2bool is not environment_str2bool
    assert entities_safe_env_to_type is not environment_safe_env_to_type


def test_stream_sinks_create_tiles_and_letterbox_image_are_intentional_copies() -> None:
    # Intentional copies, same reason as above.
    from inference.core.utils.drawing import create_tiles as drawing_create_tiles
    from inference.core.utils.preprocess import (
        letterbox_image as preprocess_letterbox_image,
    )

    assert sinks.create_tiles is not drawing_create_tiles
    assert sinks.letterbox_image is not preprocess_letterbox_image
