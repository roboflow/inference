"""WP-A00 public-contract freeze for the stream package's user-facing surface.

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

# Five levels up from tests/inference/unit_tests/core/interfaces/ is the repo
# root - used as the subprocess's cwd so `import inference` resolves the
# local checkout the same way the harness's own PYTHONPATH-driven pytest
# invocation does.
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_PROBE_SCRIPT = Path(__file__).resolve().parent / "_stream_contract_probe.py"

# Running the probe as `python _stream_contract_probe.py` would put its own
# directory (this one) first on sys.path - which shadows the stdlib `http`
# package with the sibling `http/` fixture directory used elsewhere in this
# test suite. `runpy.run_path` executes the file without that insertion, so
# stdlib imports inside `inference` (e.g. `torch` -> `urllib.request` ->
# `http.client`) keep resolving to the real stdlib.
_RUN_PROBE_SNIPPET = (
    "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')"
)

# Every environment variable a frozen contract's literal defaults are bound
# from (see the module docstring). Removed from the child environment before
# every capture, then `env_overrides` is layered back on top for the cases
# that deliberately probe a nondefault value.
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


# (qualified name, expected signature hash, expected docstring hash) - the
# live signatures/docstrings are captured by `_stream_contract_probe.py`,
# never imported directly into this process. Hashes are sha256, truncated to
# 16 hex chars - collision risk is irrelevant here, this is a change-detector,
# not a security control.
_FROZEN_CONTRACTS = [
    (
        "InferencePipeline.init",
        "627d57d8713d9b42",
        "374e25e04372baa3",
    ),
    (
        "InferencePipeline.init_with_yolo_world",
        "c8170bd0ad38cb20",
        "56febab383103087",
    ),
    (
        "InferencePipeline.init_with_workflow",
        "760611c783a67072",
        "0ab9a69949b8cd59",
    ),
    (
        "InferencePipeline.init_with_custom_logic",
        "a93da889f006ec31",
        "f1af5463774d8132",
    ),
    ("Stream.__init__", "ad88b06bfd0ef5aa", "a3785ad7cc10ce9a"),
    (
        "sinks.display_image",
        "f8e554f57455543c",
        "e3b0c44298fc1c14",
    ),
    ("sinks.render_boxes", "09f7d31d9d2f5569", "2c90c3fea73de025"),
    (
        "sinks.render_statistics",
        "12aa6eb0579b10d5",
        "e3b0c44298fc1c14",
    ),
    ("sinks.multi_sink", "c989ebaff3f51249", "e4fee59b9518c801"),
    # WP-A02 (plan §4.D) replaced the concrete ActiveLearningMiddleware
    # annotation of `active_learning_middleware` with the structural
    # `sinks.ActiveLearningBatchRegistrar`; the baseline hash was
    # "bd6d542bf3dcee6e". Everything else about the signature is pinned
    # separately by test_active_learning_sink_signature_shape_is_unchanged.
    (
        "sinks.active_learning_sink",
        "66435a496fcd65c7",
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
        f"{name} signature changed - update EXTRACT_INFERENCE_PIPELINE_AND_"
        f"MANAGER_PLAN.MD if intentional, then refreeze this hash:\n"
        f"{captured['signature']}"
    )
    assert captured["doc_hash"] == expected_doc_hash, (
        f"{name} docstring changed - refreeze this hash if intentional "
        f"(actual hash: {captured['doc_hash']})"
    )


def test_canonical_environment_yields_documented_defaults(
    canonical_contracts: dict,
) -> None:
    # Absence of every isolated variable must reproduce the documented
    # defaults (plan P5): queue 512, buffer 64, tensor mode off, and the
    # queue size not marked explicit.
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
            # Tensor mode's implicit buffer: 8, not the numpy-path default of
            # 64, when VIDEO_SOURCE_BUFFER_SIZE is absent (env.py:1763).
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
            # Explicit even though it equals the canonical default - the
            # explicit flag records whether the host set it, not whether the
            # value differs (configuration.py's `predictions_queue_size_explicit`).
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
    # A configured API key changes `Stream.__init__`'s literal default - but
    # only a synthetic sentinel is ever used here; the real parent-process
    # key (if any) is always stripped before the child starts
    # (`_ISOLATED_ENV_KEYS`), so it can never reach the subprocess.
    sentinel = "sentinel-do-not-use-0f2c9b"

    captured = _capture_contracts({"ROBOFLOW_API_KEY": sentinel})
    signature = captured["contracts"]["Stream.__init__"]["signature"]

    # Checked as a plain bool, not inline in the `assert` expression: pytest's
    # assertion rewriting reintrospects the operands of an `in` comparison and
    # would print the raw (unredacted) signature/sentinel on failure even
    # though the explicit message here is redacted.
    found = f"api_key: <class 'str'> = {sentinel!r}" in signature
    assert found, _redact_api_key(signature, sentinel)


def test_redact_api_key_scrubs_configured_secret() -> None:
    secret = "super-secret-roboflow-key"

    assert _redact_api_key(f"api_key: str = {secret!r}", secret) == (
        "api_key: str = '<redacted-api-key>'"
    )
    assert _redact_api_key("no secret present", secret) == "no secret present"


def test_active_learning_sink_signature_shape_is_unchanged() -> None:
    # The one intended A02 change is the middleware annotation; parameter
    # names, kinds, defaults, the other annotations and the return annotation
    # must still be exactly the baseline's.
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
