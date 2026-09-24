"""WP-A00 public-contract freeze for the stream package's user-facing surface.

`InferencePipeline.init*`, `Stream.__init__`, and the `sinks` module are the
call paths external callers actually use; a future extraction that
accidentally changes a default, drops a parameter, or edits the published
docs for one of them would otherwise go unnoticed by the decontamination
tests (which only check imports, not the contract). This freezes each
callable's `inspect.signature()` string and docstring by hash - the
docstrings are large, and hashing avoids dumping multi-KB fixture text into
this file while still failing loudly (with the actual value) on any drift.
"""

import hashlib
import inspect
from enum import Enum
from typing import Callable, Optional

import pytest

from inference.core.interfaces.stream import sinks
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.stream import Stream

# (qualified name, callable, expected signature hash, expected docstring hash)
# Hashes are sha256, truncated to 16 hex chars - collision risk is irrelevant
# here, this is a change-detector, not a security control.
_FROZEN_CONTRACTS = [
    (
        "InferencePipeline.init",
        InferencePipeline.init,
        "627d57d8713d9b42",
        "374e25e04372baa3",
    ),
    (
        "InferencePipeline.init_with_yolo_world",
        InferencePipeline.init_with_yolo_world,
        "c8170bd0ad38cb20",
        "56febab383103087",
    ),
    (
        "InferencePipeline.init_with_workflow",
        InferencePipeline.init_with_workflow,
        "760611c783a67072",
        "0ab9a69949b8cd59",
    ),
    (
        "InferencePipeline.init_with_custom_logic",
        InferencePipeline.init_with_custom_logic,
        "a93da889f006ec31",
        "f1af5463774d8132",
    ),
    ("Stream.__init__", Stream.__init__, "ad88b06bfd0ef5aa", "a3785ad7cc10ce9a"),
    (
        "sinks.display_image",
        sinks.display_image,
        "f8e554f57455543c",
        "e3b0c44298fc1c14",
    ),
    ("sinks.render_boxes", sinks.render_boxes, "09f7d31d9d2f5569", "2c90c3fea73de025"),
    (
        "sinks.render_statistics",
        sinks.render_statistics,
        "12aa6eb0579b10d5",
        "e3b0c44298fc1c14",
    ),
    ("sinks.multi_sink", sinks.multi_sink, "c989ebaff3f51249", "e4fee59b9518c801"),
    # WP-A02 (plan §4.D) replaced the concrete ActiveLearningMiddleware
    # annotation of `active_learning_middleware` with the structural
    # `sinks.ActiveLearningBatchRegistrar`; the baseline hash was
    # "bd6d542bf3dcee6e". Everything else about the signature is pinned
    # separately by test_active_learning_sink_signature_shape_is_unchanged.
    (
        "sinks.active_learning_sink",
        sinks.active_learning_sink,
        "66435a496fcd65c7",
        "6c08e7c5e5752f95",
    ),
]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


_STABLE_DEFAULT_TYPES = (type(None), bool, int, float, str, tuple, frozenset)


def _stable_default_repr(default: object, fn: Callable) -> str:
    if isinstance(default, _STABLE_DEFAULT_TYPES):
        return repr(default)
    if isinstance(default, Enum):
        return f"{type(default).__qualname__}.{default.name}"
    if callable(default):
        return f"{default.__module__}.{default.__qualname__}"
    fn_module = inspect.getmodule(fn)
    if fn_module is not None:
        for attr_name in dir(fn_module):
            if attr_name.startswith("_"):
                continue
            try:
                attr_value = getattr(fn_module, attr_name)
                if attr_value is default:
                    return f"{fn_module.__name__}.{attr_name}"
            except (AttributeError, TypeError):
                pass
    raise ValueError(
        f"Cannot serialize non-literal default {default!r} of type "
        f"{type(default).__module__}.{type(default).__qualname__} - "
        f"expected enum, callable, or module-level sentinel"
    )


def _stable_signature(fn: Callable) -> str:
    sig = inspect.signature(fn)
    parts = []
    last_kind = None
    for param in sig.parameters.values():
        if (
            last_kind is inspect.Parameter.POSITIONAL_ONLY
            and param.kind is not inspect.Parameter.POSITIONAL_ONLY
        ):
            parts.append("/")
        if param.kind is inspect.Parameter.KEYWORD_ONLY and last_kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            parts.append("*")
        piece = param.name
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            piece = f"*{piece}"
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            piece = f"**{piece}"
        if param.annotation is not inspect.Parameter.empty:
            piece += f": {param.annotation}"
        if param.default is not inspect.Parameter.empty:
            default_repr = _stable_default_repr(param.default, fn)
            piece += f" = {default_repr}"
        parts.append(piece)
        last_kind = param.kind
    if last_kind is inspect.Parameter.POSITIONAL_ONLY:
        parts.append("/")
    rendered = f"({', '.join(parts)})"
    if sig.return_annotation is not inspect.Signature.empty:
        rendered += f" -> {sig.return_annotation}"
    return rendered


@pytest.mark.parametrize(
    "name, fn, expected_sig_hash, expected_doc_hash", _FROZEN_CONTRACTS
)
def test_public_signature_and_docstring_are_frozen(
    name: str,
    fn: Callable,
    expected_sig_hash: str,
    expected_doc_hash: str,
) -> None:
    signature = _stable_signature(fn)
    docstring = inspect.getdoc(fn) or ""

    sig_hash = _hash(signature)
    doc_hash = _hash(docstring)

    assert sig_hash == expected_sig_hash, (
        f"{name} signature changed - update EXTRACT_INFERENCE_PIPELINE_AND_"
        f"MANAGER_PLAN.MD if intentional, then refreeze this hash:\n{signature}"
    )
    assert doc_hash == expected_doc_hash, (
        f"{name} docstring changed - refreeze this hash if intentional "
        f"(actual hash: {doc_hash})"
    )


def test_active_learning_sink_signature_shape_is_unchanged() -> None:
    # The one intended A02 change is the middleware annotation; parameter
    # names, kinds, defaults, the other annotations and the return annotation
    # must still be exactly the baseline's.
    import ast
    import subprocess
    from pathlib import Path

    baseline_source = subprocess.run(
        [
            "git",
            "show",
            "65ad2beaaca0825bffc2fbbe99199d3a40994324:"
            "inference/core/interfaces/stream/sinks.py",
        ],
        cwd=Path(__file__).resolve().parents[5],
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
