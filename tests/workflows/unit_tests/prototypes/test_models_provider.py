import ast
import inspect
import textwrap
from typing import get_type_hints

from inference.core.managers.base import ModelManager
from inference.core.workflows.prototypes.models_provider import ModelsProvider

REQUIRED_METHODS = [
    "add_model",
    "run_tensor_native_inference",
    "get_class_names",
    "get_keypoints_classes",
    "model_supports_stream_pipeline",
    "get_model_pipeline_depth",
    "flush_model_stream_pipeline",
    "shutdown_model_stream_pipeline",
    "__contains__",
]


def test_protocol_declares_every_member_workflows_uses() -> None:
    for member in REQUIRED_METHODS:
        assert hasattr(ModelsProvider, member), member
    assert "content_addressed_artifact_cache" in get_type_hints(ModelsProvider)


def test_model_manager_is_structurally_compatible_with_the_port() -> None:
    # `hasattr` alone proves almost nothing - it would pass against a stub with
    # the wrong signature. Check that ModelManager can actually be CALLED the
    # way the port declares, so a signature drift in the server surfaces here
    # rather than at runtime inside a block.
    from inference.core.interfaces.workflows_models_provider import (
        ModelManagerModelsProvider,
    )

    for member in REQUIRED_METHODS:
        assert hasattr(ModelManager, member), member
        assert hasattr(ModelManagerModelsProvider, member), member
    port_add = inspect.signature(ModelsProvider.add_model)
    real_add = inspect.signature(ModelManager.add_model)
    for name in ("model_id", "api_key", "model_id_alias"):
        assert name in real_add.parameters, name
    # Everything the port passes positionally/by-name must be bindable.
    real_add.bind_partial(None, model_id="m/1", api_key="k", model_id_alias=None)
    # A default that drifts between the port and the real signature (e.g. the
    # port declaring `api_key: Optional[str] = None` while the real method
    # requires it) would still bind_partial cleanly above, so check defaults
    # explicitly.
    for name, port_param in port_add.parameters.items():
        if name in ("self", "kwargs"):
            continue
        assert port_param.default == real_add.parameters[name].default, name
    assert "kwargs" in port_add.parameters
    # The dunders are called by subscription/`in`, i.e. positionally, so a name
    # mismatch is invisible at runtime - but it is still a signature drift, and
    # anything that binds by keyword (a mock, a wrapper, a future refactor)
    # would break. Compare ordered parameter NAMES only; annotations differ
    # deliberately (the port returns `Any`, ModelManager returns `Model`).
    for member in ("__contains__",):
        port_names = [
            name
            for name in inspect.signature(getattr(ModelsProvider, member)).parameters
            if name != "self"
        ]
        real_names = [
            name
            for name in inspect.signature(getattr(ModelManager, member)).parameters
            if name != "self"
        ]
        assert port_names == real_names, (member, port_names, real_names)
    init_source = textwrap.dedent(inspect.getsource(ModelManager.__init__))
    init_tree = ast.parse(init_source)
    assigns_content_addressed_artifact_cache = any(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr == "content_addressed_artifact_cache"
        for node in ast.walk(init_tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            [node.target] if isinstance(node, ast.AnnAssign) else node.targets
        )
    )
    assert assigns_content_addressed_artifact_cache


def test_port_dropped_getitem_and_declares_first_class_model_access() -> None:
    from inference.core.managers.decorators.base import ModelManagerDecorator

    assert "__getitem__" not in ModelsProvider.__dict__
    for name in (
        "get_keypoints_classes",
        "model_supports_stream_pipeline",
        "get_model_pipeline_depth",
        "flush_model_stream_pipeline",
        "shutdown_model_stream_pipeline",
        "load_action_recognition_model",
    ):
        assert name in ModelManager.__dict__, f"{name} must be declared on ModelManager"
        # A decorator that inherits instead of forwarding hits the `_models`
        # property that raises (decorators/base.py:35-37), so the override must
        # exist in the decorator's own __dict__.
        assert (
            name in ModelManagerDecorator.__dict__
        ), f"{name} must be forwarded by ModelManagerDecorator"


def test_port_no_longer_carries_provisional_members() -> None:
    from pathlib import Path

    assert "infer_from_request_sync" not in ModelsProvider.__dict__
    assert "__getitem__" not in ModelsProvider.__dict__
    # tests/workflows/unit_tests/prototypes/<file> -> parents[4] is the repo root
    source = (
        Path(__file__).resolve().parents[4]
        / "inference/core/workflows/prototypes/models_provider.py"
    ).read_text(encoding="utf-8")
    assert "PROVISIONAL" not in source
