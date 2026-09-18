"""A hook that branches on a manifest field must handle a selector value.

Codex round-001 finding: a declaration hook that reads a literal setting gives a
COMPLETE answer, so if the field can instead hold a `$inputs.x` / `$steps.x.y`
selector the hook would be asserting something it cannot know at compile time.
Two outcomes are acceptable, and nothing else is:

* the field cannot hold a selector (a plain `Literal`), so branching on it is
  always safe - proven here by validation, not by reading the source;
* the field can hold a selector, and the hook says so with
  `incomplete_discovery([...], ["<field>_selector_unresolved:$steps.<name>"])`.

Falling back to the field's default value would be a guess and is never
acceptable.
"""

import ast
import inspect
import textwrap
from typing import Any, List, Set, Type

import pytest
from pydantic import TypeAdapter, ValidationError
from roboflow_workflows.core_steps.models.foundation.lmm.v1 import (
    BlockManifest as LMMV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.lmm_classifier.v1 import (
    BlockManifest as LMMClassifierV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.qwen_vlm.v1 import (
    BlockManifest as QwenVLMV1Manifest,
)
from roboflow_workflows.core_steps.models.foundation.qwen_vlm.v2 import (
    BlockManifest as QwenVLMV2Manifest,
)
from roboflow_workflows.core_steps.models.foundation.qwen_vlm.v3 import (
    BlockManifest as QwenVLMV3Manifest,
)
from roboflow_workflows.core_steps.models.foundation.qwen_vlm.v4 import (
    BlockManifest as QwenVLMV4Manifest,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    WorkOperation,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from roboflow_workflows.prototypes.block import WorkflowBlockManifest

MODELS_PACKAGE = "roboflow_workflows.core_steps.models"

# The two hooks this worker owns. `discover_dependent_resources` is excluded on
# purpose: its documented contract is to return a selector VERBATIM inside the
# resource metadata (with a resolver), so reading an unresolved field there is
# correct rather than a gap.
OWNED_HOOKS = ("discover_work_operations", "discover_portable_restrictions")

SELECTOR_PROBE = "$inputs.probe"

QWEN_VLM_MANIFESTS = [
    QwenVLMV1Manifest,
    QwenVLMV2Manifest,
    QwenVLMV3Manifest,
    QwenVLMV4Manifest,
]


def _field_accepts_a_selector(manifest_class: Type, field_name: str) -> bool:
    """Does the field's own annotation admit a selector string?

    The annotation is validated in isolation, so a cross-field validator on the
    manifest cannot mask the answer.
    """
    annotation = manifest_class.model_fields[field_name].annotation
    try:
        TypeAdapter(annotation).validate_python(SELECTOR_PROBE)
    except Exception:
        return False
    return True


def _fields_read_by(hook: Any) -> Set[str]:
    """Manifest fields a hook branches on (`self.name` is not a setting)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(hook)))
    read = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    return read - {"name"}


def _model_block_manifests() -> List[Type[WorkflowBlockManifest]]:
    return [
        block.manifest_class
        for block in load_workflow_blocks()
        if block.manifest_class.__module__.startswith(MODELS_PACKAGE)
    ]


@pytest.fixture(scope="module")
def model_block_manifests() -> List[Type[WorkflowBlockManifest]]:
    manifests = _model_block_manifests()
    assert manifests
    return manifests


# ---------------------------------------------------------------------------
# the registry-wide tripwire
# ---------------------------------------------------------------------------


def test_every_owned_hook_guards_the_selector_capable_fields_it_reads(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    offenders = []
    for manifest_class in model_block_manifests:
        for hook_name in OWNED_HOOKS:
            hook = getattr(manifest_class, hook_name)
            if not getattr(hook, "__module__", "").startswith(MODELS_PACKAGE):
                continue
            source = inspect.getsource(hook)
            for field_name in sorted(_fields_read_by(hook)):
                if field_name not in manifest_class.model_fields:
                    continue
                if not _field_accepts_a_selector(manifest_class, field_name):
                    continue
                if "is_workflow_selector" in source:
                    continue
                offenders.append(
                    f"{manifest_class.__module__}.{hook_name} branches on "
                    f"selector-capable `{field_name}` without a guard"
                )

    assert offenders == []


# ---------------------------------------------------------------------------
# qwen_vlm: `backend` provably cannot hold a selector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("manifest_class", QWEN_VLM_MANIFESTS)
def test_qwen_vlm_backend_cannot_hold_a_selector(manifest_class: Type) -> None:
    # then - a plain Literal: the annotation rejects the selector on its own ...
    assert _field_accepts_a_selector(manifest_class, "backend") is False

    # ... and so does building the manifest, so `discover_work_operations()`
    # can branch on it and still return a COMPLETE answer.
    with pytest.raises(ValidationError):
        manifest_class(
            type=manifest_class.model_fields["type"].annotation.__args__[0],
            name="step",
            images="$inputs.image",
            prompt="describe",
            backend=SELECTOR_PROBE,
        )


@pytest.mark.parametrize("manifest_class", QWEN_VLM_MANIFESTS)
def test_qwen_vlm_operations_stay_complete_for_both_backends(
    manifest_class: Type,
) -> None:
    for backend in ("native", "openrouter"):
        manifest = manifest_class(
            type=manifest_class.model_fields["type"].annotation.__args__[0],
            name="step",
            images="$inputs.image",
            prompt="describe",
            backend=backend,
        )

        operations = manifest.discover_work_operations()

        assert isinstance(operations, list), "a literal must yield a complete list"
        assert WorkOperation.MODEL_INFERENCE in operations
        assert (WorkOperation.EXTERNAL_REQUEST in operations) is (
            backend == "openrouter"
        )


# ---------------------------------------------------------------------------
# lmm: `lmm_type` CAN hold a selector, and the hook says so
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manifest_class, extra",
    [
        (LMMV1Manifest, {"prompt": "describe"}),
        (LMMClassifierV1Manifest, {"classes": ["cat", "dog"]}),
    ],
)
def test_lmm_type_is_selector_capable_and_never_falls_back_to_a_default(
    manifest_class: Type, extra: dict
) -> None:
    # given - the annotation admits a selector, so a complete answer would be a
    # claim the block cannot make
    assert _field_accepts_a_selector(manifest_class, "lmm_type") is True
    manifest = manifest_class(
        type=manifest_class.model_fields["type"].annotation.__args__[0],
        name="step",
        images="$inputs.image",
        lmm_type="$inputs.lmm_type",
        **extra,
    )

    # when
    operations = manifest.discover_work_operations()

    # then - explicit incompleteness, carrying only what holds for every value
    assert isinstance(operations, Discovery)
    assert operations.complete is False
    assert list(operations.items) == [WorkOperation.MODEL_INFERENCE]
    assert operations.unknown_reasons == ["lmm_type_selector_unresolved:$steps.step"]
    # and the gpt_4v answer is NOT silently reused
    assert WorkOperation.EXTERNAL_REQUEST not in operations.items


def test_the_tripwire_would_actually_catch_an_unguarded_hook() -> None:
    # A negative control: without it, the registry-wide test above could pass
    # simply because the detection never finds anything.
    class _Offender:
        def discover_work_operations(self):
            if self.model_id == "yolov8n-640":
                return [WorkOperation.MODEL_INFERENCE]
            return []

    fields = _fields_read_by(_Offender.discover_work_operations)

    assert fields == {"model_id"}
    assert "is_workflow_selector" not in inspect.getsource(
        _Offender.discover_work_operations
    )


def test_the_tripwire_reaches_the_hooks_that_branch(
    model_block_manifests: List[Type[WorkflowBlockManifest]],
) -> None:
    # The other half of the control: the registry really does contain hooks
    # that read a manifest field, so the assertion has subjects.
    branching = {
        f"{manifest_class.__module__}.{hook_name}"
        for manifest_class in model_block_manifests
        for hook_name in OWNED_HOOKS
        if _fields_read_by(getattr(manifest_class, hook_name))
    }

    assert len(branching) >= 6, branching
    assert any("qwen_vlm" in entry for entry in branching)
    assert any("lmm" in entry for entry in branching)
