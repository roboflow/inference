"""Regression coverage for model-owned eligibility and request-local fallbacks."""

import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from inference_models.errors import ModelRuntimeError
from inference_models.models.optimization.contracts import (
    CompatibilityResult,
    DeviceCompatibility,
    ExecutionContext,
    InputCompatibility,
    OptimizationMetadata,
    OptimizationStage,
)
from inference_models.models.optimization.errors import RecoverableStageExecutionError
from inference_models.models.optimization.registry import ImplementationRegistry
from inference_models.models.rfdetr.optimization.preprocessor_selection import (
    PreprocessorSelector,
)


class _Stage:
    def __init__(self, name, *, fallback="base"):
        self.metadata = OptimizationMetadata(
            implementation_id=name,
            stage=OptimizationStage.PREPROCESS,
            version="1",
            target=DeviceCompatibility(device_kind="any"),
            inputs=InputCompatibility(scenarios=("*",)),
            dependencies=(),
            fallback_id=fallback,
            changes_numerics=False,
            supports_concurrency=True,
            supports_cuda_graphs=False,
        )
        self.calls = Counter()
        self.static = True
        self.model = True
        self.runtime = True

    def is_compatible(self, context):
        self.calls["static"] += 1
        return self.static

    def check_model_compatibility(self, **kwargs):
        self.calls["model"] += 1
        return self._result(self.model, "model")

    def check_request_compatibility(self, *, request, context):
        self.calls["request"] += 1
        return self._result(
            self.metadata.implementation_id not in request.reject, "request"
        )

    def check_runtime_compatibility(self, **kwargs):
        self.calls["runtime"] += 1
        return self._result(self.runtime, "runtime")

    def _result(self, supported, check):
        result = (
            CompatibilityResult.compatible()
            if supported
            else CompatibilityResult.incompatible(
                f"{self.metadata.implementation_id}: {check}"
            )
        )
        return result


def _setup(*, context=None):
    context = context or ExecutionContext(device_kind="cpu", device="cpu")
    primary = _Stage("primary", fallback="middle")
    middle = _Stage("middle")
    base = _Stage("base")
    registry = ImplementationRegistry(scope_name="test")
    for stage in (primary, middle, base):
        registry.register_factory(
            metadata=stage.metadata, factory=lambda stage=stage: stage
        )
    registry.set_auto_preferences(
        stage=OptimizationStage.PREPROCESS, implementation_ids=("primary", "middle")
    )
    selector = PreprocessorSelector(
        registry=registry,
        context=context,
        image_pre_processing=object(),
        network_input=object(),
    )
    return selector, context, primary, middle, base, registry


def _request(selector, context, primary, *, reject=()):
    request = SimpleNamespace(reject=reject)
    selection = selector.resolve_request(
        implementation=primary,
        request=request,
        context=context,
        allow_fallback=True,
    )
    selection = selector.resolve_runtime_fallback(
        selection=selection,
        request=request,
        context=context,
        allow_fallback=True,
    )
    return selection


def test_warm_primary_caches_model_checks_and_leaves_fallbacks_lazy():
    selector, context, primary, middle, base, _ = _setup()
    selected = selector.resolve_model(requested_id="auto", allow_fallback=True)
    assert selected.implementation is primary
    for _ in range(10):
        assert _request(selector, context, primary).implementation is primary
    assert primary.calls == {"static": 1, "model": 1, "request": 10, "runtime": 10}
    assert not middle.calls
    assert not base.calls


@pytest.mark.parametrize("failed_check", ["static", "model", "request", "runtime"])
def test_full_chain_revalidates_every_fallback_but_caches_fixed_checks(failed_check):
    selector, context, primary, middle, base, _ = _setup()
    reject = {"primary"}
    if failed_check == "request":
        reject.add("middle")
    else:
        setattr(middle, failed_check, False)
    for _ in range(10):
        selected = _request(selector, context, primary, reject=reject)
        assert selected.implementation is base
        assert "primary: request" in selected.fallback_reason
        assert f"middle: {failed_check}" in selected.fallback_reason
    assert middle.calls["static"] == 1
    assert middle.calls["model"] == (0 if failed_check == "static" else 1)
    assert primary.calls["model"] == base.calls["model"] == 1
    assert base.calls["request"] == base.calls["runtime"] == 10


def test_request_fallback_is_not_sticky():
    selector, context, primary, middle, _, _ = _setup()
    assert (
        _request(selector, context, primary, reject={"primary"}).implementation
        is middle
    )
    assert _request(selector, context, primary).implementation is primary
    assert primary.calls["request"] == 2


def test_runtime_failure_is_fresh_without_duplicate_request_check():
    selector, context, primary, middle, base, _ = _setup()
    assert _request(selector, context, primary).implementation is primary
    primary.runtime = middle.runtime = False
    assert _request(selector, context, primary).implementation is base
    assert primary.calls["request"] == 2
    assert middle.calls["request"] == base.calls["request"] == 1
    primary.runtime = True
    assert _request(selector, context, primary).implementation is primary


def test_target_rejection_skips_factory_before_native_loading():
    selector, context, primary, middle, base, registry = _setup(
        context=ExecutionContext(
            device_kind="gpu", device="cuda", host_architecture="aarch64"
        )
    )
    middle.metadata = replace(
        middle.metadata,
        target=DeviceCompatibility(device_kind="any", host_architectures=("x86_64",)),
    )
    # Build a fresh registry to use the changed metadata, without constructing SIMD.
    registry = ImplementationRegistry(scope_name="test")
    registry.register(primary)
    registry.register(base)
    registry.register_factory(
        metadata=middle.metadata, factory=lambda: pytest.fail("native load")
    )
    selector = PreprocessorSelector(
        registry=registry,
        context=context,
        image_pre_processing=None,
        network_input=None,
    )
    assert (
        _request(selector, context, primary, reject={"primary"}).implementation is base
    )


def test_model_incompatibility_traverses_full_chain():
    selector, _, primary, middle, base, _ = _setup()
    primary.model = middle.model = False
    selected = selector.resolve_model(requested_id="primary", allow_fallback=True)
    assert selected.implementation is base
    assert selected.fallback_reason == "primary: model; middle: model"


def test_auto_prefers_middle_to_base_and_skips_unavailable_model():
    selector, _, primary, middle, _, _ = _setup()
    primary.static = False
    assert (
        selector.resolve_model(requested_id="auto", allow_fallback=True).implementation
        is middle
    )
    selector, _, primary, middle, base, _ = _setup()
    primary.static = False
    middle.model = False
    assert (
        selector.resolve_model(requested_id="auto", allow_fallback=True).implementation
        is base
    )


def test_disabled_compatibility_fallback_does_not_construct_next_candidate():
    selector, context, primary, middle, _, _ = _setup()
    with pytest.raises(ModelRuntimeError, match="fallback is disabled"):
        selector.resolve_request(
            implementation=primary,
            request=SimpleNamespace(reject={"primary"}),
            context=context,
            allow_fallback=False,
        )
    assert not middle.calls


def test_disabled_runtime_fallback_does_not_construct_next_candidate():
    selector, context, primary, middle, _, _ = _setup()
    request = SimpleNamespace(reject=())
    selected = selector.resolve_request(
        implementation=primary,
        request=request,
        context=context,
        allow_fallback=True,
    )
    primary.runtime = False
    with pytest.raises(RecoverableStageExecutionError, match="fallback is disabled"):
        selector.resolve_runtime_fallback(
            selection=selected,
            request=request,
            context=context,
            allow_fallback=False,
        )
    assert not middle.calls


def test_invalid_cycle_and_exhausted_chain_raise():
    selector, context, primary, middle, _, registry = _setup()
    middle.metadata = replace(middle.metadata, fallback_id="primary")
    cyclic_registry = ImplementationRegistry(scope_name="test")
    cyclic_registry.register(primary)
    cyclic_registry.register(middle)
    selector = PreprocessorSelector(
        registry=cyclic_registry,
        context=context,
        image_pre_processing=None,
        network_input=None,
    )
    with pytest.raises(ModelRuntimeError, match="cycle"):
        _request(selector, context, primary, reject={"primary", "middle"})
    selector, context, primary, _, _, _ = _setup()
    with pytest.raises(ModelRuntimeError, match="Fallback 'base' is unsupported"):
        _request(selector, context, primary, reject={"primary", "middle", "base"})


def test_unknown_ids_and_factory_errors_are_not_silently_swallowed():
    selector, _, _, _, _, registry = _setup()
    with pytest.raises(ModelRuntimeError, match="Unknown"):
        selector.resolve_model(requested_id="missing", allow_fallback=True)
    broken = _Stage("broken")

    def fail():
        raise RuntimeError("constructor bug")

    registry.register_factory(metadata=broken.metadata, factory=fail)
    with pytest.raises(RuntimeError, match="constructor bug"):
        selector.resolve_model(requested_id="broken", allow_fallback=True)


def test_concurrent_cold_fallback_checks_model_once():
    selector, context, primary, middle, base, _ = _setup()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                _request, selector, context, primary, reject={"primary", "middle"}
            )
            for _ in range(40)
        ]
        assert all(future.result().implementation is base for future in futures)
    for stage in (primary, middle, base):
        assert stage.calls["model"] == stage.calls["static"] == 1
        assert stage.calls["request"] == 40


def test_cache_is_model_owned_and_does_not_retain_stream():
    selector, context, primary, _, _, registry = _setup(
        context=ExecutionContext(
            device_kind="cpu", device="cpu", current_stream=object()
        )
    )
    selector.resolve_model(requested_id="auto", allow_fallback=True)
    other = PreprocessorSelector(
        registry=registry,
        context=context,
        image_pre_processing=None,
        network_input=None,
    )
    other.resolve_model(requested_id="auto", allow_fallback=True)
    assert primary.calls["model"] == 2
    assert selector._context.current_stream is None


def test_cache_stays_bounded_and_does_not_retain_request_images():
    selector, context, primary, _, _, _ = _setup()
    image_refs = []
    for width in range(1, 100):
        image = np.zeros((2, width, 3), dtype=np.uint8)
        image_refs.append(weakref.ref(image))
        request = SimpleNamespace(reject={"primary", "middle"}, images=image)
        selection = selector.resolve_request(
            implementation=primary,
            request=request,
            context=context,
            allow_fallback=True,
        )
        selector.resolve_runtime_fallback(
            selection=selection,
            request=request,
            context=context,
            allow_fallback=True,
        )
    del image, request
    assert all(reference() is None for reference in image_refs)
    assert len(selector._candidates) == 3
