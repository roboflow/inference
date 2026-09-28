"""Loader argument regression checks with real execution-plan construction."""

import warnings
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from inference_models.models.rfdetr.optimization.execution_plan import (
    RFDetrExecutionPlan,
)

from .test_backend_execution_plan import config


@pytest.fixture(params=["torch-package", "torch-checkpoint", "torch-file", "onnx"])
def _loader(request, monkeypatch):
    """Stub weights/session I/O while preserving loaders and model constructors."""
    if request.param == "onnx":
        pytest.importorskip("onnxruntime")
        from inference_models.models.rfdetr import (
            rfdetr_object_detection_onnx as adapter,
        )

        session = Mock()
        session.get_inputs.return_value = [
            SimpleNamespace(shape=[1, 3, 32, 32], name="images")
        ]
        monkeypatch.setattr(
            adapter.onnxruntime, "InferenceSession", Mock(return_value=session)
        )
        monkeypatch.setattr(
            adapter,
            "align_device_with_onnx_session",
            lambda **kwargs: torch.device("cpu"),
        )
        monkeypatch.setattr(
            adapter,
            "set_onnx_execution_provider_defaults",
            lambda **kwargs: kwargs["providers"],
        )
        loader = partial(
            adapter.RFDetrForObjectDetectionONNX.from_pretrained,
            "unused-package",
            device=torch.device("cpu"),
            onnx_execution_providers=["CPUExecutionProvider"],
        )
    else:
        from inference_models.models.rfdetr import (
            rfdetr_object_detection_pytorch as adapter,
        )

        monkeypatch.setattr(
            adapter.torch,
            "load",
            Mock(return_value={"model": {"class_embed.bias": torch.zeros(2)}}),
        )
        monkeypatch.setattr(adapter, "build_model", Mock())
        monkeypatch.setattr(
            adapter, "parse_model_type", Mock(return_value="rfdetr-small")
        )
        monkeypatch.setitem(
            adapter.CONFIG_FOR_MODEL_TYPE,
            "rfdetr-small",
            lambda **kwargs: SimpleNamespace(
                num_windows=1, patch_size=16, resolution=32
            ),
        )
        monkeypatch.setattr(
            adapter.os.path, "isfile", lambda _: request.param == "torch-file"
        )
        method = (
            adapter.RFDetrForObjectDetectionTorch.from_checkpoint_file
            if request.param == "torch-checkpoint"
            else adapter.RFDetrForObjectDetectionTorch.from_pretrained
        )
        loader = partial(
            method,
            "unused-package",
            device=torch.device("cpu"),
            model_type="rfdetr-small",
            labels=["cat", "background"],
        )

    monkeypatch.setattr(
        adapter,
        "get_model_package_contents",
        lambda **kwargs: {name: name for name in kwargs["elements"]},
    )
    monkeypatch.setattr(adapter, "parse_class_names_file", Mock(return_value=["cat"]))
    monkeypatch.setattr(adapter, "parse_inference_config", Mock(return_value=config()))

    return loader


def _strict_plan():
    plan = RFDetrExecutionPlan(
        preprocessor_id="base",
        postprocessor_id="base",
        allow_compatibility_fallback=False,
        allow_runtime_failure_fallback=False,
    )

    return plan


@pytest.mark.parametrize("serialized", [False, True])
def test_canonical_plan_overrides_environment_and_preserves_policies(
    _loader, monkeypatch, serialized
):
    """Honor explicit reference-only plans at each real loader boundary.

    Args:
        _loader (Callable): Loader with dependency I/O replaced.
        monkeypatch (pytest.MonkeyPatch): Environment replacement fixture.
        serialized (bool): Whether to pass the canonical mapping instead of a plan.
    """
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_PREPROCESSOR", "unknown-environment-choice"
    )
    monkeypatch.setenv(
        "INFERENCE_MODELS_RFDETR_POSTPROCESSOR", "unknown-environment-choice"
    )
    plan = _strict_plan()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = _loader(execution_plan=plan.to_dict() if serialized else plan)

    assert not [warning for warning in captured if warning.category is FutureWarning]
    assert model.rfdetr_execution_plan == plan
    assert model.preprocessor_implementation_id == "base"


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("explicit_none", [False, True])
def test_deprecated_alias_warns_once_and_preserves_plan(
    _loader, serialized, explicit_none
):
    """Accept the old loader keyword consistently, including checkpoint delegation.

    Args:
        _loader (Callable): Loader with dependency I/O replaced.
        serialized (bool): Whether to pass a serialized plan.
        explicit_none (bool): Whether to also supply execution_plan=None.
    """
    plan = _strict_plan()
    kwargs = {"rfdetr_execution_plan": plan.to_dict() if serialized else plan}
    if explicit_none:
        kwargs["execution_plan"] = None

    with pytest.warns(
        FutureWarning, match="October 24, 2026.*'execution_plan'"
    ) as captured:
        model = _loader(**kwargs)

    assert len(captured) == 1
    assert captured[0].filename == __file__
    assert model.rfdetr_execution_plan == plan


@pytest.mark.parametrize("legacy_plan", [None, _strict_plan()])
def test_conflicting_arguments_raise_before_loading(_loader, legacy_plan):
    """Reject both names even when the deprecated alias is None.

    Args:
        _loader (Callable): Loader with dependency I/O replaced.
        legacy_plan (RFDetrExecutionPlan | None): Conflicting deprecated argument.
    """
    with pytest.raises(TypeError, match="Cannot pass both"):
        _loader(execution_plan=_strict_plan(), rfdetr_execution_plan=legacy_plan)


@pytest.mark.parametrize(
    "kwargs", [{}, {"execution_plan": None}, {"rfdetr_execution_plan": None}]
)
def test_omitted_or_none_plan_uses_environment(_loader, monkeypatch, kwargs):
    """Preserve environment-based defaults when no explicit plan is requested.

    Args:
        _loader (Callable): Loader with dependency I/O replaced.
        monkeypatch (pytest.MonkeyPatch): Environment replacement fixture.
        kwargs (dict): Omitted, canonical None, or deprecated None argument.
    """
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_PREPROCESSOR", "base")
    monkeypatch.setenv("INFERENCE_MODELS_RFDETR_POSTPROCESSOR", "base")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = _loader(**kwargs)

    assert len(
        [warning for warning in captured if warning.category is FutureWarning]
    ) == int("rfdetr_execution_plan" in kwargs)
    assert model.rfdetr_execution_plan == RFDetrExecutionPlan(
        preprocessor_id="base", postprocessor_id="base"
    )


@pytest.mark.parametrize("argument", ["execution_plan", "rfdetr_execution_plan"])
def test_invalid_serialized_plan_is_rejected(_loader, argument):
    """Reject incomplete mappings instead of selecting an environment plan.

    Args:
        _loader (Callable): Loader with dependency I/O replaced.
        argument (str): Canonical or deprecated loader keyword.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(ValueError, match="must contain exactly"):
            _loader(**{argument: {"preprocessor": "base"}})
