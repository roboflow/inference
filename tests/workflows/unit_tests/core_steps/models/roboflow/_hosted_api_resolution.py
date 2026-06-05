"""Shared harness for hosted-vs-non-hosted API path-resolution tests.

Not a test module (underscore prefix keeps it out of collection). Each Roboflow
model block's ``run_remotely`` must, against the hosted target, talk to its
dedicated hosted URL over the v0 API (``select_api_v0``); against any other
target it must use ``LOCAL_INFERENCE_API_URL`` and not force v0. This regression
guard exists because the hosted v1 endpoints were retired in favour of the
serverless backend, which only serves the v0 / path-parameter API.

Block families and versions are enumerated explicitly in each per-family test
file (``BLOCK_CLASSES``).
"""

import importlib
import inspect
from unittest.mock import MagicMock, patch

import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode

_MODULE_PREFIX = "inference.core.workflows.core_steps.models.roboflow"


class _StopAfterClientSetup(Exception):
    """Raised by the mocked ``client.infer`` to halt ``run_remotely`` once the
    API selection has happened, before any prediction post-processing runs."""


def _load_block(family: str, version: str, class_name: str):
    module = importlib.import_module(f"{_MODULE_PREFIX}.{family}.{version}")
    return module, getattr(module, class_name)


def _make_block(block_cls):
    return block_cls(
        model_manager=MagicMock(),
        api_key="test-key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )


def _run_remotely_kwargs(method) -> dict:
    # run_remotely signatures vary per family/version; fill model-tuning params
    # with None (InferenceConfiguration is mocked, so values don't affect path
    # resolution, which depends only on WORKFLOWS_REMOTE_API_TARGET).
    kwargs = {}
    for name in inspect.signature(method).parameters:
        if name == "images":
            kwargs[name] = [MagicMock()]
        elif name == "model_id":
            kwargs[name] = "workspace/model/1"
        else:
            kwargs[name] = None
    return kwargs


def _invoke_run_remotely(module, block_cls, remote_target: str):
    block = _make_block(block_cls)
    kwargs = _run_remotely_kwargs(block.run_remotely)
    with patch.object(module, "InferenceHTTPClient") as client_cls, patch.object(
        module, "InferenceConfiguration"
    ), patch.object(module, "WORKFLOWS_REMOTE_API_TARGET", remote_target):
        client = client_cls.return_value
        client.infer.side_effect = _StopAfterClientSetup
        with pytest.raises(_StopAfterClientSetup):
            block.run_remotely(**kwargs)
    return client_cls, client


def assert_hosted_selects_v0(
    family: str, version: str, class_name: str, hosted_url_attr: str
):
    module, block_cls = _load_block(family, version, class_name)
    client_cls, client = _invoke_run_remotely(module, block_cls, "hosted")
    assert client_cls.call_args.kwargs["api_url"] == getattr(module, hosted_url_attr)
    client.select_api_v0.assert_called_once()


def assert_non_hosted_uses_local(family: str, version: str, class_name: str):
    module, block_cls = _load_block(family, version, class_name)
    client_cls, client = _invoke_run_remotely(module, block_cls, "self-hosted")
    assert client_cls.call_args.kwargs["api_url"] == getattr(
        module, "LOCAL_INFERENCE_API_URL"
    )
    client.select_api_v0.assert_not_called()
