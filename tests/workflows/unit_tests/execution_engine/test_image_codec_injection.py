import base64

import cv2
import numpy as np
import pytest

from inference.core.workflows.core_steps.loader import KINDS_DESERIALIZERS
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.types import IMAGE_KIND
from inference.core.workflows.prototypes.image_codec import (
    WorkflowsLocalImageCodec,
    reset_image_codec,
)

BLUR_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/image_blur@v1",
            "name": "blur",
            "image": "$inputs.image",
            "blur_type": "gaussian",
            "kernel_size": 5,
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "blurred", "selector": "$steps.blur.image"}
    ],
}


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_image_codec()
    yield
    reset_image_codec()


class _RecordingCodec(WorkflowsLocalImageCodec):
    """Behaves like the refusing default, but records what it was asked to do."""

    def __init__(self):
        self.calls = []

    def decode_string(self, value, cv_imread_flags=cv2.IMREAD_COLOR):
        self.calls.append("decode_string")
        return super().decode_string(value, cv_imread_flags=cv_imread_flags)

    def fetch_url(self, value, cv_imread_flags=cv2.IMREAD_COLOR):
        self.calls.append("fetch_url")
        return super().fetch_url(value, cv_imread_flags=cv_imread_flags)


def _base64_image() -> str:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[..., 1] = 200
    return base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode("ascii")


def test_engine_uses_the_injected_codec_for_image_inputs() -> None:
    # The whole point of Path A: a codec handed to ExecutionEngine.init is what
    # deserializes that engine's runtime images.
    codec = _RecordingCodec()
    engine = ExecutionEngine.init(
        workflow_definition=BLUR_WORKFLOW,
        init_parameters={"workflows_core.image_codec": codec},
    )

    result = engine.run(runtime_parameters={"image": _base64_image()})

    assert codec.calls == ["decode_string"]
    assert result[0]["blurred"].numpy_image.shape == (16, 24, 3)


def test_injected_codec_refusal_propagates_from_a_real_run(monkeypatch) -> None:
    # Round-4 Defect 1: before Path A exists, the stock deserializer hands this
    # URL to the server loader with request_timeout=None (image_utils.py:475 ->
    # url_input.py:340), so an unguarded RED run would depend on DNS and the
    # network and could stall. DNS is faked and the socket is blocked BELOW
    # address validation (the same blocker as Task 10.5 Step 8); the assertion
    # on `transport_attempts` is what fails at RED, without any connection.
    import socket

    import urllib3.connectionpool as connectionpool

    transport_attempts = []

    def _blocked(*args, **kwargs):
        transport_attempts.append("connection")
        raise AssertionError("transport reached")

    def _fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", port),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr(connectionpool.HTTPConnectionPool, "_new_conn", _blocked)
    monkeypatch.setattr(connectionpool.HTTPSConnectionPool, "_new_conn", _blocked)

    codec = _RecordingCodec()
    engine = ExecutionEngine.init(
        workflow_definition=BLUR_WORKFLOW,
        init_parameters={"workflows_core.image_codec": codec},
    )

    with pytest.raises(Exception) as error:
        engine.run(runtime_parameters={"image": "https://cdn.example.com/i.jpg"})

    assert (
        transport_attempts == []
    ), "the run reached the transport instead of the injected codec"
    assert "URL" in str(error.value)
    assert codec.calls == ["fetch_url"]


def test_injection_does_not_leak_into_another_engine() -> None:
    # `kinds_deserializers` is served out of COMPILATION_CACHE
    # (compiler/core.py:124-128) and CompiledWorkflow is frozen, so the rebinding
    # MUST happen on a copied map. If it mutated the cached one, the second
    # engine below would inherit the first engine's codec.
    first_codec = _RecordingCodec()
    first = ExecutionEngine.init(
        workflow_definition=BLUR_WORKFLOW,
        init_parameters={"workflows_core.image_codec": first_codec},
    )
    second = ExecutionEngine.init(
        workflow_definition=BLUR_WORKFLOW,
        init_parameters={},
    )

    first.run(runtime_parameters={"image": _base64_image()})
    second.run(runtime_parameters={"image": _base64_image()})

    assert first_codec.calls == ["decode_string"], "second engine leaked into the first"


def test_shared_registry_is_not_disturbed_by_engine_scoped_injection() -> None:
    codec = _RecordingCodec()
    ExecutionEngine.init(
        workflow_definition=BLUR_WORKFLOW,
        init_parameters={"workflows_core.image_codec": codec},
    )
    from inference.core.workflows.prototypes.image_codec import get_image_codec

    assert (
        get_image_codec() is not codec
    ), "init_parameters must not install a process-wide codec as a side effect"


def test_engine_without_the_init_parameter_keeps_the_stock_deserializer() -> None:
    # Round-2 Defect 3: `ExecutionEngine` is a wrapper that stores the versioned
    # engine on `_engine` (`execution_engine/core.py:71`); it has no
    # `_compiled_workflow` of its own.
    engine = ExecutionEngine.init(workflow_definition=BLUR_WORKFLOW, init_parameters={})
    bound = engine._engine._compiled_workflow.kinds_deserializers[IMAGE_KIND.name]
    assert bound is KINDS_DESERIALIZERS[IMAGE_KIND.name]


def test_engine_with_the_init_parameter_rebinds_only_the_image_kind() -> None:
    codec = _RecordingCodec()
    engine = ExecutionEngine.init(
        workflow_definition=BLUR_WORKFLOW,
        init_parameters={"workflows_core.image_codec": codec},
    )
    deserializers = engine._engine._compiled_workflow.kinds_deserializers
    assert deserializers[IMAGE_KIND.name] is not KINDS_DESERIALIZERS[IMAGE_KIND.name]
    assert deserializers[IMAGE_KIND.name].keywords == {"image_codec": codec}
    # Every other kind is untouched, and the cached map itself is not mutated.
    for kind, function in KINDS_DESERIALIZERS.items():
        if kind == IMAGE_KIND.name:
            continue
        assert deserializers[kind] is function


def test_a_plugin_deserializer_without_the_parameter_is_left_alone() -> None:
    # Plugins may register their own image-kind deserializer with the historic
    # 3-argument signature. Binding a keyword it does not accept would raise at
    # run time, so the engine must detect that and leave it untouched.
    from inference.core.workflows.execution_engine.v1 import core as v1_core

    def legacy_plugin_deserializer(
        parameter, value, prevent_local_images_loading=False
    ):
        return value

    rebound = v1_core._bind_image_codec_to_deserializers(
        kinds_deserializers={IMAGE_KIND.name: legacy_plugin_deserializer},
        image_codec=_RecordingCodec(),
    )
    assert rebound[IMAGE_KIND.name] is legacy_plugin_deserializer
