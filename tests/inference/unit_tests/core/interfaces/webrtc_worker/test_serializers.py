import importlib

import pytest

import inference.core.interfaces.webrtc_worker.serializers as webrtc_serializers


def test_wildcard_serializer_matches_the_tensor_flag(monkeypatch):
    import inference.core.env as env
    from inference.core.workflows.core_steps.common import (
        serializers as numpy_serializers,
    )
    from inference.core.workflows.core_steps.common import (
        serializers_tensor as tensor_serializers,
    )

    expected = (
        tensor_serializers.serialize_wildcard_kind
        if env.ENABLE_TENSOR_DATA_REPRESENTATION
        else numpy_serializers.serialize_wildcard_kind
    )
    assert webrtc_serializers.serialize_wildcard_kind is expected

    flipped = not env.ENABLE_TENSOR_DATA_REPRESENTATION
    monkeypatch.setattr(env, "ENABLE_TENSOR_DATA_REPRESENTATION", flipped)
    try:
        reloaded = importlib.reload(webrtc_serializers)
        flipped_expected = (
            tensor_serializers.serialize_wildcard_kind
            if flipped
            else numpy_serializers.serialize_wildcard_kind
        )
        assert reloaded.serialize_wildcard_kind is flipped_expected
    finally:
        monkeypatch.undo()
        importlib.reload(webrtc_serializers)


def test_default_encoder_raises_on_unknown_objects_instead_of_recursing():
    from inference.core.interfaces.webrtc_worker.webrtc import default_encoder

    assert default_encoder(b"abc") == "YWJj"
    with pytest.raises(TypeError, match="Cannot serialize object"):
        default_encoder(object())
