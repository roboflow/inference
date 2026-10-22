from types import SimpleNamespace

from inference_model_manager.backends.base import detect_max_batch_size


class _AttrModel:
    def __init__(self):
        self._max_batch_size = 4


class _MethodModel:
    def max_batch_size(self):
        return 4


class _PropertyModel:
    @property
    def max_batch_size(self):
        return 16


def test_detects_underscore_attribute():
    assert detect_max_batch_size(_AttrModel()) == 4


def test_detects_property():
    assert detect_max_batch_size(_PropertyModel()) == 16


def test_callable_attribute_degrades_to_none():
    assert detect_max_batch_size(_MethodModel()) is None


def test_result_is_never_callable():
    for model in (_AttrModel(), _PropertyModel(), _MethodModel()):
        assert not callable(detect_max_batch_size(model))


def test_trt_profile_takes_precedence_over_inference_config_static_size():
    inference_config = SimpleNamespace(
        forward_pass=SimpleNamespace(static_batch_size=1)
    )
    trt_config = SimpleNamespace(static_batch_size=None, dynamic_batch_size_max=32)
    model = SimpleNamespace(_inference_config=inference_config, _trt_config=trt_config)

    assert detect_max_batch_size(model) == 32

    model._trt_config = None
    assert detect_max_batch_size(model) == 1
