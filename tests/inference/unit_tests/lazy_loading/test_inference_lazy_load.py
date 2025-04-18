import sys
import pytest


def test_lazy_loading_success(monkeypatch):
    """Test that attributes are loaded on demand and only when needed."""
    for mod_name in [
        "inference",
        "inference.core.interfaces.stream.stream",
        "inference.core.interfaces.stream.inference_pipeline",
        "inference.models.utils",
    ]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import inference

    assert "inference.core.interfaces.stream.stream" not in sys.modules
    assert "inference.core.interfaces.stream.inference_pipeline" not in sys.modules
    assert "inference.models.utils" not in sys.modules
    # # Access Stream attribute - should import the stream module
    Stream = inference.Stream
    assert "inference.core.interfaces.stream.stream" in sys.modules
    assert "inference.core.interfaces.stream.inference_pipeline" not in sys.modules

    assert Stream is inference.Stream
    assert Stream is sys.modules["inference.core.interfaces.stream.stream"].Stream

    inference.InferencePipeline
    assert "inference.core.interfaces.stream.inference_pipeline" in sys.modules

    get_model = inference.get_model
    utils_module = "inference.models.utils"
    assert utils_module in sys.modules
    assert get_model is inference.get_model
    assert get_model is sys.modules[utils_module].get_model

    get_roboflow_model = inference.get_roboflow_model
    assert get_roboflow_model is inference.get_roboflow_model
    assert get_roboflow_model is sys.modules[utils_module].get_roboflow_model


def test_lazy_loading_fail():
    """Test that AttributeError is raised for nonexistent attributes."""
    import inference

    with pytest.raises(
        AttributeError, match="module 'inference' has no attribute 'nonexistent'"
    ):
        inference.nonexistent


def test_different_import_order(monkeypatch):
    """Test that importing attributes in different orders works correctly."""
    for mod_name in [
        "inference",
        "inference.core.interfaces.stream.stream",
        "inference.core.interfaces.stream.inference_pipeline",
        "inference.models.utils",
    ]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import inference

    order1_get_model = inference.get_model
    order1_stream = inference.Stream

    for mod_name in [
        "inference",
        "inference.core.interfaces.stream.stream",
        "inference.core.interfaces.stream.inference_pipeline",
        "inference.models.utils",
    ]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import inference

    order2_stream = inference.Stream
    order2_get_model = inference.get_model

    assert order1_get_model.__module__ == order2_get_model.__module__
    assert order1_stream.__name__ == order2_stream.__name__


def test_import_from_function():
    """Test the _import_from helper function."""
    from inference import _import_from

    result = _import_from("sys", "version")
    assert result is sys.version

    # Test with nonexistent attribute
    with pytest.raises(AttributeError):
        _import_from("sys", "nonexistent_attribute")

    # Test with nonexistent module
    with pytest.raises(ImportError):
        _import_from("nonexistent_module", "attribute")


def test_import_model_util():
    """Test the _import_model_util helper function."""
    from inference import _import_model_util

    get_model = _import_model_util("get_model")
    get_roboflow_model = _import_model_util("get_roboflow_model")

    from inference.models.utils import get_model as direct_get_model
    from inference.models.utils import get_roboflow_model as direct_get_roboflow_model

    assert get_model is direct_get_model
    assert get_roboflow_model is direct_get_roboflow_model

    with pytest.raises(KeyError):
        _import_model_util("nonexistent_function")


def test_getattr_implementation(monkeypatch):
    """Test that the __getattr__ implementation is being called correctly."""
    for mod_name in [
        "inference",
        "inference.core.interfaces.stream.stream",
    ]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    import inference

    original_getattr = inference.__getattr__
    getattr_calls = []

    def tracking_getattr(name):
        getattr_calls.append(name)
        return original_getattr(name)

    monkeypatch.setattr(inference, "__getattr__", tracking_getattr)

    _ = inference.Stream
    _ = getattr(inference, "InferencePipeline")

    assert "Stream" in getattr_calls
    assert "InferencePipeline" in getattr_calls

    with pytest.raises(AttributeError):
        _ = inference.nonexistent_attribute
