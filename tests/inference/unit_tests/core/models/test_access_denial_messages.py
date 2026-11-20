import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import RoboflowAPINotAuthorizedError
from inference.core.managers.decorators import fixed_size_cache
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.models import roboflow as roboflow_models

KEY = "sentinel-private-key"

_SAM3_PACKAGE_MODULES = [
    "sam3",
    "sam3.eval",
    "sam3.eval.postprocessors",
    "sam3.model",
    "sam3.model.data_misc",
    "sam3.model.sam3_image_processor",
    "sam3.model.utils",
    "sam3.model.utils.misc",
    "sam3.train",
    "sam3.train.masks_ops",
    "sam3.train.data",
    "sam3.train.data.collator",
    "sam3.train.data.sam3_image_dataset",
    "sam3.train.transforms",
    "sam3.train.transforms.basic_for_api",
]


def _refuse_access(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "MODELS_CACHE_AUTH_ENABLED", True)
    monkeypatch.setattr(module, "OFFLINE_MODE", False)
    monkeypatch.setattr(
        module, "_check_if_api_key_has_access_to_model", lambda **kwargs: False
    )


def _assert_denial_without_key(error: pytest.ExceptionInfo, model_id: str) -> None:
    assert KEY not in str(error.value)
    assert f"does not have access to model {model_id}" in str(error.value)


@pytest.fixture()
def sam3_modules(monkeypatch):
    """Imports the SAM 3 model modules with Meta's `sam3` package stubbed.

    Modules imported here are removed again after the test, so other test
    files import them with their own stubs.
    """
    before = set(sys.modules)
    for name in _SAM3_PACKAGE_MODULES:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    import inference.models.sam3.segment_anything3 as segment_anything3
    import inference.models.sam3.visual_segmentation as visual_segmentation

    yield segment_anything3, visual_segmentation

    for name in set(sys.modules) - before:
        sys.modules.pop(name, None)


def test_fixed_size_cache_access_denial_does_not_disclose_api_key(monkeypatch):
    _refuse_access(monkeypatch, fixed_size_cache)
    manager = WithFixedSizeCache(MagicMock(), max_size=8)

    with pytest.raises(RoboflowAPINotAuthorizedError) as error:
        manager.add_model(model_id="example/1", api_key=KEY)

    _assert_denial_without_key(error, "example/1")


def test_model_artifacts_access_denial_does_not_disclose_api_key(monkeypatch):
    _refuse_access(monkeypatch, roboflow_models)
    model = SimpleNamespace(api_key=KEY, endpoint="example/1")

    with pytest.raises(RoboflowAPINotAuthorizedError) as error:
        roboflow_models.RoboflowInferenceModel.get_model_artifacts(model)

    _assert_denial_without_key(error, "example/1")


def test_core_model_weights_access_denial_does_not_disclose_api_key(monkeypatch):
    _refuse_access(monkeypatch, roboflow_models)
    model = SimpleNamespace(
        api_key=KEY,
        endpoint="clip/1",
        countinference=None,
        service_secret=None,
    )

    with pytest.raises(RoboflowAPINotAuthorizedError) as error:
        roboflow_models.RoboflowCoreModel.download_weights(model)

    _assert_denial_without_key(error, "clip/1")


@pytest.mark.parametrize(
    "module_index, class_name",
    [(0, "SegmentAnything3"), (1, "Sam3ForInteractiveImageSegmentation")],
)
def test_sam3_weights_access_denial_does_not_disclose_api_key(
    monkeypatch, sam3_modules, module_index, class_name
):
    module = sam3_modules[module_index]
    _refuse_access(monkeypatch, module)
    model = SimpleNamespace(
        api_key=KEY,
        endpoint="sam3/sam3_final",
        countinference=None,
        service_secret=None,
        get_infer_bucket_file_list=lambda: [],
        _is_core_sam3_endpoint=lambda: True,
    )

    with pytest.raises(RoboflowAPINotAuthorizedError) as error:
        getattr(module, class_name).download_weights(model)

    _assert_denial_without_key(error, "sam3/sam3_final")
