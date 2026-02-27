from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import InferenceModelNotFound
from inference.core.managers.base import ModelManager
from inference.core.managers.entities import ModelDescription
from inference.core.managers.model_load_collector import (
    ModelLoadCollector,
    RequestModelIds,
    model_load_info,
    request_model_ids,
)


def test_add_model_when_model_already_loaded() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": "A"}

    # when
    model_manager.add_model(model_id="some/1", api_key="some_api_key")

    # then
    assert model_manager._models["some/1"] == "A"


def test_add_model_when_model_not_loaded() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    # when
    model_manager.add_model(model_id="some/1", api_key="some_api_key")

    # then
    assert "some/1" in model_manager.models()


def test_check_for_model_when_model_loaded() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": "A"}

    # when
    model_manager.check_for_model(model_id="some/1")

    # then no error


def test_check_for_model_when_model_not_loaded() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    # when
    with pytest.raises(InferenceModelNotFound):
        model_manager.check_for_model(model_id="some/1")


@pytest.mark.asyncio
async def test_infer_from_request_when_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    with pytest.raises(InferenceModelNotFound):
        await model_manager.infer_from_request(model_id="some/1", request=MagicMock())


@pytest.mark.asyncio
async def test_infer_from_request_when_model_is_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock()}
    request = MagicMock()

    # when
    result = await model_manager.infer_from_request(model_id="some/1", request=request)

    # then
    assert result == model_manager._models["some/1"].infer_from_request.return_value
    model_manager._models["some/1"].infer_from_request.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_infer_from_request_when_model_is_available_but_exception_raised() -> (
    None
):
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_mock = MagicMock()
    error = ValueError()
    model_mock.infer_from_request.side_effect = error
    model_manager._models = {"some/1": model_mock}
    request = MagicMock()

    # when
    with pytest.raises(ValueError) as thrown_exception:
        _ = await model_manager.infer_from_request(model_id="some/1", request=request)

    # then
    assert thrown_exception.value is error
    model_manager._models["some/1"].infer_from_request.assert_called_once_with(request)


def test_make_response_when_model_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock()}
    predictions = MagicMock()

    # when
    result = model_manager.make_response(
        model_id="some/1", predictions=predictions, a=38
    )

    # then
    assert result == model_manager._models["some/1"].make_response.return_value
    model_manager._models["some/1"].make_response.assert_called_once_with(
        predictions, a=38
    )


def test_make_response_when_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    with pytest.raises(InferenceModelNotFound):
        model_manager.make_response(model_id="some/1", predictions=MagicMock())


def test_postprocess_when_model_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock()}
    predictions = MagicMock()
    preprocess_return_metadata = MagicMock()
    # when
    result = model_manager.postprocess(
        model_id="some/1",
        predictions=predictions,
        preprocess_return_metadata=preprocess_return_metadata,
        a=38,
    )

    # then
    assert result == model_manager._models["some/1"].postprocess.return_value
    model_manager._models["some/1"].postprocess.assert_called_once_with(
        predictions, preprocess_return_metadata, a=38
    )


def test_postprocess_when_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    preprocess_return_metadata = MagicMock()

    with pytest.raises(InferenceModelNotFound):
        model_manager.postprocess(
            model_id="some/1",
            predictions=MagicMock(),
            preprocess_return_metadata=preprocess_return_metadata,
        )


def test_preprocess_when_model_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock()}
    request = MagicMock()

    # when
    result = model_manager.preprocess(model_id="some/1", request=request)

    # then
    assert result == model_manager._models["some/1"].preprocess.return_value
    model_manager._models["some/1"].preprocess.assert_called_once_with(**request.dict())


def test_preprocess_when_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    with pytest.raises(InferenceModelNotFound):
        model_manager.preprocess(model_id="some/1", request=MagicMock())


def test_predict_when_model_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_mock = MagicMock()
    model_mock.metrics = {"num_inferences": 0, "avg_inference_time": 0}
    model_manager._models = {"some/1": model_mock}
    prediction_input = MagicMock()

    # when
    result = model_manager.predict(model_id="some/1", prediction_input=prediction_input)

    # then
    assert result == model_mock.predict.return_value
    model_mock.predict.assert_called_once_with(prediction_input=prediction_input)
    assert model_mock.metrics["num_inferences"] == 1
    assert model_mock.metrics["avg_inference_time"] > 0


def test_predict_when_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    with pytest.raises(InferenceModelNotFound):
        _ = model_manager.predict(model_id="some/1", prediction_input=MagicMock())


def test_get_class_names_when_model_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock()}

    # when
    result = model_manager.get_class_names(model_id="some/1")

    # then
    assert result == model_manager._models["some/1"].class_names


def test_get_class_names_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    with pytest.raises(InferenceModelNotFound):
        _ = model_manager.get_class_names(model_id="some/1")


def test_get_task_type_when_model_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock()}

    # when
    result = model_manager.get_task_type(model_id="some/1")

    # then
    assert result == model_manager._models["some/1"].task_type


def test_get_task_type_when_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    with pytest.raises(InferenceModelNotFound):
        _ = model_manager.get_task_type(model_id="some/1")


def test_remove_when_model_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_mock = MagicMock()
    model_manager._models = {"some/1": model_mock}

    # when
    model_manager.remove(model_id="some/1")

    # then
    model_mock.clear_cache.assert_called_once()
    assert model_manager.models() == {}


def test_remove_when_model_not_available() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    model_manager.remove(model_id="some/1")


def test_clear() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock(), "some/2": MagicMock()}

    # when
    model_manager.clear()

    # then
    assert model_manager.models() == {}


def test_model_manager_contains_when_result_should_be_positive() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock(), "some/2": MagicMock()}

    # when
    result = "some/1" in model_manager

    # then
    assert result is True


def test_model_manager_contains_when_result_should_be_negative() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": MagicMock(), "some/2": MagicMock()}

    # when
    result = "some/3" in model_manager

    # then
    assert result is False


def test_model_manager_get_item_when_model_exists() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model = MagicMock()
    model_manager._models = {"some/1": model, "some/2": MagicMock()}

    # when
    result = model_manager["some/1"]

    # then
    assert result is model


def test_model_manager_get_item_when_model_does_not_exist() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {}

    # when
    with pytest.raises(InferenceModelNotFound):
        _ = model_manager["some/1"]


def test_model_manager_describe_models() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_1, model_2 = MagicMock(), MagicMock()
    model_1.task_type = "object-detection"
    model_1.batch_size = 12
    model_1.img_size_w = 640
    model_1.img_size_h = 480
    model_2.task_type = "instance-segmentation"
    model_2.batch_size = 1
    model_2.img_size_w = 480
    model_2.img_size_h = 480
    model_manager._models = {"some/1": model_1, "some/2": model_2}

    # when
    result = model_manager.describe_models()

    # then
    assert sorted(result, key=lambda e: e.model_id) == [
        ModelDescription(
            model_id="some/1",
            task_type="object-detection",
            batch_size=12,
            input_width=640,
            input_height=480,
        ),
        ModelDescription(
            model_id="some/2",
            task_type="instance-segmentation",
            batch_size=1,
            input_width=480,
            input_height=480,
        ),
    ]


def test_add_model_records_cold_start_when_collector_set() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    collector = ModelLoadCollector()
    token = model_load_info.set(collector)

    try:
        # when
        model_manager.add_model(model_id="some/1", api_key="some_api_key")
    finally:
        model_load_info.reset(token)

    # then
    assert "some/1" in model_manager.models()
    assert collector.has_data() is True
    total, detail = collector.summarize()
    assert total > 0


def test_add_model_does_not_record_when_model_already_loaded() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": "A"}
    collector = ModelLoadCollector()
    token = model_load_info.set(collector)

    try:
        # when
        model_manager.add_model(model_id="some/1", api_key="some_api_key")
    finally:
        model_load_info.reset(token)

    # then - model was warm, collector should be empty
    assert collector.has_data() is False


def test_add_model_does_not_record_when_no_collector_set() -> None:
    # given - model_load_info defaults to None (no HTTP context)
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)

    # when - should not raise even though no collector is set
    model_manager.add_model(model_id="some/1", api_key="some_api_key")

    # then
    assert "some/1" in model_manager.models()


def test_add_model_records_multiple_cold_starts() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    collector = ModelLoadCollector()
    token = model_load_info.set(collector)

    try:
        # when
        model_manager.add_model(model_id="model-a/1", api_key="key")
        model_manager.add_model(model_id="model-b/2", api_key="key")
    finally:
        model_load_info.reset(token)

    # then
    total, detail = collector.summarize()
    import json

    parsed = json.loads(detail)
    assert len(parsed) == 2
    assert {e["m"] for e in parsed} == {"model-a/1", "model-b/2"}
    assert all(e["t"] > 0 for e in parsed)


def test_add_model_records_cold_start_only_for_new_models() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"existing/1": "A"}
    collector = ModelLoadCollector()
    token = model_load_info.set(collector)

    try:
        # when - one warm, one cold
        model_manager.add_model(model_id="existing/1", api_key="key")
        model_manager.add_model(model_id="new-model/1", api_key="key")
    finally:
        model_load_info.reset(token)

    # then - only the new model should be recorded
    import json

    total, detail = collector.summarize()
    parsed = json.loads(detail)
    assert len(parsed) == 1
    assert parsed[0]["m"] == "new-model/1"


def test_add_model_records_model_id_when_ids_collector_set() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    ids = RequestModelIds()
    token = request_model_ids.set(ids)

    try:
        # when
        model_manager.add_model(model_id="some/1", api_key="key")
    finally:
        request_model_ids.reset(token)

    # then
    assert ids.get_ids() == {"some/1"}


def test_add_model_records_model_id_for_warm_model() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"some/1": "A"}
    ids = RequestModelIds()
    token = request_model_ids.set(ids)

    try:
        # when - model already loaded (warm)
        model_manager.add_model(model_id="some/1", api_key="key")
    finally:
        request_model_ids.reset(token)

    # then - model ID should still be recorded
    assert ids.get_ids() == {"some/1"}


def test_add_model_records_all_model_ids() -> None:
    # given
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models = {"warm/1": "A"}
    ids = RequestModelIds()
    token = request_model_ids.set(ids)

    try:
        # when - one warm, one cold
        model_manager.add_model(model_id="warm/1", api_key="key")
        model_manager.add_model(model_id="cold/1", api_key="key")
    finally:
        request_model_ids.reset(token)

    # then
    assert ids.get_ids() == {"warm/1", "cold/1"}
