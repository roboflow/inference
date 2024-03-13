import json
from collections import OrderedDict
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from inference.core.active_learning import core
from inference.core.active_learning.core import (
    collect_tags,
    execute_datapoint_registration,
    execute_sampling,
    is_prediction_registration_forbidden,
    prepare_image_to_registration,
    register_datapoint_at_roboflow,
    safe_register_image_at_roboflow,
)
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    BatchReCreationInterval,
    ImageDimensions,
    SamplingMethod,
    StrategyLimit,
    StrategyLimitType,
)
from inference.core.exceptions import RoboflowAPIConnectionError


def test_execute_sampling() -> None:
    # given
    sampling_methods = [
        SamplingMethod(name="method_a", sample=MagicMock()),
        SamplingMethod(name="method_b", sample=MagicMock()),
        SamplingMethod(name="method_c", sample=MagicMock()),
    ]
    sampling_methods[0].sample.return_value = True
    sampling_methods[1].sample.return_value = False
    sampling_methods[2].sample.return_value = True
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    prediction = {"some": "prediction"}
    prediction_type = "object-detection"

    # when
    result = execute_sampling(
        image=image,
        prediction=prediction,
        prediction_type=prediction_type,
        sampling_methods=sampling_methods,
    )

    # then
    assert result == ["method_a", "method_c"]
    for i in range(3):
        sampling_methods[i].sample.assert_called_once_with(
            image, prediction, prediction_type
        )


def test_prepare_image_to_registration_when_desired_size_is_not_given(
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = prepare_image_to_registration(
        image=image_as_numpy, desired_size=None, jpeg_compression_level=95
    )
    bytes_array = np.frombuffer(result[0], dtype=np.uint8)
    decoded_result = cv2.imdecode(bytes_array, flags=cv2.IMREAD_UNCHANGED)

    # then
    assert decoded_result.shape == image_as_numpy.shape
    assert np.allclose(decoded_result, image_as_numpy)
    assert abs(result[1] - 1.0) < 1e-5


def test_prepare_image_to_registration_when_desired_size_given(
    image_as_numpy: np.ndarray,
) -> None:
    # when
    result = prepare_image_to_registration(
        image=image_as_numpy,
        desired_size=ImageDimensions(height=32, width=16),
        jpeg_compression_level=95,
    )
    bytes_array = np.frombuffer(result[0], dtype=np.uint8)
    decoded_result = cv2.imdecode(bytes_array, flags=cv2.IMREAD_UNCHANGED)

    # then
    assert decoded_result.shape == (16, 16, 3)
    assert abs(result[1] - 1 / 8) < 1e-5


@mock.patch.object(core, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_env_tags_not_set() -> None:
    # given
    configuration = ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=True,
        sampling_methods=[],
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
        strategies_limits={"default_strategy": []},
        tags=["a", "b"],
        strategies_tags={"default_strategy": ["c", "d"], "other_strategy": ["e", "f"]},
    )

    # when
    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    # then
    assert result == ["a", "b", "e", "f", "coin-detection-3"]


@mock.patch.object(core, "ACTIVE_LEARNING_TAGS", ["factory-x", "line-y"])
def test_collect_tags_when_env_tags_are_set() -> None:
    # given
    configuration = ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=True,
        sampling_methods=[],
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
        strategies_limits={"default_strategy": []},
        tags=["a", "b"],
        strategies_tags={"default_strategy": ["c", "d"], "other_strategy": ["e", "f"]},
    )

    # when
    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    # then
    assert result == ["factory-x", "line-y", "a", "b", "e", "f", "coin-detection-3"]


@mock.patch.object(core, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_predictions_not_to_be_persisted() -> None:
    # given
    configuration = ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=False,
        sampling_methods=[],
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
        strategies_limits={"default_strategy": []},
        tags=["a", "b"],
        strategies_tags={"default_strategy": ["c", "d"], "other_strategy": ["e", "f"]},
    )

    # when
    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    # then
    assert result == ["a", "b", "e", "f"]


@mock.patch.object(core, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_strategy_tags_missing() -> None:
    # given
    configuration = ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=True,
        sampling_methods=[],
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
        strategies_limits={"default_strategy": []},
        tags=["a", "b"],
        strategies_tags={"default_strategy": ["c", "d"], "other_strategy": []},
    )

    # when
    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    # then
    assert result == ["a", "b", "coin-detection-3"]


@mock.patch.object(core, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_strategy_tags_missing_and_configuration_tags_missing() -> (
    None
):
    # given
    configuration = ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=True,
        sampling_methods=[],
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
        strategies_limits={"default_strategy": []},
        tags=[],
        strategies_tags={"default_strategy": ["c", "d"], "other_strategy": []},
    )

    # when
    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    # then
    assert result == ["coin-detection-3"]


@mock.patch.object(core, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_invalid_strategy_used() -> None:
    # given
    configuration = ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=True,
        sampling_methods=[],
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
        strategies_limits={"default_strategy": []},
        tags=["a", "b"],
        strategies_tags={"default_strategy": ["c", "d"], "other_strategy": []},
    )

    # when
    with pytest.raises(KeyError):
        _ = collect_tags(configuration=configuration, sampling_strategy="invalid")


@mock.patch.object(core, "return_strategy_credit")
@mock.patch.object(core, "register_image_at_roboflow")
def test_safe_register_image_at_roboflow_when_registration_fails(
    register_image_at_roboflow_mock: MagicMock,
    return_strategy_credit_mock: MagicMock,
) -> None:
    # given
    error = RoboflowAPIConnectionError("some")
    register_image_at_roboflow_mock.side_effect = error
    cache = MagicMock()
    configuration = MagicMock()

    # when
    with pytest.raises(RoboflowAPIConnectionError) as result_error:
        _ = safe_register_image_at_roboflow(
            cache=cache,
            strategy_with_spare_credit="my-strategy",
            encoded_image=b"IMAGE",
            local_image_id="local-id",
            configuration=configuration,
            api_key="api-key",
            batch_name="some-batch",
            tags=[],
            inference_id=None,
        )

    # then
    register_image_at_roboflow_mock.assert_called_once_with(
        api_key="api-key",
        dataset_id=configuration.dataset_id,
        local_image_id="local-id",
        image_bytes=b"IMAGE",
        batch_name="some-batch",
        tags=[],
        inference_id=None,
    )
    return_strategy_credit_mock.assert_called_once_with(
        cache=cache,
        workspace=configuration.workspace_id,
        project=configuration.dataset_id,
        strategy_name="my-strategy",
    )
    assert result_error.value is error


@mock.patch.object(core, "return_strategy_credit")
@mock.patch.object(core, "register_image_at_roboflow")
def test_safe_register_image_at_roboflow_when_registration_detects_duplicate(
    register_image_at_roboflow_mock: MagicMock,
    return_strategy_credit_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {
        "duplicate": True,
        "id": "roboflow-id",
    }
    cache = MagicMock()
    configuration = MagicMock()

    # when
    result = safe_register_image_at_roboflow(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        tags=[],
        inference_id=None,
    )

    # then
    register_image_at_roboflow_mock.assert_called_once_with(
        api_key="api-key",
        dataset_id=configuration.dataset_id,
        local_image_id="local-id",
        image_bytes=b"IMAGE",
        batch_name="some-batch",
        tags=[],
    )
    return_strategy_credit_mock.assert_called_once_with(
        cache=cache,
        workspace=configuration.workspace_id,
        project=configuration.dataset_id,
        strategy_name="my-strategy",
    )
    assert result is None


@mock.patch.object(core, "return_strategy_credit")
@mock.patch.object(core, "register_image_at_roboflow")
def test_safe_register_image_at_roboflow_when_registration_detects_duplicate(
    register_image_at_roboflow_mock: MagicMock,
    return_strategy_credit_mock: MagicMock,
) -> None:
    # given
    register_image_at_roboflow_mock.return_value = {
        "success": True,
        "id": "roboflow-id",
    }
    cache = MagicMock()
    configuration = MagicMock()

    # when
    result = safe_register_image_at_roboflow(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        tags=[],
        inference_id="inference-id-234",
    )

    # then
    register_image_at_roboflow_mock.assert_called_once_with(
        api_key="api-key",
        dataset_id=configuration.dataset_id,
        local_image_id="local-id",
        image_bytes=b"IMAGE",
        batch_name="some-batch",
        tags=[],
        inference_id="inference-id-234",
    )
    return_strategy_credit_mock.assert_not_called()
    assert result == "roboflow-id"


@mock.patch.object(core, "annotate_image_at_roboflow")
@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_predictions_not_to_be_persisted(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = False
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.return_value = "roboflow-id"

    # when
    register_datapoint_at_roboflow(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        inference_id="inference-id-987",
    )

    # then
    collect_tags_mock.assert_called_once_with(
        configuration=configuration,
        sampling_strategy="my-strategy",
    )
    safe_register_image_at_roboflow_mock.assert_called_once_with(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        tags=["a", "b"],
        inference_id="inference-id-987",
    )
    annotate_image_at_roboflow_mock.assert_not_called()


@mock.patch.object(core, "annotate_image_at_roboflow")
@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_predictions_to_be_persisted_but_duplicate_found(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = True
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.return_value = None

    # when
    register_datapoint_at_roboflow(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        inference_id="inference-id-123",
    )

    # then
    collect_tags_mock.assert_called_once_with(
        configuration=configuration,
        sampling_strategy="my-strategy",
    )
    safe_register_image_at_roboflow_mock.assert_called_once_with(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        tags=["a", "b"],
        inference_id="inference-id-123",
    )
    annotate_image_at_roboflow_mock.assert_not_called()


@mock.patch.object(core, "annotate_image_at_roboflow")
@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_predictions_to_be_persisted(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = True
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.return_value = "roboflow-id"

    # when
    register_datapoint_at_roboflow(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        prediction={"predictions": [{"x": 100.0}]},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        inference_id="inference-id-ABC",
    )

    # then
    collect_tags_mock.assert_called_once_with(
        configuration=configuration,
        sampling_strategy="my-strategy",
    )
    safe_register_image_at_roboflow_mock.assert_called_once_with(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        tags=["a", "b"],
        inference_id="inference-id-ABC",
    )
    annotate_image_at_roboflow_mock.assert_called_once_with(
        api_key="api-key",
        dataset_id=configuration.dataset_id,
        local_image_id="local-id",
        roboflow_image_id="roboflow-id",
        annotation_content=json.dumps({"predictions": [{"x": 100.0}]}),
        annotation_file_type="json",
        is_prediction=True,
    )


@mock.patch.object(core, "annotate_image_at_roboflow")
@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_image_registration_error_occurs(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
    annotate_image_at_roboflow_mock: MagicMock,
) -> None:
    # given
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = True
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.side_effect = RoboflowAPIConnectionError(
        "error"
    )

    # when
    with pytest.raises(RoboflowAPIConnectionError):
        register_datapoint_at_roboflow(
            cache=cache,
            strategy_with_spare_credit="my-strategy",
            encoded_image=b"IMAGE",
            local_image_id="local-id",
            prediction={"some": "prediction"},
            prediction_type="object-detection",
            configuration=configuration,
            api_key="api-key",
            batch_name="some-batch",
            inference_id="inference-id-876",
        )

    # then
    collect_tags_mock.assert_called_once_with(
        configuration=configuration,
        sampling_strategy="my-strategy",
    )
    safe_register_image_at_roboflow_mock.assert_called_once_with(
        cache=cache,
        strategy_with_spare_credit="my-strategy",
        encoded_image=b"IMAGE",
        local_image_id="local-id",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        tags=["a", "b"],
        inference_id="inference-id-876",
    )
    annotate_image_at_roboflow_mock.assert_not_called()


@mock.patch.object(core, "register_datapoint_at_roboflow")
@mock.patch.object(core, "use_credit_of_matching_strategy")
@mock.patch.object(core, "adjust_prediction_to_client_scaling_factor")
def test_execute_datapoint_registration_when_no_spare_credit_found(
    adjust_prediction_to_client_scaling_factor_mock: MagicMock,
    use_credit_of_matching_strategy_mock: MagicMock,
    register_datapoint_at_roboflow_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    # given
    cache, configuration = MagicMock(), MagicMock()
    configuration.max_image_size = None
    configuration.jpeg_compression_level = 75
    configuration.strategies_limits = {
        "strategy_a": [],
        "strategy_b": [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100)],
        "strategy_c": [],
    }
    use_credit_of_matching_strategy_mock.return_value = None

    # when
    execute_datapoint_registration(
        cache=cache,
        matching_strategies=["strategy_a", "strategy_b"],
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
    )

    # then
    adjust_prediction_to_client_scaling_factor_mock.assert_called_once_with(
        prediction={"some": "prediction"},
        scaling_factor=1.0,
        prediction_type="object-detection",
    )
    use_credit_of_matching_strategy_mock.assert_called_once_with(
        cache=cache,
        workspace=configuration.workspace_id,
        project=configuration.dataset_id,
        matching_strategies_limits=OrderedDict(
            [
                ("strategy_a", []),
                (
                    "strategy_b",
                    [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100)],
                ),
            ]
        ),
    )
    register_datapoint_at_roboflow_mock.assert_not_called()


@mock.patch.object(core, "register_datapoint_at_roboflow")
@mock.patch.object(core, "use_credit_of_matching_strategy")
@mock.patch.object(core, "adjust_prediction_to_client_scaling_factor")
def test_execute_datapoint_registration_when_spare_credit_found(
    adjust_prediction_to_client_scaling_factor_mock: MagicMock,
    use_credit_of_matching_strategy_mock: MagicMock,
    register_datapoint_at_roboflow_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    # given
    cache, configuration = MagicMock(), MagicMock()
    configuration.max_image_size = None
    configuration.jpeg_compression_level = 75
    configuration.strategies_limits = {
        "strategy_a": [],
        "strategy_b": [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100)],
        "strategy_c": [],
    }
    use_credit_of_matching_strategy_mock.return_value = "strategy_b"

    # when
    execute_datapoint_registration(
        cache=cache,
        matching_strategies=["strategy_a", "strategy_b"],
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
    )

    # then
    adjust_prediction_to_client_scaling_factor_mock.assert_called_once_with(
        prediction={"some": "prediction"},
        scaling_factor=1.0,
        prediction_type="object-detection",
    )
    use_credit_of_matching_strategy_mock.assert_called_once_with(
        cache=cache,
        workspace=configuration.workspace_id,
        project=configuration.dataset_id,
        matching_strategies_limits=OrderedDict(
            [
                ("strategy_a", []),
                (
                    "strategy_b",
                    [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100)],
                ),
            ]
        ),
    )
    register_datapoint_at_roboflow_mock.assert_called_once()
    assert (
        register_datapoint_at_roboflow_mock.call_args[1]["strategy_with_spare_credit"]
        == "strategy_b"
    )


def test_is_prediction_registration_forbidden_when_stub_prediction_given() -> None:
    # when
    result = is_prediction_registration_forbidden(
        prediction={"is_stub": True},
        persist_predictions=True,
        roboflow_image_id="roboflow_id",
    )

    # then
    assert result is True


def test_is_prediction_registration_forbidden_when_prediction_persistence_turned_off() -> (
    None
):
    # when
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [], "top": "cat"},
        persist_predictions=False,
        roboflow_image_id="roboflow_id",
    )

    # then
    assert result is True


def test_is_prediction_registration_forbidden_when_roboflow_image_id_not_registered() -> (
    None
):
    # when
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [{"x": 37}], "top": "cat"},
        persist_predictions=True,
        roboflow_image_id=None,
    )

    # then
    assert result is True


def test_is_prediction_registration_forbidden_when_prediction_should_be_rejected_based_on_empty_content() -> (
    None
):
    # when
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [], "top": "cat"},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    # then
    assert result is False


def test_is_prediction_registration_forbidden_when_prediction_should_be_registered() -> (
    None
):
    # when
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [{"x": 37}], "top": "cat"},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    # then
    assert result is False


def test_is_prediction_registration_forbidden_when_classification_output_only_with_top_category_provided() -> (
    None
):
    # when
    result = is_prediction_registration_forbidden(
        prediction={"top": "cat"},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    # then
    assert result is False


def test_is_prediction_registration_forbidden_when_detection_output_without_predictions_provided() -> (
    None
):
    # when
    result = is_prediction_registration_forbidden(
        prediction={"predictions": []},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    # then
    assert result is True
