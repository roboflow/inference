import json
from collections import OrderedDict
from unittest import mock
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from roboflow_workflows.prototypes.platform_errors import RoboflowAPIConnectionError

from inference_server.active_learning import core
from inference_server.active_learning.core import (
    collect_tags,
    execute_datapoint_registration,
    execute_sampling,
    is_prediction_registration_forbidden,
    prepare_image_to_registration_with_metadata,
    register_datapoint_at_roboflow,
    safe_register_image_at_roboflow,
)
from inference_server.active_learning.entities import (
    ActiveLearningConfiguration,
    BatchReCreationInterval,
    ImageDimensions,
    SamplingMethod,
    StrategyLimit,
    StrategyLimitType,
)


def test_execute_sampling() -> None:
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

    result = execute_sampling(
        image=image,
        prediction=prediction,
        prediction_type=prediction_type,
        sampling_methods=sampling_methods,
    )

    assert result == ["method_a", "method_c"]
    for i in range(3):
        sampling_methods[i].sample.assert_called_once_with(
            image, prediction, prediction_type
        )


def test_prepare_image_to_registration_with_metadata_when_desired_size_is_not_given(
    image_as_numpy: np.ndarray,
) -> None:
    result = prepare_image_to_registration_with_metadata(
        image=image_as_numpy,
        desired_size=None,
        jpeg_compression_level=95,
    )

    assert result.original_size_wh == (
        image_as_numpy.shape[1],
        image_as_numpy.shape[0],
    )
    assert result.final_size_wh == result.original_size_wh
    assert abs(result.scale_x - 1.0) < 1e-5
    assert abs(result.scale_y - 1.0) < 1e-5


def test_prepare_image_to_registration_when_desired_size_given(
    image_as_numpy: np.ndarray,
) -> None:
    result = prepare_image_to_registration_with_metadata(
        image=image_as_numpy,
        desired_size=ImageDimensions(height=32, width=16),
        jpeg_compression_level=95,
    )
    bytes_array = np.frombuffer(result.encoded_image, dtype=np.uint8)
    decoded_result = cv2.imdecode(bytes_array, flags=cv2.IMREAD_UNCHANGED)

    assert decoded_result.shape == (16, 16, 3)
    assert abs(result.scaling_factor - 1 / 8) < 1e-5
    assert result.final_size_wh == (16, 16)
    assert abs(result.scale_x - result.scale_y) < 1e-9


def test_prepare_image_to_registration_reports_anisotropic_scales() -> None:
    image = np.zeros((1080, 4000, 3), dtype=np.uint8)
    result = prepare_image_to_registration_with_metadata(
        image=image,
        desired_size=ImageDimensions(height=2080, width=2080),
        jpeg_compression_level=95,
    )
    assert result.original_size_wh == (4000, 1080)
    assert result.final_size_wh == (2080, 561)
    assert abs(result.scale_x - 2080 / 4000) < 1e-9
    assert abs(result.scale_y - 561 / 1080) < 1e-9
    assert abs(result.scale_x - result.scale_y) > 1e-4
    assert abs(result.scaling_factor - result.scale_y) < 1e-9


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 0, 3), dtype=np.uint8),
    ],
)
def test_prepare_image_to_registration_rejects_empty_image_dimensions(
    image: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="dimensions are invalid"):
        prepare_image_to_registration_with_metadata(
            image=image,
            desired_size=None,
            jpeg_compression_level=95,
        )


@mock.patch.object(core.server_configuration, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_env_tags_not_set() -> None:
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

    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    assert result == ["a", "b", "e", "f", "coin-detection-3"]


@mock.patch.object(
    core.server_configuration, "ACTIVE_LEARNING_TAGS", ["factory-x", "line-y"]
)
def test_collect_tags_when_env_tags_are_set() -> None:
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

    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    assert result == ["factory-x", "line-y", "a", "b", "e", "f", "coin-detection-3"]


@mock.patch.object(core.server_configuration, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_predictions_not_to_be_persisted() -> None:
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

    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    assert result == ["a", "b", "e", "f"]


@mock.patch.object(core.server_configuration, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_strategy_tags_missing() -> None:
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

    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    assert result == ["a", "b", "coin-detection-3"]


@mock.patch.object(core.server_configuration, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_strategy_tags_missing_and_configuration_tags_missing() -> (
    None
):
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

    result = collect_tags(
        configuration=configuration, sampling_strategy="other_strategy"
    )

    assert result == ["coin-detection-3"]


@mock.patch.object(core.server_configuration, "ACTIVE_LEARNING_TAGS", None)
def test_collect_tags_when_invalid_strategy_used() -> None:
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

    with pytest.raises(KeyError):
        _ = collect_tags(configuration=configuration, sampling_strategy="invalid")


@mock.patch.object(core, "return_strategy_credit")
def test_safe_register_image_at_roboflow_when_registration_fails(
    return_strategy_credit_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    error = RoboflowAPIConnectionError("some")
    platform_client.register_image_at_roboflow.side_effect = error
    cache = MagicMock()
    configuration = MagicMock()

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
            platform_client=platform_client,
        )

    platform_client.register_image_at_roboflow.assert_called_once_with(
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
def test_safe_register_image_at_roboflow_when_registration_detects_duplicate(
    return_strategy_credit_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    platform_client.register_image_at_roboflow.return_value = {
        "duplicate": True,
        "id": "roboflow-id",
    }
    cache = MagicMock()
    configuration = MagicMock()

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
        platform_client=platform_client,
    )

    platform_client.register_image_at_roboflow.assert_called_once_with(
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
    assert result is None


@mock.patch.object(core, "return_strategy_credit")
def test_safe_register_image_at_roboflow_when_registration_succeeds(
    return_strategy_credit_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    platform_client.register_image_at_roboflow.return_value = {
        "success": True,
        "id": "roboflow-id",
    }
    cache = MagicMock()
    configuration = MagicMock()

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
        platform_client=platform_client,
    )

    platform_client.register_image_at_roboflow.assert_called_once_with(
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


@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_predictions_not_to_be_persisted(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = False
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.return_value = "roboflow-id"

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
        platform_client=platform_client,
    )

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
        platform_client=platform_client,
    )
    platform_client.annotate_image_at_roboflow.assert_not_called()


@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_predictions_to_be_persisted_but_duplicate_found(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = True
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.return_value = None

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
        platform_client=platform_client,
    )

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
        platform_client=platform_client,
    )
    platform_client.annotate_image_at_roboflow.assert_not_called()


@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_predictions_to_be_persisted(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = True
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.return_value = "roboflow-id"

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
        platform_client=platform_client,
    )

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
        platform_client=platform_client,
    )
    platform_client.annotate_image_at_roboflow.assert_called_once_with(
        api_key="api-key",
        dataset_id=configuration.dataset_id,
        local_image_id="local-id",
        roboflow_image_id="roboflow-id",
        annotation_content=json.dumps({"predictions": [{"x": 100.0}]}),
        annotation_file_type="json",
        is_prediction=True,
    )


@mock.patch.object(core, "safe_register_image_at_roboflow")
@mock.patch.object(core, "collect_tags")
def test_register_datapoint_at_roboflow_when_image_registration_error_occurs(
    collect_tags_mock: MagicMock,
    safe_register_image_at_roboflow_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    cache, configuration = MagicMock(), MagicMock()
    configuration.persist_predictions = True
    collect_tags_mock.return_value = ["a", "b"]
    safe_register_image_at_roboflow_mock.side_effect = RoboflowAPIConnectionError(
        "error"
    )

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
            platform_client=platform_client,
        )

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
        platform_client=platform_client,
    )
    platform_client.annotate_image_at_roboflow.assert_not_called()


@mock.patch.object(core, "register_datapoint_at_roboflow")
@mock.patch.object(core, "use_credit_of_matching_strategy")
@mock.patch.object(core, "adjust_prediction_to_client_scaling_factor")
def test_execute_datapoint_registration_when_no_spare_credit_found(
    adjust_prediction_to_client_scaling_factor_mock: MagicMock,
    use_credit_of_matching_strategy_mock: MagicMock,
    register_datapoint_at_roboflow_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    platform_client = MagicMock()
    cache, configuration = MagicMock(), MagicMock()
    configuration.max_image_size = None
    configuration.jpeg_compression_level = 75
    configuration.strategies_limits = {
        "strategy_a": [],
        "strategy_b": [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100)],
        "strategy_c": [],
    }
    use_credit_of_matching_strategy_mock.return_value = None

    execute_datapoint_registration(
        cache=cache,
        matching_strategies=["strategy_a", "strategy_b"],
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        platform_client=platform_client,
    )

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
    platform_client = MagicMock()
    cache, configuration = MagicMock(), MagicMock()
    configuration.max_image_size = None
    configuration.jpeg_compression_level = 75
    configuration.strategies_limits = {
        "strategy_a": [],
        "strategy_b": [StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100)],
        "strategy_c": [],
    }
    use_credit_of_matching_strategy_mock.return_value = "strategy_b"

    execute_datapoint_registration(
        cache=cache,
        matching_strategies=["strategy_a", "strategy_b"],
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        platform_client=platform_client,
    )

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
    result = is_prediction_registration_forbidden(
        prediction={"is_stub": True},
        persist_predictions=True,
        roboflow_image_id="roboflow_id",
    )

    assert result is True


def test_is_prediction_registration_forbidden_when_prediction_persistence_turned_off() -> (
    None
):
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [], "top": "cat"},
        persist_predictions=False,
        roboflow_image_id="roboflow_id",
    )

    assert result is True


def test_is_prediction_registration_forbidden_when_roboflow_image_id_not_registered() -> (
    None
):
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [{"x": 37}], "top": "cat"},
        persist_predictions=True,
        roboflow_image_id=None,
    )

    assert result is True


def test_is_prediction_registration_forbidden_when_prediction_should_be_rejected_based_on_empty_content() -> (
    None
):
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [], "top": "cat"},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    assert result is False


def test_is_prediction_registration_forbidden_when_prediction_should_be_registered() -> (
    None
):
    result = is_prediction_registration_forbidden(
        prediction={"predictions": [{"x": 37}], "top": "cat"},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    assert result is False


def test_is_prediction_registration_forbidden_when_classification_output_only_with_top_category_provided() -> (
    None
):
    result = is_prediction_registration_forbidden(
        prediction={"top": "cat"},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    assert result is False


def test_is_prediction_registration_forbidden_when_detection_output_without_predictions_provided() -> (
    None
):
    result = is_prediction_registration_forbidden(
        prediction={"predictions": []},
        persist_predictions=True,
        roboflow_image_id="some+id",
    )

    assert result is True
