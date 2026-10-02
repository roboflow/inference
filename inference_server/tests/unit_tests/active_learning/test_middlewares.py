from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from inference_server.active_learning import middlewares
from inference_server.active_learning.middlewares import ActiveLearningMiddleware


@mock.patch.object(middlewares, "execute_sampling")
def test_active_learning_registration_when_no_configuration_provided(
    execute_sampling_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    platform_client = MagicMock()
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=None,
        cache=MagicMock(),
        platform_client=platform_client,
    )

    middleware.register(
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    execute_sampling_mock.assert_not_called()
    assert platform_client.mock_calls == []


@mock.patch.object(middlewares, "execute_sampling")
def test_active_learning_registration_when_no_matching_strategy(
    execute_sampling_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    configuration = MagicMock()
    execute_sampling_mock.return_value = []
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=configuration,
        cache=MagicMock(),
        platform_client=MagicMock(),
    )

    middleware.register(
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    execute_sampling_mock.assert_called_once_with(
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        sampling_methods=configuration.sampling_methods,
    )


@mock.patch.object(middlewares, "image_can_be_submitted_to_batch")
@mock.patch.object(middlewares, "generate_batch_name")
@mock.patch.object(middlewares, "execute_sampling")
def test_active_learning_registration_when_matching_strategies_found_but_batch_limit_exceeded(
    execute_sampling_mock: MagicMock,
    generate_batch_name_mock: MagicMock,
    image_can_be_submitted_to_batch_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    configuration, platform_client = MagicMock(), MagicMock()
    execute_sampling_mock.return_value = ["strategy-a", "strategy-b"]
    generate_batch_name_mock.return_value = "some-batch"
    image_can_be_submitted_to_batch_mock.return_value = False
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=configuration,
        cache=MagicMock(),
        platform_client=platform_client,
    )

    middleware.register(
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    execute_sampling_mock.assert_called_once_with(
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        sampling_methods=configuration.sampling_methods,
    )
    generate_batch_name_mock.assert_called_once_with(configuration=configuration)
    image_can_be_submitted_to_batch_mock.assert_called_once_with(
        batch_name="some-batch",
        workspace_id=configuration.workspace_id,
        dataset_id=configuration.dataset_id,
        max_batch_images=configuration.max_batch_images,
        api_key="api-key",
        platform_client=platform_client,
    )
    platform_client.register_image_at_roboflow.assert_not_called()


@mock.patch.object(middlewares, "execute_datapoint_registration")
@mock.patch.object(middlewares, "image_can_be_submitted_to_batch")
@mock.patch.object(middlewares, "generate_batch_name")
@mock.patch.object(middlewares, "execute_sampling")
def test_active_learning_registration_when_datapoint_is_to_be_registered(
    execute_sampling_mock: MagicMock,
    generate_batch_name_mock: MagicMock,
    image_can_be_submitted_to_batch_mock: MagicMock,
    execute_datapoint_registration_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    configuration, cache, platform_client = MagicMock(), MagicMock(), MagicMock()
    execute_sampling_mock.return_value = ["strategy-a", "strategy-b"]
    generate_batch_name_mock.return_value = "some-batch"
    image_can_be_submitted_to_batch_mock.return_value = True
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=configuration,
        cache=cache,
        platform_client=platform_client,
    )

    middleware.register(
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    execute_sampling_mock.assert_called_once_with(
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        sampling_methods=configuration.sampling_methods,
    )
    generate_batch_name_mock.assert_called_once_with(configuration=configuration)
    image_can_be_submitted_to_batch_mock.assert_called_once_with(
        batch_name="some-batch",
        workspace_id=configuration.workspace_id,
        dataset_id=configuration.dataset_id,
        max_batch_images=configuration.max_batch_images,
        api_key="api-key",
        platform_client=platform_client,
    )
    execute_datapoint_registration_mock.assert_called_once_with(
        cache=cache,
        matching_strategies=["strategy-a", "strategy-b"],
        image=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
        configuration=configuration,
        api_key="api-key",
        batch_name="some-batch",
        platform_client=platform_client,
        inference_id=None,
    )


@mock.patch.object(middlewares, "execute_sampling")
def test_active_learning_registration_when_error_raised(
    execute_sampling_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    execute_sampling_mock.side_effect = Exception("some")
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=MagicMock(),
        cache=MagicMock(),
        platform_client=MagicMock(),
    )

    with pytest.raises(Exception) as registered_error:
        middleware.register(
            image=image_as_numpy,
            prediction={"some": "prediction"},
            prediction_type="object-detection",
        )

    assert registered_error.value is execute_sampling_mock.side_effect


def test_active_learning_registration_of_batch(
    image_as_numpy: np.ndarray,
) -> None:
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=MagicMock(),
        cache=MagicMock(),
        platform_client=MagicMock(),
    )
    middleware.register = MagicMock()

    middleware.register_batch(
        images=[image_as_numpy, image_as_numpy],
        predictions=[{"some": "prediction"}, {"other": "prediction"}],
        prediction_type="object-detection",
    )

    middleware.register.assert_has_calls(
        [
            call(
                image=image_as_numpy,
                prediction={"some": "prediction"},
                prediction_type="object-detection",
                inference_id=None,
            ),
            call(
                image=image_as_numpy,
                prediction={"other": "prediction"},
                prediction_type="object-detection",
                inference_id=None,
            ),
        ]
    )


@mock.patch.object(middlewares, "prepare_active_learning_configuration")
def test_init_loads_the_configuration_through_the_platform_client(
    prepare_active_learning_configuration_mock: MagicMock,
) -> None:
    cache, platform_client = MagicMock(), MagicMock()

    middleware = ActiveLearningMiddleware.init(
        api_key="api-key",
        target_dataset="target",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )

    prepare_active_learning_configuration_mock.assert_called_once_with(
        api_key="api-key",
        target_dataset="target",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )
    assert (
        middleware._configuration
        is prepare_active_learning_configuration_mock.return_value
    )


@mock.patch.object(middlewares, "prepare_active_learning_configuration_inplace")
def test_init_from_config_builds_the_configuration_from_the_document(
    prepare_active_learning_configuration_inplace_mock: MagicMock,
) -> None:
    cache, platform_client = MagicMock(), MagicMock()

    middleware = ActiveLearningMiddleware.init_from_config(
        api_key="api-key",
        target_dataset="target",
        model_id="some/1",
        cache=cache,
        config={"enabled": True},
        platform_client=platform_client,
    )

    prepare_active_learning_configuration_inplace_mock.assert_called_once_with(
        api_key="api-key",
        target_dataset="target",
        model_id="some/1",
        active_learning_configuration={"enabled": True},
        platform_client=platform_client,
    )
    assert (
        middleware._configuration
        is prepare_active_learning_configuration_inplace_mock.return_value
    )
