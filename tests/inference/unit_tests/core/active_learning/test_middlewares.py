from queue import Queue
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
import pytest

from inference.core.active_learning import middlewares
from inference.core.active_learning.middlewares import (
    ActiveLearningMiddleware,
    ThreadingActiveLearningMiddleware,
)


@mock.patch.object(middlewares, "load_image")
def test_active_learning_registration_when_no_configuration_provided(
    load_image_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    # given
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=None,
        cache=MagicMock(),
    )

    # when
    middleware.register(
        inference_input=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    # then
    load_image_mock.assert_not_called()


@mock.patch.object(middlewares, "execute_sampling")
def test_active_learning_registration_when_no_matching_strategy(
    execute_sampling_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    # given
    configuration = MagicMock()
    execute_sampling_mock.return_value = []
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=configuration,
        cache=MagicMock(),
    )

    # when
    middleware.register(
        inference_input=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    # then
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
    # given
    configuration = MagicMock()
    execute_sampling_mock.return_value = ["strategy-a", "strategy-b"]
    generate_batch_name_mock.return_value = "some-batch"
    image_can_be_submitted_to_batch_mock.return_value = False
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=configuration,
        cache=MagicMock(),
    )

    # when
    middleware.register(
        inference_input=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    # then
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
    )


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
    # given
    configuration, cache = MagicMock(), MagicMock()
    execute_sampling_mock.return_value = ["strategy-a", "strategy-b"]
    generate_batch_name_mock.return_value = "some-batch"
    image_can_be_submitted_to_batch_mock.return_value = True
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=configuration,
        cache=cache,
    )

    # when
    middleware.register(
        inference_input=image_as_numpy,
        prediction={"some": "prediction"},
        prediction_type="object-detection",
    )

    # then
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
        inference_id=None,
    )


@mock.patch.object(middlewares, "load_image")
def test_active_learning_registration_when_error_raised(
    load_image_mock: MagicMock,
    image_as_numpy: np.ndarray,
) -> None:
    # given
    load_image_mock.side_effect = Exception("some")
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=MagicMock(),
        cache=MagicMock(),
    )

    # when
    with pytest.raises(Exception) as registered_error:
        middleware.register(
            inference_input=image_as_numpy,
            prediction={"some": "prediction"},
            prediction_type="object-detection",
        )

    # then
    assert registered_error.value is load_image_mock.side_effect


def test_active_learning_registration_of_batch(
    image_as_numpy: np.ndarray,
) -> None:
    # given
    middleware = ActiveLearningMiddleware(
        api_key="api-key",
        configuration=MagicMock(),
        cache=MagicMock(),
    )
    middleware.register = MagicMock()

    # when
    middleware.register_batch(
        inference_inputs=[image_as_numpy, image_as_numpy],
        predictions=[{"some": "prediction"}, {"other": "prediction"}],
        prediction_type="object-detection",
        disable_preproc_auto_orient=False,
    )

    # then
    middleware.register.assert_has_calls(
        [
            call(
                inference_input=image_as_numpy,
                prediction={"some": "prediction"},
                prediction_type="object-detection",
                disable_preproc_auto_orient=False,
                inference_id=None,
            ),
            call(
                inference_input=image_as_numpy,
                prediction={"other": "prediction"},
                prediction_type="object-detection",
                disable_preproc_auto_orient=False,
                inference_id=None,
            ),
        ]
    )


@pytest.mark.timeout(30)
def test_threading_active_learning_middleware() -> None:
    # given
    image = MagicMock()
    task_queue = Queue()
    middleware = ThreadingActiveLearningMiddleware(
        api_key="api-key",
        configuration=MagicMock(),
        cache=MagicMock(),
        task_queue=task_queue,
    )
    middleware._execute_registration = MagicMock()
    middleware._execute_registration.side_effect = [None, Exception, None]

    # when
    with middleware:
        middleware.register_batch(
            inference_inputs=[image, image, image],
            predictions=[
                {"some": "prediction"},
                {"other": "prediction"},
                {"third": "prediction"},
            ],
            prediction_type="object-detection",
            disable_preproc_auto_orient=False,
        )

    # then
    assert middleware._registration_thread is None
    middleware._execute_registration.assert_has_calls(
        [
            call(
                inference_input=image,
                prediction={"some": "prediction"},
                prediction_type="object-detection",
                disable_preproc_auto_orient=False,
            ),
            call(
                inference_input=image,
                prediction={"other": "prediction"},
                prediction_type="object-detection",
                disable_preproc_auto_orient=False,
            ),
            call(
                inference_input=image,
                prediction={"third": "prediction"},
                prediction_type="object-detection",
                disable_preproc_auto_orient=False,
            ),
        ]
    )
