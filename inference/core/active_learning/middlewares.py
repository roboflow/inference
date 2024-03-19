import queue
from queue import Queue
from threading import Thread
from typing import Any, List, Optional

from inference.core import logger
from inference.core.active_learning.accounting import image_can_be_submitted_to_batch
from inference.core.active_learning.batching import generate_batch_name
from inference.core.active_learning.configuration import (
    prepare_active_learning_configuration,
    prepare_active_learning_configuration_inplace,
)
from inference.core.active_learning.core import (
    execute_datapoint_registration,
    execute_sampling,
)
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    Prediction,
    PredictionType,
)
from inference.core.cache.base import BaseCache
from inference.core.utils.image_utils import load_image

MAX_REGISTRATION_QUEUE_SIZE = 512


class NullActiveLearningMiddleware:
    def register_batch(
        self,
        inference_inputs: List[Any],
        predictions: List[Prediction],
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
    ) -> None:
        pass

    def register(
        self,
        inference_input: Any,
        prediction: dict,
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
    ) -> None:
        pass

    def start_registration_thread(self) -> None:
        pass

    def stop_registration_thread(self) -> None:
        pass

    def __enter__(self) -> "NullActiveLearningMiddleware":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class ActiveLearningMiddleware:
    @classmethod
    def init(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: BaseCache,
    ) -> "ActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            cache=cache,
        )
        return cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
        )

    @classmethod
    def init_from_config(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: BaseCache,
        config: Optional[dict],
    ) -> "ActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration_inplace(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            active_learning_configuration=config,
        )
        return cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
        )

    def __init__(
        self,
        api_key: str,
        configuration: Optional[ActiveLearningConfiguration],
        cache: BaseCache,
    ):
        self._api_key = api_key
        self._configuration = configuration
        self._cache = cache

    def register_batch(
        self,
        inference_inputs: List[Any],
        predictions: List[Prediction],
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
        inference_id=None,
    ) -> None:
        for inference_input, prediction in zip(inference_inputs, predictions):
            self.register(
                inference_input=inference_input,
                prediction=prediction,
                prediction_type=prediction_type,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
                inference_id=inference_id,
            )

    def register(
        self,
        inference_input: Any,
        prediction: dict,
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
        inference_id=None,
    ) -> None:
        self._execute_registration(
            inference_input=inference_input,
            prediction=prediction,
            prediction_type=prediction_type,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
            inference_id=inference_id,
        )

    def _execute_registration(
        self,
        inference_input: Any,
        prediction: dict,
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
        inference_id=None,
    ) -> None:
        if self._configuration is None:
            return None
        image, is_bgr = load_image(
            value=inference_input,
            disable_preproc_auto_orient=disable_preproc_auto_orient,
        )
        if not is_bgr:
            image = image[:, :, ::-1]
        matching_strategies = execute_sampling(
            image=image,
            prediction=prediction,
            prediction_type=prediction_type,
            sampling_methods=self._configuration.sampling_methods,
        )
        if len(matching_strategies) == 0:
            return None
        batch_name = generate_batch_name(configuration=self._configuration)
        if not image_can_be_submitted_to_batch(
            batch_name=batch_name,
            workspace_id=self._configuration.workspace_id,
            dataset_id=self._configuration.dataset_id,
            max_batch_images=self._configuration.max_batch_images,
            api_key=self._api_key,
        ):
            logger.debug(f"Limit on Active Learning batch size reached.")
            return None
        execute_datapoint_registration(
            cache=self._cache,
            matching_strategies=matching_strategies,
            image=image,
            prediction=prediction,
            prediction_type=prediction_type,
            configuration=self._configuration,
            api_key=self._api_key,
            batch_name=batch_name,
            inference_id=inference_id,
        )


class ThreadingActiveLearningMiddleware(ActiveLearningMiddleware):
    @classmethod
    def init(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: BaseCache,
        max_queue_size: int = MAX_REGISTRATION_QUEUE_SIZE,
    ) -> "ThreadingActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            cache=cache,
        )
        task_queue = Queue(max_queue_size)
        return cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
            task_queue=task_queue,
        )

    @classmethod
    def init_from_config(
        cls,
        api_key: str,
        target_dataset: str,
        model_id: str,
        cache: BaseCache,
        config: Optional[dict],
        max_queue_size: int = MAX_REGISTRATION_QUEUE_SIZE,
    ) -> "ThreadingActiveLearningMiddleware":
        configuration = prepare_active_learning_configuration_inplace(
            api_key=api_key,
            target_dataset=target_dataset,
            model_id=model_id,
            active_learning_configuration=config,
        )
        task_queue = Queue(max_queue_size)
        return cls(
            api_key=api_key,
            configuration=configuration,
            cache=cache,
            task_queue=task_queue,
        )

    def __init__(
        self,
        api_key: str,
        configuration: ActiveLearningConfiguration,
        cache: BaseCache,
        task_queue: Queue,
    ):
        super().__init__(api_key=api_key, configuration=configuration, cache=cache)
        self._task_queue = task_queue
        self._registration_thread: Optional[Thread] = None

    def register(
        self,
        inference_input: Any,
        prediction: dict,
        prediction_type: PredictionType,
        disable_preproc_auto_orient: bool = False,
        inference_id=None,
    ) -> None:
        logger.debug(f"Putting registration task into queue")
        try:
            self._task_queue.put_nowait(
                (
                    inference_input,
                    prediction,
                    prediction_type,
                    disable_preproc_auto_orient,
                )
            )
        except queue.Full:
            logger.warning(
                f"Dropping datapoint registered in Active Learning due to insufficient processing "
                f"capabilities."
            )

    def start_registration_thread(self) -> None:
        if self._registration_thread is not None:
            logger.warning(f"Registration thread already started.")
            return None
        logger.debug("Staring registration thread")
        self._registration_thread = Thread(target=self._consume_queue)
        self._registration_thread.start()

    def stop_registration_thread(self) -> None:
        if self._registration_thread is None:
            logger.warning("Registration thread is already stopped.")
            return None
        logger.debug("Stopping registration thread")
        self._task_queue.put(None)
        self._registration_thread.join()
        if self._registration_thread.is_alive():
            logger.warning(f"Registration thread stopping was unsuccessful.")
        self._registration_thread = None

    def _consume_queue(self) -> None:
        queue_closed = False
        while not queue_closed:
            queue_closed = self._consume_queue_task()

    def _consume_queue_task(self) -> bool:
        logger.debug("Consuming registration task")
        task = self._task_queue.get()
        logger.debug("Received registration task")
        if task is None:
            logger.debug("Terminating registration thread")
            self._task_queue.task_done()
            return True
        inference_input, prediction, prediction_type, disable_preproc_auto_orient = task
        try:
            self._execute_registration(
                inference_input=inference_input,
                prediction=prediction,
                prediction_type=prediction_type,
                disable_preproc_auto_orient=disable_preproc_auto_orient,
            )
        except Exception as error:
            # Error handling to be decided
            logger.warning(
                f"Error in datapoint registration for Active Learning. Details: {error}. "
                f"Error is suppressed in favour of normal operations of registration thread."
            )
        self._task_queue.task_done()
        return False

    def __enter__(self) -> "ThreadingActiveLearningMiddleware":
        self.start_registration_thread()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop_registration_thread()
