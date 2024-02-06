import asyncio
from asyncio import BoundedSemaphore
from time import perf_counter, time
from typing import Any, Dict, List, Optional

import orjson
from redis.asyncio import Redis

from inference.core.entities.requests.inference import (
    InferenceRequest,
    request_from_type,
)
from inference.core.entities.responses.inference import response_from_type
from inference.core.env import NUM_PARALLEL_TASKS
from inference.core.managers.base import ModelManager
from inference.core.registries.base import ModelRegistry
from inference.core.registries.roboflow import get_model_type
from inference.enterprise.parallel.tasks import preprocess
from inference.enterprise.parallel.utils import FAILURE_STATE, SUCCESS_STATE


class ResultsChecker:
    """
    Class responsible for queuing asyncronous inference runs,
    keeping track of running requests, and awaiting their results.
    """

    def __init__(self, redis: Redis):
        self.tasks: Dict[str, asyncio.Event] = {}
        self.dones = dict()
        self.errors = dict()
        self.running = True
        self.redis = redis
        self.semaphore: BoundedSemaphore = BoundedSemaphore(NUM_PARALLEL_TASKS)

    async def add_task(self, task_id: str, request: InferenceRequest):
        """
        Wait until there's available cylce to queue a task.
        When there are cycles, add the task's id to a list to keep track of its results,
        launch the preprocess celeryt task, set the task's status to in progress in redis.
        """
        await self.semaphore.acquire()
        self.tasks[task_id] = asyncio.Event()
        preprocess.s(request.dict()).delay()

    def get_result(self, task_id: str) -> Any:
        """
        Check the done tasks and errored tasks for this task id.
        """
        if task_id in self.dones:
            return self.dones.pop(task_id)
        elif task_id in self.errors:
            message = self.errors.pop(task_id)
            raise Exception(message)
        else:
            raise RuntimeError(
                "Task result not found in either success or error dict. Unreachable"
            )

    async def loop(self):
        """
        Main loop. Check all in progress tasks for their status, and if their status is final,
        (either failure or success) then add their results to the appropriate results dictionary.
        """
        async with self.redis.pubsub() as pubsub:
            await pubsub.subscribe("results")
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                message = orjson.loads(message["data"])
                task_id = message.pop("task_id")
                if task_id not in self.tasks:
                    continue
                self.semaphore.release()
                status = message.pop("status")
                if status == FAILURE_STATE:
                    self.errors[task_id] = message["payload"]
                elif status == SUCCESS_STATE:
                    self.dones[task_id] = message["payload"]
                else:
                    raise RuntimeError(
                        "Task result not found in possible states. Unreachable"
                    )
                self.tasks[task_id].set()
                await asyncio.sleep(0)

    async def wait_for_response(self, key: str):
        event = self.tasks[key]
        await event.wait()
        del self.tasks[key]
        return self.get_result(key)


class DispatchModelManager(ModelManager):
    def __init__(
        self,
        model_registry: ModelRegistry,
        checker: ResultsChecker,
        models: Optional[dict] = None,
    ):
        super().__init__(model_registry, models)
        self.checker = checker

    async def model_infer(self, model_id: str, request: InferenceRequest, **kwargs):
        if request.visualize_predictions:
            raise NotImplementedError("Visualisation of prediction is not supported")
        request.start = time()
        t = perf_counter()
        task_type = self.get_task_type(model_id, request.api_key)

        list_mode = False
        if isinstance(request.image, list):
            list_mode = True
            request_dict = request.dict()
            images = request_dict.pop("image")
            del request_dict["id"]
            requests = [
                request_from_type(task_type, dict(**request_dict, image=image))
                for image in images
            ]
        else:
            requests = [request]

        start_task_awaitables = []
        results_awaitables = []
        for r in requests:
            start_task_awaitables.append(self.checker.add_task(r.id, r))
            results_awaitables.append(self.checker.wait_for_response(r.id))

        await asyncio.gather(*start_task_awaitables)
        response_jsons = await asyncio.gather(*results_awaitables)
        responses = []
        for response_json in response_jsons:
            response = response_from_type(task_type, response_json)
            response.time = perf_counter() - t
            responses.append(response)

        if list_mode:
            return responses
        return responses[0]

    def add_model(
        self, model_id: str, api_key: str, model_id_alias: str = None
    ) -> None:
        pass

    def __contains__(self, model_id: str) -> bool:
        return True

    def get_task_type(self, model_id: str, api_key: str = None) -> str:
        return get_model_type(model_id, api_key)[0]
