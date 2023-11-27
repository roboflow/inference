import asyncio
import json
from time import perf_counter, time
from typing import Any, List, Dict, Optional

from redis import Redis

from inference.core.entities.requests.inference import (
    InferenceRequest,
    request_from_type,
)
from inference.core.entities.responses.inference import response_from_type
from inference.core.env import NUM_PARALLEL_TASKS, REDIS_HOST, REDIS_PORT
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import get_model_type
from inference.enterprise.parallel.tasks import preprocess
from inference.enterprise.parallel.utils import (
    FAILURE_STATE,
    INITIAL_STATE,
    SUCCESS_STATE,
    TASK_RESULT_KEY,
    TASK_STATUS_KEY,
)
from inference.core.registries.base import ModelRegistry

NOT_FINISHED_RESPONSE = "===NOTFINISHED==="


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

    async def add_task(self, task_id: str, request: InferenceRequest):
        """
        Wait until there's available cylce to queue a task.
        When there are cycles, add the task's id to a list to keep track of its results,
        launch the preprocess celeryt task, set the task's status to in progress in redis.
        """
        interval = 0.1
        while len(self.tasks) > NUM_PARALLEL_TASKS:
            await asyncio.sleep(interval)
        self.tasks[task_id] = asyncio.Event()
        preprocess.s(request.dict()).delay()
        self.redis.set(TASK_STATUS_KEY.format(task_id), INITIAL_STATE)

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
        interval = 0.1
        while self.running:
            self.check_tasks([t for t in self.tasks])
            await asyncio.sleep(interval)

    def check_tasks(self, task_ids: List[str]):
        task_names = [TASK_STATUS_KEY.format(id_) for id_ in task_ids]
        donenesses = self.redis.mget(task_names)
        donenesses = [int(d) for d in donenesses]
        for id_, doneness in zip(task_ids, donenesses):
            if doneness in [SUCCESS_STATE, FAILURE_STATE]:
                self.handle_result(id_, doneness)

    def handle_result(self, task_id: str, doneness: int):
        pipe = self.redis.pipeline()
        pipe.get(TASK_RESULT_KEY.format(task_id))
        pipe.delete(TASK_RESULT_KEY.format(task_id))
        pipe.delete(TASK_STATUS_KEY.format(task_id))
        result, _, _ = pipe.execute()
        if doneness == SUCCESS_STATE:
            self.dones[task_id] = result
        elif doneness == FAILURE_STATE:
            self.errors[task_id] = result
        else:
            raise RuntimeError(f"Unspecified state {doneness}. Unreachable")
        self.tasks.pop(task_id).set()

    async def wait_for_response(self, key: str):
        await self.tasks[key].wait()
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
        response_strings = await asyncio.gather(*results_awaitables)
        responses = []
        for response_json_string in response_strings:
            response = response_from_type(task_type, json.loads(response_json_string))
            response.time = perf_counter() - t
            responses.append(response)

        if request.visualize_predictions:
            for response in responses:
                response.visualization = self.draw_predictions(request, response)

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
