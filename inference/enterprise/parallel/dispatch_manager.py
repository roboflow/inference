import asyncio
import json
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, Coroutine, Optional

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

NOT_FINISHED_RESPONSE = "===NOTFINISHED==="


class ResultsChecker:
    """
    Class responsible for queuing asyncronous inference runs,
    keeping track of running requests, and awaiting their results.
    """

    def __init__(self):
        self.tasks = []
        self.dones = dict()
        self.errors = dict()
        self.running = True

    def add_redis(self, r: Redis):
        """
        After instantiation, give results checker a redis object
        """
        self.r = r

    async def add_task(self, t, request):
        """
        Wait until there's available cylce to queue a task.
        When there are cycles, add the task's id to a list to keep track of its results,
        launch the preprocess celeryt task, set the task's status to in progress in redis.
        """
        interval = 0.1
        while len(self.tasks) > NUM_PARALLEL_TASKS:
            await asyncio.sleep(interval)
        self.tasks.append(t)
        preprocess.s(request.dict()).delay()
        self.r.set(TASK_STATUS_KEY.format(t), INITIAL_STATE)

    def check_task(self, t: str) -> Any:
        """
        Check the done tasks and errored tasks for this task id.
        """
        if t in self.dones:
            return self.dones.pop(t)
        if t in self.errors:
            message = self.errors.pop(t)
            raise Exception(message)
        return NOT_FINISHED_RESPONSE

    async def loop(self):
        """
        Main loop. Check all in progress tasks for their status, and if their status is final,
        (either failure or success) then add their results to the appropriate results dictionary.
        """
        interval = 0.1
        while self.running:
            tasks = [t for t in self.tasks]
            task_names = [TASK_STATUS_KEY.format(id_) for id_ in tasks]
            donenesses = self.r.mget(task_names)
            donenesses = [int(d) for d in donenesses]
            for id_, doneness in zip(tasks, donenesses):
                if doneness in [SUCCESS_STATE, FAILURE_STATE]:
                    pipe = self.r.pipeline()
                    pipe.get(TASK_RESULT_KEY.format(id_))
                    pipe.delete(TASK_RESULT_KEY.format(id_))
                    pipe.delete(TASK_STATUS_KEY.format(id_))
                    result, _, _ = pipe.execute()
                    self.tasks.remove(id_)
                    if doneness == SUCCESS_STATE:
                        self.dones[id_] = result
                    if doneness == FAILURE_STATE:
                        self.errors[id_] = result
            await asyncio.sleep(interval)

    async def wait_for_response(self, key):
        interval = 0.1
        while True:
            result = self.check_task(key)
            if result is not NOT_FINISHED_RESPONSE:
                return result
            await asyncio.sleep(interval)


@dataclass
class DispatchModelManager(ModelManager):
    def __post_init__(self):
        if REDIS_HOST is None:
            raise ValueError(
                "Set REDIS_HOST and REDIS_PORT to use DispatchModelManager"
            )

        self.redis: Redis = Redis(REDIS_HOST, REDIS_PORT, decode_responses=True)

    async def model_infer(self, model_id: str, request: InferenceRequest):
        assert model_id == request.model_id
        assert request.api_key is not None
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

        if list_mode:
            return responses
        return responses[0]

    def add_checker(self, checker):
        self._checker = checker

    @property
    def checker(self) -> ResultsChecker:
        try:
            return self._checker
        except AttributeError:
            raise AttributeError(
                "Call self.add_checker with a results checker before using"
            )

    def add_model(
        self, model_id: str, api_key: str, model_id_alias: str = None
    ) -> None:
        pass

    def __contains__(self, model_id: str) -> bool:
        return True

    def get_task_type(self, model_id: str, api_key: str = None) -> str:
        return get_model_type(model_id, api_key)[0]
