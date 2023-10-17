import asyncio
import json
from dataclasses import dataclass
from time import perf_counter, time
from typing import Any, Coroutine, Optional

from redis import Redis

from inference.core.data_models import (
    InferenceRequest,
    InferenceResponse,
    response_from_type,
)
from inference.core.env import REDIS_HOST, REDIS_PORT
from inference.core.managers.base import ModelManager
from inference.core.parallel.tasks import preprocess
from inference.core.registries.roboflow import get_model_type

TASK_RESULT_KEY = "results:{}"
TASK_STATUS_KEY = "status:{}"
FINAL_STATE = 1
INITIAL_STATE = 0

NOT_FINISHED_RESPONSE = "===NOTFINISHED==="


class ResultsChecker:
    def __init__(self):
        self.tasks = []
        self.dones = dict()
        self.running = True

    def add_redis(self, r: Redis):
        self.r = r

    def add_task(self, t):
        self.tasks.append(t)

    def check_task(self, t):
        if t in self.dones:
            return self.dones.pop(t)
        return NOT_FINISHED_RESPONSE

    async def loop(self):
        interval = 0.1
        while self.running:
            tasks = [t for t in self.tasks]
            task_names = [TASK_STATUS_KEY.format(id_) for id_ in tasks]
            donenesses = [self.r.get(t) for t in task_names]
            donenesses = [int(d) for d in donenesses]
            for id_, doneness in zip(tasks, donenesses):
                if doneness == FINAL_STATE:
                    pipe = self.r.pipeline()
                    pipe.get(TASK_RESULT_KEY.format(id_))
                    pipe.delete(TASK_RESULT_KEY.format(id_))
                    pipe.delete(TASK_STATUS_KEY.format(id_))
                    result, _, _ = pipe.execute()
                    self.tasks.remove(id_)
                    self.dones[id_] = result
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
        self.redis.set(TASK_STATUS_KEY.format(request.id), INITIAL_STATE)
        self.checker.add_task(request.id)
        preprocess.s(request.dict()).delay()
        response_json_string = await self.checker.wait_for_response(request.id)
        response = response_from_type(self.get_task_type(model_id))(
            **json.loads(response_json_string)
        )
        response.time = perf_counter() - t
        return response

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
