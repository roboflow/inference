import json
from multiprocessing import shared_memory
from typing import Dict, List, Tuple

import numpy as np
from celery import Celery
from redis import ConnectionPool, Redis

import inference.enterprise.parallel.celeryconfig
from inference.core.entities.requests.inference import (
    InferenceRequest,
    request_from_type,
)
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.env import REDIS_HOST, REDIS_PORT, STUB_CACHE_SIZE
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.managers.decorators.locked_load import (
    LockedLoadModelManagerDecorator,
)
from inference.core.managers.stub_loader import StubLoaderManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.enterprise.parallel.utils import (
    SUCCESS_STATE,
    TASK_RESULT_KEY,
    TASK_STATUS_KEY,
    failure_handler,
    shm_manager,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES

pool = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
app = Celery("tasks", broker=f"redis://{REDIS_HOST}:{REDIS_PORT}")
app.config_from_object(inference.enterprise.parallel.celeryconfig)
model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = StubLoaderManager(model_registry)
model_manager = WithFixedSizeCache(
    LockedLoadModelManagerDecorator(model_manager), max_size=STUB_CACHE_SIZE
)


def queue_infer_task(
    redis: Redis,
    shm_name: str,
    image: np.ndarray,
    request: InferenceRequest,
    preprocess_return_metadata: Dict,
):
    request.image.value = None
    return_vals = {
        "chunk_name": shm_name,
        "image_shape": image.shape,
        "image_dtype": image.dtype.name,
        "request": request.dict(),
        "metadata": preprocess_return_metadata,
    }
    return_vals = json.dumps(return_vals)
    pipe = redis.pipeline()
    pipe.zadd(f"infer:{request.model_id}", {return_vals: request.start})
    pipe.hincrby(f"requests", request.model_id, 1)
    pipe.execute()


def write_response(redis: Redis, response: InferenceResponse, request_id: str):
    response = response.json(exclude_none=True, by_alias=True)
    pipe = redis.pipeline()
    pipe.set(TASK_RESULT_KEY.format(request_id), response)
    pipe.set(TASK_STATUS_KEY.format(request_id), SUCCESS_STATE)
    pipe.execute()


@app.task(queue="pre")
def preprocess(request: Dict):
    r = Redis(connection_pool=pool, decode_responses=True)
    with failure_handler(r, request["id"]):
        model_manager.add_model(request["model_id"], request["api_key"])
        model_type = model_manager.get_task_type(request["model_id"])
        request = request_from_type(model_type, request)
        image, preprocess_return_metadata = model_manager.preprocess(
            request.model_id, request
        )
        image = image[0]
        shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        with shm_manager(shm, close_on_success=False):
            shared = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
            shared[:] = image[:]
            shm.close()
            queue_infer_task(r, shm.name, image, request, preprocess_return_metadata)


def load_outputs(
    shm_info_list: Dict, shms: List[shared_memory.SharedMemory]
) -> Tuple[np.ndarray, ...]:
    outputs = []
    for args, shm in zip(shm_info_list, shms):
        output = np.ndarray(
            [1] + args["image_shape"], dtype=args["image_dtype"], buffer=shm.buf
        )
        outputs.append(output)
    return tuple(outputs)


@app.task(queue="post")
def postprocess(
    shm_info_list: List[Dict], request: Dict, preproc_return_metadata: Dict
):
    r = Redis(connection_pool=pool, decode_responses=True)
    with failure_handler(r, request["id"]):
        with shm_manager(*[shm["chunk_name"] for shm in shm_info_list]) as shms:
            model_manager.add_model(request["model_id"], request["api_key"])
            model_type = model_manager.get_task_type(request["model_id"])
            request = request_from_type(model_type, request)

            outputs = load_outputs(shm_info_list, shms)

            request_dict = dict(**request.dict())
            model_id = request_dict.pop("model_id")

            results = model_manager.postprocess(
                model_id,
                outputs,
                preproc_return_metadata,
                **request_dict,
                return_image_dims=True,
            )

            response = model_manager.make_response(request.model_id, *results)[0]
            write_response(r, response, request.id)
