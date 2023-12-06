import json
from dataclasses import asdict
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
    SharedMemoryMetadata,
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


@app.task(queue="pre")
def preprocess(request: Dict):
    redis_client = Redis(connection_pool=pool)
    with failure_handler(redis_client, request["id"]):
        model_manager.add_model(request["model_id"], request["api_key"])
        model_type = model_manager.get_task_type(request["model_id"])
        request = request_from_type(model_type, request)
        image, preprocess_return_metadata = model_manager.preprocess(
            request.model_id, request
        )
        # multi image requests are split into single image requests upstream and rebatched later
        image = image[0]
        request.image.value = None  # avoid writing image again since it's in memory
        shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        with shm_manager(shm):
            shared = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
            shared[:] = image[:]
            shm_metadata = SharedMemoryMetadata(shm.name, image.shape, image.dtype.name)
            queue_infer_task(
                redis_client, shm_metadata, request, preprocess_return_metadata
            )


@app.task(queue="post")
def postprocess(
    shm_info_list: Tuple[Dict], request: Dict, preproc_return_metadata: Dict
):
    redis_client = Redis(connection_pool=pool)
    shm_info_list: List[SharedMemoryMetadata] = [
        SharedMemoryMetadata(**metadata) for metadata in shm_info_list
    ]
    with failure_handler(redis_client, request["id"]):
        with shm_manager(
            *[shm_metadata.shm_name for shm_metadata in shm_info_list],
            unlink_on_success=True,
        ) as shms:
            model_manager.add_model(request["model_id"], request["api_key"])
            model_type = model_manager.get_task_type(request["model_id"])
            request = request_from_type(model_type, request)

            outputs = load_outputs(shm_info_list, shms)

            request_dict = dict(**request.dict())
            model_id = request_dict.pop("model_id")

            response = model_manager.postprocess(
                model_id,
                outputs,
                preproc_return_metadata,
                **request_dict,
                return_image_dims=True,
            )[0]

            write_response(redis_client, response, request.id)


def load_outputs(
    shm_info_list: List[SharedMemoryMetadata], shms: List[shared_memory.SharedMemory]
) -> Tuple[np.ndarray, ...]:
    outputs = []
    for args, shm in zip(shm_info_list, shms):
        output = np.ndarray(
            [1] + args.array_shape, dtype=args.array_dtype, buffer=shm.buf
        )
        outputs.append(output)
    return tuple(outputs)


def queue_infer_task(
    redis: Redis,
    shm_metadata: SharedMemoryMetadata,
    request: InferenceRequest,
    preprocess_return_metadata: Dict,
):
    return_vals = {
        "shm_metadata": asdict(shm_metadata),
        "request": request.dict(),
        "preprocess_metadata": preprocess_return_metadata,
    }
    return_vals = json.dumps(return_vals)
    pipe = redis.pipeline()
    pipe.zadd(f"infer:{request.model_id}", {return_vals: request.start})
    pipe.hincrby(f"requests", request.model_id, 1)
    pipe.execute()


def write_response(redis: Redis, response: InferenceResponse, request_id: str):
    response = response.dict(exclude_none=True, by_alias=True)
    redis.publish(
        f"results",
        json.dumps(
            {"status": SUCCESS_STATE, "task_id": request_id, "payload": response}
        ),
    )
