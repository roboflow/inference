import json
from multiprocessing import shared_memory
from typing import List, Dict

import numpy as np
from celery import Celery
from redis import ConnectionPool, Redis

from inference.core.entities.requests.inference import request_from_type
from inference.core.env import REDIS_HOST, REDIS_PORT, STUB_CACHE_SIZE
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.managers.decorators.locked_load import (
    LockedLoadModelManagerDecorator,
)
from inference.core.managers.stub_loader import StubLoaderManager
from inference.core.parallel.utils import failure_handler, shm_manager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES

pool = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
app = Celery("tasks", broker=f"redis://{REDIS_HOST}:{REDIS_PORT}")
model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = StubLoaderManager(model_registry)
model_manager = WithFixedSizeCache(
    LockedLoadModelManagerDecorator(model_manager), max_size=STUB_CACHE_SIZE
)


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
        img_dims = preprocess_return_metadata["img_dims"]
        image = image[0]
        shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        with shm_manager(shm, close_on_success=False):
            shared = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
            shared[:] = image[:]
            shm.close()
            request.image.value = None
            return_vals = {
                "chunk_name": shm.name,
                "image_shape": image.shape,
                "image_dtype": image.dtype.name,
                "image_dim": img_dims[0],
                "request": request.dict(),
                "metadata": preprocess_return_metadata,
            }
            return_vals = json.dumps(return_vals)
            pipe = r.pipeline()
            pipe.zadd(f"infer:{request.model_id}", {return_vals: request.start})
            pipe.hincrby(f"requests", request.model_id, 1)
            pipe.execute()


@app.task(queue="post")
def postprocess(
    shm_info_list: List[Dict], request: Dict, preproc_return_metadata: Dict
):
    r = Redis(connection_pool=pool, decode_responses=True)
    with failure_handler(r, request["id"]):
        with shm_manager(*[shm["name"] for shm in shm_info_list]) as shms:
            model_manager.add_model(request["model_id"], request["api_key"])
            model_type = model_manager.get_task_type(request["model_id"])
            request = request_from_type(model_type, request)
            outputs = []
            for args, shm in zip(shm_info_list, shms):
                output = np.ndarray(
                    [1] + args["image_shape"], dtype=args["image_dtype"], buffer=shm.buf
                )
                outputs.append(output)
            request_dict = dict(**request.dict())
            del request_dict["model_id"]
            results = model_manager.postprocess(
                request.model_id,
                tuple(outputs),
                preproc_return_metadata,
                **request_dict,
                return_image_dims=True,
            )

            results = model_manager.make_response(request.model_id, *results)[0]
            results = results.json(exclude_none=True, by_alias=True)
            pipe = r.pipeline()
            pipe.set(f"results:{request.id}", results)
            pipe.set(f"status:{request.id}", 1)
            pipe.execute()
