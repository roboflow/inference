from celery import Celery
from multiprocessing import shared_memory
import numpy as np
from PIL import Image
import io
import base64
from redis import Redis, ConnectionPool
import json
import time
from inference.models.utils import get_roboflow_model
from inference.core.data_models import (
    InferenceRequest,
    InferenceResponse,
    request_from_type,
)
from inference.core.managers.stub_loader import StubLoaderManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.managers.decorators.locked_load import (
    LockedLoadModelManagerDecorator,
)
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.models.types import PreprocessReturnMetadata

from inference.models.utils import ROBOFLOW_MODEL_TYPES
from contextlib import contextmanager


pool = ConnectionPool(host="localhost", port=6379, decode_responses=True)
app = Celery("tasks", broker="redis://localhost:6379")
model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = StubLoaderManager(model_registry)
model_manager = WithFixedSizeCache(
    LockedLoadModelManagerDecorator(model_manager), max_size=256
)


@app.task(queue="pre")
def preprocess(request):
    model_manager.add_model(request["model_id"], request["api_key"])
    model_type = model_manager.get_task_type(request["model_id"])
    request = request_from_type(model_type)(**request)
    image, preprocess_return_metadata = model_manager.preprocess(
        request.model_id, request
    )
    img_dims = preprocess_return_metadata["img_dims"]
    image = image[0]
    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
    shared = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
    shared[:] = image[:]
    shm.close()
    return_vals = {
        "chunk_name": shm.name,
        "image_shape": image.shape,
        "image_dtype": image.dtype.name,
        "image_dim": img_dims[0],
        "request": request.dict(),
        "metadata": preprocess_return_metadata,
    }
    return_vals = json.dumps(return_vals)
    r = Redis(connection_pool=pool, decode_responses=True)
    pipe = r.pipeline()
    pipe.zadd(f"infer:{request.model_id}", {return_vals: request.start})
    pipe.hincrby(f"requests", request.model_id, 1)
    pipe.execute()


@app.task(queue="post")
def postprocess(arg_list, request, metadata):
    model_manager.add_model(request["model_id"], request["api_key"])
    model_type = model_manager.get_task_type(request["model_id"])
    request = request_from_type(model_type)(**request)
    outputs = []
    for args in arg_list:
        shm = shared_memory.SharedMemory(name=args["chunk_name"])
        output = np.ndarray(
            [1] + args["image_shape"], dtype=args["image_dtype"], buffer=shm.buf
        )
        outputs.append(output)
    print(metadata)
    request_dict = dict(**request.dict())
    del request_dict["model_id"]
    results = model_manager.postprocess(
        request.model_id,
        tuple(outputs),
        metadata,
        **request_dict,
        return_image_dims=True
    )

    dim = metadata["img_dims"]
    results = model_manager.make_response(request.model_id, *results)[0]
    results = results.json(exclude_none=True, by_alias=True)
    r = Redis(connection_pool=pool, decode_responses=True)
    pipe = r.pipeline()
    pipe.set(f"results:{request.id}", results)
    pipe.set(f"status:{request.id}", 1)
    pipe.execute()
    shm.close()
    shm.unlink()
