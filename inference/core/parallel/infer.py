import base64
import io
import json
import logging
import time
from multiprocessing import shared_memory
from time import perf_counter
from typing import Tuple

import numpy as np
from redis import ConnectionPool, Redis

from inference.core.entities.requests.inference import request_from_type
from inference.core.env import MAX_ACTIVE_MODELS, MAX_BATCH_SIZE
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.parallel.tasks import postprocess
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.parallel.utils import shm_closer, failure_handler

pool = ConnectionPool(host="localhost", port=6379, decode_responses=True)
r = Redis(connection_pool=pool, decode_responses=True)
logging.basicConfig(level=logging.INFO)

from inference.models.utils import ROBOFLOW_MODEL_TYPES

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = ModelManager(model_registry)
model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)


BATCH_SIZE = min(MAX_BATCH_SIZE, 128)


class InferServer:
    def get_batch(self, model_names):
        batches = [
            r.zrange(f"infer:{m}", 0, BATCH_SIZE - 1, withscores=True)
            for m in model_names
        ]
        now = time.time()
        average_ages = [
            np.mean([float(b[1]) - now for b in batch]) for batch in batches
        ]
        lengths = [len(batch) / BATCH_SIZE for batch in batches]
        fitnesses = [age / 30 + length for age, length in zip(average_ages, lengths)]
        model_index = fitnesses.index(max(fitnesses))
        batch = batches[model_index]
        selected_model = model_names[model_index]
        r.zrem(f"infer:{selected_model}", *[b[0] for b in batch])
        r.hincrby(f"requests", selected_model, -len(batch))
        batch = [json.loads(b[0]) for b in batch]
        return batch, selected_model

    def infer_loop(self):
        while True:
            try:
                request_counts = r.hgetall("requests")
                model_names = [
                    model_name
                    for model_name, count in request_counts.items()
                    if int(count) > 0
                ]
                if not model_names:
                    time.sleep(0.005)
                    continue
                batch, model_id = self.get_batch(model_names)
                with failure_handler(r, *[b["request"]["id"] for b in batch]):
                    model_manager.add_model(model_id, batch[0]["request"]["api_key"])
                    model_type = model_manager.get_task_type(model_id)
                    for b in batch:
                        request = request_from_type(model_type, b["request"])
                        b["request"] = request
                    shms = []
                    for b in batch:
                        shm = shared_memory.SharedMemory(name=b["chunk_name"])
                        shms.append(shm)
                    with shm_closer(*shms):
                        images = []
                        metadatas = []
                        for b, shm in zip(batch, shms):
                            image = np.ndarray(
                                b["image_shape"], dtype=b["image_dtype"], buffer=shm.buf
                            )
                            images.append(image)
                            metadatas.append(b["metadata"])

                        outputs = model_manager.predict(model_id, images)

                        del images
                        for output, b, metadata in zip(zip(*outputs), batch, metadatas):
                            self.write_response(output, b["request"], metadata)
            except:
                continue

    def write_response(self, im_arrs: Tuple[np.ndarray, ...], request, metadata):
        shms = []
        for im_arr in im_arrs:
            shm = shared_memory.SharedMemory(create=True, size=im_arr.nbytes)
            shms.append(shm)

        with shm_closer(*shms, on_success=False):
            returns = []
            for im_arr, shm in zip(im_arrs, shms):
                shared = np.ndarray(im_arr.shape, dtype=im_arr.dtype, buffer=shm.buf)
                shared[:] = im_arr[:]
                return_val = {
                    "chunk_name": shm.name,
                    "image_shape": im_arr.shape,
                    "image_dtype": im_arr.dtype.name,
                }
                shm.close()
                returns.append(return_val)

            postprocess.s(tuple(returns), request.dict(), metadata).delay()


if __name__ == "__main__":
    InferServer().infer_loop()
