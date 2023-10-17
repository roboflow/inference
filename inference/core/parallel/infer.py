import base64
import io
import json
import logging
import time
from multiprocessing import shared_memory
from time import perf_counter
from typing import Tuple

import numpy as np
from PIL import Image
from redis import ConnectionPool, Redis

from inference.core.cache import cache
from inference.core.data_models import request_from_type
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.parallel.tasks import postprocess
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import get_roboflow_model

pool = ConnectionPool(host="localhost", port=6379, decode_responses=True)
r = Redis(connection_pool=pool, decode_responses=True)
BATCH_SIZE = 64
logging.basicConfig(level=logging.INFO)

from inference.models.utils import ROBOFLOW_MODEL_TYPES

model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
model_manager = ModelManager(model_registry)
model_manager = WithFixedSizeCache(model_manager, max_size=8)


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
        model_id = selected_model
        model_manager.add_model(model_id, batch[0]["request"]["api_key"])
        model_type = model_manager.get_task_type(model_id)
        for b in batch:
            request = request_from_type(model_type)(**b["request"])
            b["request"] = request
        return batch

    def infer_loop(self):
        while True:
            request_counts = r.hgetall("requests")
            model_names = [
                model_name
                for model_name, count in request_counts.items()
                if int(count) > 0
            ]
            if not model_names:
                time.sleep(0.005)
                continue
            batch = self.get_batch(model_names)
            images = []
            metadatas = []
            shms = []
            for b in batch:
                shm = shared_memory.SharedMemory(name=b["chunk_name"])
                image = np.ndarray(
                    b["image_shape"], dtype=b["image_dtype"], buffer=shm.buf
                )
                images.append(image)
                metadatas.append(b["metadata"])
                shms.append(shm)
            outputs = model_manager.predict(batch[0]["request"].model_id, images)
            del images
            for shm in shms:
                shm.close()
                shm.unlink()
            for output, b, metadata in zip(outputs, batch, metadatas):
                info = self.write_response(output)
                postprocess.s(info, b["request"].dict(), metadata).delay()

    def write_response(self, im_arrs: Tuple[np.ndarray, ...]):
        returns = list()
        for im_arr in im_arrs:
            shm2 = shared_memory.SharedMemory(create=True, size=im_arr.nbytes)
            shared = np.ndarray(im_arr.shape, dtype=im_arr.dtype, buffer=shm2.buf)
            shared[:] = im_arr[:]
            return_val = {
                "chunk_name": shm2.name,
                "image_shape": im_arr.shape,
                "image_dtype": im_arr.dtype.name,
            }
            shm2.close()
            returns.append(return_val)
        return tuple(returns)


if __name__ == "__main__":
    InferServer().infer_loop()
