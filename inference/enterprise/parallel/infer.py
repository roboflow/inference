import json
import logging
import time
from multiprocessing import shared_memory
from threading import Thread
from typing import Dict, List, Tuple

import numpy as np
from redis import ConnectionPool, Redis

from inference.core.entities.requests.inference import (
    InferenceRequest,
    request_from_type,
)
from inference.core.env import MAX_ACTIVE_MODELS, MAX_BATCH_SIZE, REDIS_HOST, REDIS_PORT
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.enterprise.parallel.tasks import postprocess
from inference.enterprise.parallel.utils import failure_handler, shm_manager
from inference.core.models.roboflow import RoboflowInferenceModel

logging.basicConfig(level=logging.INFO)

from inference.models.utils import ROBOFLOW_MODEL_TYPES

BATCH_SIZE = min(MAX_BATCH_SIZE, 128)


def get_requested_model_names(redis: Redis):
    request_counts = redis.hgetall("requests")
    model_names = [
        model_name for model_name, count in request_counts.items() if int(count) > 0
    ]
    return model_names


def get_batch(redis: Redis, model_names: List[str]) -> Tuple[List[Dict], str]:
    batch_sizes = [
        RoboflowInferenceModel.model_metadata_from_memcache_endpoint(m)["batch_size"]
        for m in model_names
    ]
    batch_sizes = [b if not isinstance(b, str) else BATCH_SIZE for b in batch_sizes]
    batches = [
        redis.zrange(f"infer:{m}", 0, b - 1, withscores=True)
        for m, b in zip(model_names, batch_sizes)
    ]
    now = time.time()
    average_ages = [np.mean([float(b[1]) - now for b in batch]) for batch in batches]
    lengths = [
        len(batch) / batch_size for batch, batch_size in zip(batches, batch_sizes)
    ]
    fitnesses = [age / 30 + length for age, length in zip(average_ages, lengths)]
    model_index = fitnesses.index(max(fitnesses))
    batch = batches[model_index]
    selected_model = model_names[model_index]
    redis.zrem(f"infer:{selected_model}", *[b[0] for b in batch])
    redis.hincrby(f"requests", selected_model, -len(batch))
    batch = [json.loads(b[0]) for b in batch]
    return batch, selected_model


def load_batch(
    batch: List[Dict[str, str]], shms: List[shared_memory.SharedMemory]
) -> Tuple[List[np.ndarray], List[Dict]]:
    images = []
    preproc_return_metadatas = []
    for b, shm in zip(batch, shms):
        image = np.ndarray(b["image_shape"], dtype=b["image_dtype"], buffer=shm.buf)
        images.append(image)
        preproc_return_metadatas.append(b["metadata"])
    return images, preproc_return_metadatas


def write_response(
    im_arrs: Tuple[np.ndarray, ...],
    request: InferenceRequest,
    preproc_return_metadata: Dict,
):
    shms = [shared_memory.SharedMemory(create=True, size=im.nbytes) for im in im_arrs]
    with shm_manager(*shms, close_on_success=False):
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

        postprocess.s(tuple(returns), request.dict(), preproc_return_metadata).delay()


class InferServer:
    def __init__(self) -> None:
        pool = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.redis = Redis(connection_pool=pool, decode_responses=True)
        model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
        model_manager = ModelManager(model_registry)
        self.model_manager = WithFixedSizeCache(
            model_manager, max_size=MAX_ACTIVE_MODELS
        )
        self.running = True
        self.wait = 0
        self.response_queue = []
        self.write_thread = Thread(target=self.write_responses)
        self.write_thread.start()

    def write_responses(self):
        while True:
            try:
                if not self.response_queue:
                    time.sleep(0.002)
                    continue
                response = self.response_queue.pop(0)
                write_response(*response)
            except:
                pass

    def infer_loop(self):
        while self.running:
            try:
                self.infer()
            except:
                continue

    def infer(self):
        start = time.perf_counter()
        model_names = get_requested_model_names(self.redis)
        if not model_names:
            if not self.wait:
                self.wait = time.time()
            time.sleep(0.002)
            return

        if self.wait:
            print(f"waited {(time.time() - self.wait):3f} seconds for batch")

        batch, model_id = get_batch(self.redis, model_names)
        print(f"Predicting on batch of size {len(batch)}")
        with failure_handler(self.redis, *[b["request"]["id"] for b in batch]):
            self.model_manager.add_model(model_id, batch[0]["request"]["api_key"])
            model_type = self.model_manager.get_task_type(model_id)
            for b in batch:
                request = request_from_type(model_type, b["request"])
                b["request"] = request
            metadata_processed = time.perf_counter()
            print(f"Took {(metadata_processed - start):3f} seconds to process metadata")
            with shm_manager(*[b["chunk_name"] for b in batch]) as shms:
                images, preproc_return_metadatas = load_batch(batch, shms)
                loaded = time.perf_counter()
                print(f"Took {(loaded - metadata_processed):3f} seconds to load batch")
                outputs = self.model_manager.predict(model_id, images)
                inferred = time.perf_counter()
                print(f"Took {(inferred - loaded):3f} seconds to infer")
                for output, b, metadata in zip(
                    zip(*outputs), batch, preproc_return_metadatas
                ):
                    self.response_queue.append((output, b["request"], metadata))
                written = time.perf_counter()
                print(f"Took {(written - inferred):3f} seconds to write responses")

        self.wait = 0


if __name__ == "__main__":
    InferServer().infer_loop()
