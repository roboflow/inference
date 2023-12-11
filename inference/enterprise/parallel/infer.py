import logging
import time
from asyncio import Queue as AioQueue
from dataclasses import asdict
from multiprocessing import shared_memory
from queue import Queue
from threading import Thread
from typing import Dict, List, Tuple

import numpy as np
import orjson
from redis import ConnectionPool, Redis

from inference.core.entities.requests.inference import (
    InferenceRequest,
    request_from_type,
)
from inference.core.env import MAX_ACTIVE_MODELS, MAX_BATCH_SIZE, REDIS_HOST, REDIS_PORT
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.models.roboflow import RoboflowInferenceModel
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.enterprise.parallel.tasks import postprocess
from inference.enterprise.parallel.utils import (
    SharedMemoryMetadata,
    failure_handler,
    shm_manager,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

from inference.models.utils import ROBOFLOW_MODEL_TYPES

BATCH_SIZE = MAX_BATCH_SIZE
if BATCH_SIZE == float("inf"):
    BATCH_SIZE = 32
AGE_TRADEOFF_SECONDS_FACTOR = 30


class InferServer:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
        model_manager = ModelManager(model_registry)
        self.model_manager = WithFixedSizeCache(
            model_manager, max_size=MAX_ACTIVE_MODELS
        )
        self.running = True
        self.response_queue = Queue()
        self.write_thread = Thread(target=self.write_responses)
        self.write_thread.start()
        self.batch_queue = Queue(maxsize=1)
        self.infer_thread = Thread(target=self.infer)
        self.infer_thread.start()

    def write_responses(self):
        while True:
            try:
                response = self.response_queue.get()
                write_infer_arrays_and_launch_postprocess(*response)
            except Exception as error:
                logger.warning(
                    f"Encountered error while writiing response:\n" + str(error)
                )

    def infer_loop(self):
        while self.running:
            try:
                model_names = get_requested_model_names(self.redis)
                if not model_names:
                    time.sleep(0.001)
                    continue
                self.get_batch(model_names)
            except Exception as error:
                logger.warning("Encountered error in infer loop:\n" + str(error))
                continue

    def infer(self):
        while True:
            model_id, images, batch, preproc_return_metadatas = self.batch_queue.get()
            outputs = self.model_manager.predict(model_id, images)
            for output, b, metadata in zip(
                zip(*outputs), batch, preproc_return_metadatas
            ):
                self.response_queue.put_nowait((output, b["request"], metadata))

    def get_batch(self, model_names):
        start = time.perf_counter()
        batch, model_id = get_batch(self.redis, model_names)
        logger.info(f"Inferring: model<{model_id}> batch_size<{len(batch)}>")
        with failure_handler(self.redis, *[b["request"]["id"] for b in batch]):
            self.model_manager.add_model(model_id, batch[0]["request"]["api_key"])
            model_type = self.model_manager.get_task_type(model_id)
            for b in batch:
                request = request_from_type(model_type, b["request"])
                b["request"] = request
                b["shm_metadata"] = SharedMemoryMetadata(**b["shm_metadata"])

            metadata_processed = time.perf_counter()
            logger.info(
                f"Took {(metadata_processed - start):3f} seconds to process metadata"
            )
            with shm_manager(
                *[b["shm_metadata"].shm_name for b in batch], unlink_on_success=True
            ) as shms:
                images, preproc_return_metadatas = load_batch(batch, shms)
                loaded = time.perf_counter()
                logger.info(
                    f"Took {(loaded - metadata_processed):3f} seconds to load batch"
                )
                self.batch_queue.put(
                    (model_id, images, batch, preproc_return_metadatas)
                )


def get_requested_model_names(redis: Redis) -> List[str]:
    request_counts = redis.hgetall("requests")
    model_names = [
        model_name for model_name, count in request_counts.items() if int(count) > 0
    ]
    return model_names


def get_batch(redis: Redis, model_names: List[str]) -> Tuple[List[Dict], str]:
    """
    Run a heuristic to select the best batch to infer on
    redis[Redis]: redis client
    model_names[List[str]]: list of models with nonzero number of requests
    returns:
        Tuple[List[Dict], str]
        List[Dict] represents a batch of request dicts
        str is the model id
    """
    batch_sizes = [
        RoboflowInferenceModel.model_metadata_from_memcache_endpoint(m)["batch_size"]
        for m in model_names
    ]
    batch_sizes = [b if not isinstance(b, str) else BATCH_SIZE for b in batch_sizes]
    batches = [
        redis.zrange(f"infer:{m}", 0, b - 1, withscores=True)
        for m, b in zip(model_names, batch_sizes)
    ]
    model_index = select_best_inference_batch(batches, batch_sizes)
    batch = batches[model_index]
    selected_model = model_names[model_index]
    redis.zrem(f"infer:{selected_model}", *[b[0] for b in batch])
    redis.hincrby(f"requests", selected_model, -len(batch))
    batch = [orjson.loads(b[0]) for b in batch]
    return batch, selected_model


def select_best_inference_batch(batches, batch_sizes):
    now = time.time()
    average_ages = [np.mean([float(b[1]) - now for b in batch]) for batch in batches]
    lengths = [
        len(batch) / batch_size for batch, batch_size in zip(batches, batch_sizes)
    ]
    fitnesses = [
        age / AGE_TRADEOFF_SECONDS_FACTOR + length
        for age, length in zip(average_ages, lengths)
    ]
    model_index = fitnesses.index(max(fitnesses))
    return model_index


def load_batch(
    batch: List[Dict[str, str]], shms: List[shared_memory.SharedMemory]
) -> Tuple[List[np.ndarray], List[Dict]]:
    images = []
    preproc_return_metadatas = []
    for b, shm in zip(batch, shms):
        shm_metadata: SharedMemoryMetadata = b["shm_metadata"]
        image = np.ndarray(
            shm_metadata.array_shape, dtype=shm_metadata.array_dtype, buffer=shm.buf
        ).copy()
        images.append(image)
        preproc_return_metadatas.append(b["preprocess_metadata"])
    return images, preproc_return_metadatas


def write_infer_arrays_and_launch_postprocess(
    arrs: Tuple[np.ndarray, ...],
    request: InferenceRequest,
    preproc_return_metadata: Dict,
):
    """Write inference results to shared memory and launch the postprocessing task"""
    shms = [shared_memory.SharedMemory(create=True, size=arr.nbytes) for arr in arrs]
    with shm_manager(*shms):
        shm_metadatas = []
        for arr, shm in zip(arrs, shms):
            shared = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shared[:] = arr[:]
            shm_metadata = SharedMemoryMetadata(
                shm_name=shm.name, array_shape=arr.shape, array_dtype=arr.dtype.name
            )
            shm_metadatas.append(asdict(shm_metadata))

        postprocess.s(
            tuple(shm_metadatas), request.dict(), preproc_return_metadata
        ).delay()


if __name__ == "__main__":
    pool = ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis = Redis(connection_pool=pool)
    InferServer(redis).infer_loop()
