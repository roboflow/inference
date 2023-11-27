from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Union

from redis import Redis

TASK_RESULT_KEY = "results:{}"
TASK_STATUS_KEY = "status:{}"
SUCCESS_STATE = 1
INITIAL_STATE = 0
FAILURE_STATE = -1


@contextmanager
def failure_handler(redis: Redis, *request_ids: str):
    """
    Context manager that updates the status/results key in redis with exception
    info on failure.
    """
    try:
        yield
    except Exception as error:
        message = type(error).__name__ + ": " + str(error)
        for request_id in request_ids:
            redis.set(TASK_RESULT_KEY.format(request_id), message)
            redis.set(TASK_STATUS_KEY.format(request_id), FAILURE_STATE)
        raise


@contextmanager
def shm_manager(
    *shms: Union[str, shared_memory.SharedMemory], unlink_on_success: bool = False
):
    """Context manager that closes and frees shared memory objects."""
    try:
        loaded_shms = []
        for shm in shms:
            errors = []
            try:
                if isinstance(shm, str):
                    shm = shared_memory.SharedMemory(name=shm)
                loaded_shms.append(shm)
            except BaseException as error:
                errors.append(error)
            if errors:
                raise Exception(errors)

        yield loaded_shms
    except:
        for shm in loaded_shms:
            shm.close()
            shm.unlink()
        raise
    else:
        for shm in loaded_shms:
            shm.close()
            if unlink_on_success:
                shm.unlink()


@dataclass
class SharedMemoryMetadata:
    """Info needed to load array from shared memory"""

    shm_name: str
    array_shape: List[int]
    array_dtype: str
