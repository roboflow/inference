import sys
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Union

from redis import Redis

TASK_RESULT_KEY = "results:{}"
TASK_STATUS_KEY = "status:{}"
SUCCESS_STATE = 1
INITIAL_STATE = 0
FAILURE_STATE = -1


@contextmanager
def failure_handler(redis: Redis, *request_ids):
    """
    Context manager that updates the status/results key in redis with exception
    info on failure.
    """
    try:
        yield
    except:
        ex_type, ex_value, _ = sys.exc_info()
        message = ex_type.__name__ + ": " + str(ex_value)
        for request_id in request_ids:
            redis.set(TASK_RESULT_KEY.format(request_id), message)
            redis.set(TASK_STATUS_KEY.format(request_id), FAILURE_STATE)
        raise


@contextmanager
def shm_manager(
    *shms: Union[str, shared_memory.SharedMemory],
    close_on_failure=True,
    close_on_success=True
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
            except BaseException as E:
                errors.append(E)
            if errors:
                raise Exception(errors)

        yield loaded_shms
    except:
        if close_on_failure:
            for shm in loaded_shms:
                shm.close()
                shm.unlink()
        raise
    else:
        if close_on_success:
            for shm in loaded_shms:
                shm.close()
                shm.unlink()
