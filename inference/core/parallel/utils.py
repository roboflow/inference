from contextlib import contextmanager
from redis import Redis
import sys

TASK_RESULT_KEY = "results:{}"
TASK_STATUS_KEY = "status:{}"
FINAL_STATE = 1
INITIAL_STATE = 0
FAILURE_STATE = -1

@contextmanager
def failure_handler(redis: Redis, *request_ids):
    try:
        yield
    except:
        ex_type, ex_value, _ = sys.exc_info()
        message = ex_type.__name__ + ": " + str(ex_value)
        for request_id in request_ids:
            redis.set(TASK_STATUS_KEY.format(request_id), FAILURE_STATE)
            redis.set(TASK_RESULT_KEY.format(request_id), message)
        raise

@contextmanager
def shm_closer(*shms, on_failure=True, on_success=True):
    try:
        yield
    except:
        if on_failure:
            for shm in shms:
                shm.close()
                shm.unlink()
        raise
    else:
        if on_success:
            for shm in shms:
                shm.close()
                shm.unlink()