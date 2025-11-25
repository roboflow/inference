import gc
from functools import partial

DEFAULT_GARBAGE_COLLECT_FREQUENCY = 1000


# execute 'fn()' every 'frequency' times 'func' is called
def counter_exec(frequency, fn, exec_on_first=True, post_exec=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # count and determine trigger
            if exec_on_first:
                trigger = wrapper.counter == 0
            wrapper.counter = (wrapper.counter + 1) % frequency
            if not exec_on_first:
                trigger = wrapper.counter == 0

            # run
            if trigger and (not post_exec):
                fn()
            result = func(*args, **kwargs)
            if trigger and post_exec:
                fn()
            return result

        wrapper.counter = 0
        return wrapper

    return decorator


garbage_collect = partial(
    counter_exec,
    fn=gc.collect,
    exec_on_first=False,
    frequency=DEFAULT_GARBAGE_COLLECT_FREQUENCY,
    post_exec=False,
)
