import datetime as dt
from loguru import logger


def timeit(
    level: str = "DEBUG",
    message_template: str = "duration : {duration}",
    **logger_kwargs,
):
    """Log duration of a with-scoped block of instructions.

    Parameters
    ----------
    level : str, optional
        Logger level used to log the duration, by default "DEBUG"
    message_template : str, optional
        Message template, by default "duration : {duration}". Should include the substring "{duration}" to indicate where the duration will be expanded.
    logger_kwargs : dict, optional
        Additional arguments to pass to the logger.

    Examples
    --------
    ```python
    import time

    with timeit("INFO", "the instructions below take {duration} amount of time"):
        print("sleep start")
        time.sleep(1)
        print("sleep end")
    ```
    """
    return _TimeIt(level, message_template, **logger_kwargs)


class _TimeIt:
    def __init__(self, level, message_template, **logger_kwargs) -> None:
        self.level = level
        self.message_template = message_template
        self.logger_kwargs = logger_kwargs
        self._duration = None

    def __enter__(self):
        self.start = dt.datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self._duration = dt.datetime.now() - self.start

        logger.opt(depth=1).log(
            self.level,
            self.message_template.format(duration=str(self._duration)),
            **self.logger_kwargs,
        )

    @property
    def duration(self):
        return self._duration
