from typing import Any, Callable, Protocol


class BackgroundTaskScheduler(Protocol):
    """Fire-and-forget task scheduler injected into sink blocks.

    Satisfied by ``fastapi.BackgroundTasks``, which the HTTP layer injects.
    Declared here so the block library does not import FastAPI for an
    annotation. ``REGISTERED_INITIALIZERS`` defaults it to ``None`` and every
    sink already branches on that, so no default implementation is needed.
    """

    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...
