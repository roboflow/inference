import asyncio
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop
from typing import Any, Dict, List, Optional


class BaseExecutionEngine(ABC):

    @classmethod
    @abstractmethod
    def init(
        cls,
        workflow_definition: dict,
        init_parameters: Optional[Dict[str, Any]] = None,
        max_concurrent_steps: int = 1,
        prevent_local_images_loading: bool = False,
        workflow_id: Optional[str] = None,
    ) -> "BaseExecutionEngine":
        pass

    def run(
        self,
        runtime_parameters: Dict[str, Any],
        event_loop: Optional[AbstractEventLoop] = None,
        fps: float = 0,
    ) -> List[Dict[str, Any]]:
        if event_loop is None:
            try:
                event_loop = asyncio.get_event_loop()
            except:
                event_loop = asyncio.new_event_loop()
        return event_loop.run_until_complete(
            self.run_async(runtime_parameters=runtime_parameters, fps=fps)
        )

    @abstractmethod
    async def run_async(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
    ) -> List[Dict[str, Any]]:
        pass
