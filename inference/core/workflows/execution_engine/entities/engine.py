from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from inference.core.workflows.execution_engine.profiling.core import WorkflowsProfiler


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
        profiler: Optional[WorkflowsProfiler] = None,
    ) -> "BaseExecutionEngine":
        pass

    @abstractmethod
    def run(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
        _is_preview: bool = False,
    ) -> List[Dict[str, Any]]:
        pass
