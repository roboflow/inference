from typing import Any, Dict, List, Optional

from inference.core.workflows.entities.engine import BaseExecutionEngine
from inference.core.workflows.errors import NotSupportedExecutionEngineError
from inference.core.workflows.execution_engine.v1.core import ExecutionEngineV1

REGISTERED_ENGINES = {1: ExecutionEngineV1}


class ExecutionEngine(BaseExecutionEngine):

    @classmethod
    def init(
        cls,
        workflow_definition: dict,
        init_parameters: Optional[Dict[str, Any]] = None,
        max_concurrent_steps: int = 1,
        prevent_local_images_loading: bool = False,
        workflow_id: Optional[str] = None,
    ) -> "ExecutionEngine":
        requested_engine_version = int(
            workflow_definition.get(
                "execution_engine_version",
                max(REGISTERED_ENGINES.keys()),
            )
        )
        if requested_engine_version not in REGISTERED_ENGINES:
            raise NotSupportedExecutionEngineError(
                public_message=f"There is no Execution Engine in major version: {requested_engine_version} defined. "
                f"Available execution errors: {list(REGISTERED_ENGINES.keys())}.",
                context="workflow_compilation | engine_initialisation",
            )
        engine = REGISTERED_ENGINES[requested_engine_version].init(
            workflow_definition=workflow_definition,
            init_parameters=init_parameters,
            max_concurrent_steps=max_concurrent_steps,
            prevent_local_images_loading=prevent_local_images_loading,
            workflow_id=workflow_id,
        )
        return cls(engine=engine)

    def __init__(
        self,
        engine: BaseExecutionEngine,
    ):
        self._engine = engine

    async def run_async(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
    ) -> List[Dict[str, Any]]:
        return await self._engine.run_async(
            runtime_parameters=runtime_parameters,
            fps=fps,
        )
