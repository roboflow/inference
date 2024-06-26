import asyncio
from asyncio import AbstractEventLoop
from typing import Any, Dict, List, Optional

from inference.core.workflows.execution_engine.compiler.core import compile_workflow
from inference.core.workflows.execution_engine.compiler.entities import CompiledWorkflow
from inference.core.workflows.execution_engine.executor.core import run_workflow
from inference.core.workflows.execution_engine.executor.runtime_input_assembler import (
    assembly_runtime_parameters,
)
from inference.core.workflows.execution_engine.executor.runtime_input_validator import (
    validate_runtime_input,
)


class ExecutionEngine:

    @classmethod
    def init(
        cls,
        workflow_definition: dict,
        init_parameters: Optional[Dict[str, Any]] = None,
        max_concurrent_steps: int = 1,
        prevent_local_images_loading: bool = False,
    ) -> "ExecutionEngine":
        if init_parameters is None:
            init_parameters = {}
        compiled_workflow = compile_workflow(
            workflow_definition=workflow_definition,
            init_parameters=init_parameters,
        )
        return cls(
            compiled_workflow=compiled_workflow,
            max_concurrent_steps=max_concurrent_steps,
            prevent_local_images_loading=prevent_local_images_loading,
        )

    def __init__(
        self,
        compiled_workflow: CompiledWorkflow,
        max_concurrent_steps: int,
        prevent_local_images_loading: bool,
    ):
        self._compiled_workflow = compiled_workflow
        self._max_concurrent_steps = max_concurrent_steps
        self._prevent_local_images_loading = prevent_local_images_loading

    def run(
        self,
        runtime_parameters: Dict[str, Any],
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> List[Dict[str, Any]]:
        if event_loop is None:
            try:
                event_loop = asyncio.get_event_loop()
            except:
                event_loop = asyncio.new_event_loop()
        return event_loop.run_until_complete(
            self.run_async(runtime_parameters=runtime_parameters)
        )

    async def run_async(
        self, runtime_parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        runtime_parameters = assembly_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=self._compiled_workflow.workflow_definition.inputs,
            prevent_local_images_loading=self._prevent_local_images_loading,
        )
        validate_runtime_input(
            runtime_parameters=runtime_parameters,
            input_substitutions=self._compiled_workflow.input_substitutions,
        )
        return await run_workflow(
            workflow=self._compiled_workflow,
            runtime_parameters=runtime_parameters,
            max_concurrent_steps=self._max_concurrent_steps,
        )
