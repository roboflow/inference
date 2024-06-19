from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple, Union

from networkx import DiGraph

from inference.core.workflows.constants import NODE_COMPILATION_OUTPUT_PROPERTY
from inference.core.workflows.execution_engine.compiler.entities import (
    CompoundStepInputDefinition,
    DynamicStepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_input_selector,
    is_step_output_selector,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.branching_manager import (
    BranchingManager,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchesManager,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.execution_cache import (
    ExecutionCache,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.step_input_assembler import (
    BatchModeSIMDStepInput,
    NonBatchModeSIMDStepInput,
)


class ExecutionDataManager:

    @classmethod
    def init(
        cls,
        execution_graph: DiGraph,
        runtime_parameters: Dict[str, Any],
    ) -> "ExecutionDataManager":
        execution_cache = ExecutionCache.init(execution_graph=execution_graph)
        dynamic_batches_manager = DynamicBatchesManager.init(
            execution_graph=execution_graph,
            runtime_parameters=runtime_parameters,
        )
        branching_manager = BranchingManager.init()
        return cls(
            execution_graph=execution_graph,
            runtime_parameters=runtime_parameters,
            execution_cache=execution_cache,
            dynamic_batches_manager=dynamic_batches_manager,
            branching_manager=branching_manager,
        )

    def __init__(
        self,
        execution_graph: DiGraph,
        runtime_parameters: Dict[str, Any],
        execution_cache: ExecutionCache,
        dynamic_batches_manager: DynamicBatchesManager,
        branching_manager: BranchingManager,
    ):
        self._execution_graph = execution_graph
        self._runtime_parameters = runtime_parameters
        self._execution_cache = execution_cache
        self._dynamic_batches_manager = dynamic_batches_manager
        self._branching_manager = branching_manager

    def all_inputs_impacting_step_are_registered(self, step_selector: str) -> bool:
        step_node_data: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        all_batch_oriented_parameters_registered = (
            are_all_batch_oriented_parameters_registered(
                step_node_data=step_node_data,
                execution_cache=self._execution_cache,
            )
        )
        all_execution_branches_registered = all(
            self._branching_manager.is_execution_branch_registered(
                execution_branch=execution_branch
            )
            for execution_branch in step_node_data.execution_branches_impacting_inputs
        )
        return (
            all_batch_oriented_parameters_registered
            and all_execution_branches_registered
        )

    def get_non_simd_step_input(self, step_selector: str) -> Dict[str, Any]:
        if self.is_step_simd(step_selector=step_selector):
            raise ValueError(f"SIMD step {step_selector} requested non-simd input")
        return {}

    def register_non_simd_step_output(
        self, step_selector: str, output: Dict[str, Any]
    ) -> None:
        if self.is_step_simd(step_selector=step_selector):
            raise ValueError(f"SIMD step {step_selector} registering non-simd output")

    def get_non_simd_step_output(
        self, output_selector: str
    ) -> Union[Any, Dict[str, Any]]:
        pass

    def get_simd_step_input(self, step_selector: str) -> BatchModeSIMDStepInput:
        if not self.is_step_simd(step_selector=step_selector):
            raise ValueError()
        return {}

    def iterate_over_simd_step_input(
        self, step_selector: str
    ) -> Generator[NonBatchModeSIMDStepInput, None, None]:
        if not self.is_step_simd(step_selector=step_selector):
            raise ValueError()
        yield {}

    def register_simd_step_output(
        self, step_selector: str, output: List[Dict[str, Any]]
    ) -> None:
        if self.is_step_simd(step_selector=step_selector):
            raise ValueError()

    def get_simd_step_output(self, output_selector: str) -> list:
        pass

    def get_all_simd_step_outputs(self, step_selector: str) -> list:
        pass

    def is_step_simd(self, step_selector: str) -> bool:
        step_node_data: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        return step_node_data.is_batch_oriented()


def are_all_batch_oriented_parameters_registered(
    step_node_data: StepNode,
    execution_cache: ExecutionCache,
) -> bool:
    for batch_oriented_parameter in step_node_data.batch_oriented_parameters:
        parameter = step_node_data.input_data[batch_oriented_parameter]
        if not parameter.is_compound_input():
            if not is_dynamic_step_input_registered(
                dynamic_input=parameter,
                execution_cache=execution_cache,
            ):
                return False
        else:
            if not are_all_compound_inputs_registered(
                compound_input=parameter,
                execution_cache=execution_cache,
            ):
                return False
    return True


def are_all_compound_inputs_registered(
    compound_input: CompoundStepInputDefinition,
    execution_cache: ExecutionCache,
) -> bool:
    for step_input in compound_input.iterate_through_definitions():
        if not step_input.is_batch_oriented():
            continue
        if not is_dynamic_step_input_registered(
            dynamic_input=step_input,  # type: ignore
            execution_cache=execution_cache,
        ):
            return False
    return True


def is_dynamic_step_input_registered(
    dynamic_input: DynamicStepInputDefinition,
    execution_cache: ExecutionCache,
) -> bool:
    selector = dynamic_input.selector
    if is_input_selector(selector_or_value=selector):
        return True
    if is_step_output_selector(selector_or_value=selector):
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        return execution_cache.is_step_output_registered(step_name=step_name)
    raise ValueError("Should not be possible")
