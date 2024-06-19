from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from networkx import DiGraph

from inference.core.workflows.constants import NODE_COMPILATION_OUTPUT_PROPERTY
from inference.core.workflows.entities.types import FlowControl
from inference.core.workflows.execution_engine.compiler.entities import (
    CompoundStepInputDefinition,
    DynamicStepInputDefinition,
    InputNode,
    StepNode,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_input_selector,
    is_selector,
    is_step_output_selector,
    is_step_selector,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.branching_manager import (
    BranchingManager,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchesManager,
    DynamicBatchIndex,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.execution_cache import (
    ExecutionCache,
)
from inference.core.workflows.execution_engine.executor.execution_data_manager.step_input_assembler import (
    BatchModeSIMDStepInput,
    NonBatchModeSIMDStepInput,
    construct_non_simd_step_input,
    construct_simd_step_input,
    iterate_over_simd_step_input,
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

    def get_non_simd_step_input(self, step_selector: str) -> Optional[Dict[str, Any]]:
        if self.is_step_simd(step_selector=step_selector):
            raise ValueError(f"SIMD step {step_selector} requested non-simd input")
        step_node: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        return construct_non_simd_step_input(
            step_node=step_node,
            runtime_parameters=self._runtime_parameters,
            execution_cache=self._execution_cache,
            branching_manager=self._branching_manager,
        )

    def register_non_simd_step_output(
        self, step_selector: str, output: Union[Dict[str, Any], FlowControl]
    ) -> None:
        if self.is_step_simd(step_selector=step_selector):
            raise ValueError(f"SIMD step {step_selector} registering non-simd output")
        step_name = get_last_chunk_of_selector(selector=step_selector)
        step_node: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        if isinstance(output, FlowControl):
            self._register_flow_control_output_for_non_simd_step(
                step_node=step_node,
                output=output,
            )
            return None
        self._execution_cache.register_non_batch_step_outputs(
            step_name=step_name,
            outputs=output,
        )

    def get_simd_step_input(self, step_selector: str) -> BatchModeSIMDStepInput:
        if not self.is_step_simd(step_selector=step_selector):
            raise ValueError()
        step_node: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        return construct_simd_step_input(
            step_node=step_node,
            runtime_parameters=self._runtime_parameters,
            execution_cache=self._execution_cache,
            dynamic_batches_manager=self._dynamic_batches_manager,
            branching_manager=self._branching_manager,
        )

    def iterate_over_simd_step_input(
        self, step_selector: str
    ) -> Generator[NonBatchModeSIMDStepInput, None, None]:
        if not self.is_step_simd(step_selector=step_selector):
            raise ValueError()
        step_node: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        yield from iterate_over_simd_step_input(
            step_node=step_node,
            runtime_parameters=self._runtime_parameters,
            execution_cache=self._execution_cache,
            dynamic_batches_manager=self._dynamic_batches_manager,
            branching_manager=self._branching_manager,
        )

    def register_simd_step_output(
        self,
        step_selector: str,
        indices: List[DynamicBatchIndex],
        outputs: List[
            Union[List[Dict[str, Any]], Dict[str, Any], List[FlowControl], FlowControl]
        ],
    ) -> None:
        if not self.is_step_simd(step_selector=step_selector):
            raise ValueError()
        step_node: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        if (
            step_node.output_dimensionality - step_node.step_execution_dimensionality
        ) > 0:
            # increase in dimensionality
            indices, outputs = flatten_nested_output(indices=indices, outputs=outputs)
            self._dynamic_batches_manager.register_element_indices_for_lineage(
                lineage=step_node.data_lineage,
                indices=indices,
            )
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if step_node.child_execution_branches:
            if not all(isinstance(element, FlowControl) for element in outputs):
                raise ValueError(
                    f"Flow control step {step_name} expected to only produce FlowControl objects"
                )
            self._register_flow_control_output_for_simd_step(
                step_node=step_node,
                indices=indices,
                outputs=outputs,
            )
            return None
        self._execution_cache.register_batch_of_step_outputs(
            step_name=step_name,
            indices=indices,
            outputs=outputs,
        )

    def get_selector_indices(self, selector: str) -> Optional[List[DynamicBatchIndex]]:
        print("get_selector_indices()", selector)
        selector_lineage = []
        if not is_selector(selector_or_value=selector):
            raise ValueError(f"Not a valid selector: {selector}")
        potential_step_selector = get_step_selector_from_its_output(
            step_output_selector=selector
        )
        if is_input_selector(selector_or_value=selector):
            if self.does_input_represent_batch(input_selector=selector):
                input_node: InputNode = self._execution_graph.nodes[selector][
                    NODE_COMPILATION_OUTPUT_PROPERTY
                ]
                selector_lineage = input_node.data_lineage
        elif is_step_selector(selector_or_value=potential_step_selector):
            if self.is_step_simd(step_selector=potential_step_selector):
                step_node_data: StepNode = self._execution_graph.nodes[
                    potential_step_selector
                ][NODE_COMPILATION_OUTPUT_PROPERTY]
                selector_lineage = step_node_data.data_lineage
        else:
            raise ValueError(f"Unknown selector: {selector}")
        if not selector_lineage:
            return None
        if not self._dynamic_batches_manager.is_lineage_registered(
            lineage=selector_lineage
        ):
            return []
        return self._dynamic_batches_manager.get_indices_for_data_lineage(
            lineage=selector_lineage
        )

    def get_non_batch_data(self, selector: str) -> Any:
        if not is_selector(selector_or_value=selector):
            raise ValueError(f"Not a valid selector: {selector}")
        potential_step_selector = get_step_selector_from_its_output(
            step_output_selector=selector
        )
        if is_input_selector(
            selector_or_value=selector
        ) and not self.does_input_represent_batch(input_selector=selector):
            input_name = get_last_chunk_of_selector(selector=selector)
            return self._runtime_parameters[input_name]
        elif is_step_selector(
            selector_or_value=potential_step_selector
        ) and not self.is_step_simd(step_selector=potential_step_selector):
            step_name = get_last_chunk_of_selector(selector=potential_step_selector)
            if selector.endswith(".*"):
                return self._execution_cache.get_all_non_batch_step_outputs(
                    step_name=step_name,
                )
            return self._execution_cache.get_non_batch_output(selector=selector)
        else:
            raise ValueError(f"Invalid selector {selector}!")

    def get_batch_data(
        self, selector: str, indices: List[DynamicBatchIndex]
    ) -> List[Any]:
        if not is_selector(selector_or_value=selector):
            raise ValueError(f"Not a valid selector: {selector}")
        potential_step_selector = get_step_selector_from_its_output(
            step_output_selector=selector
        )
        if is_input_selector(
            selector_or_value=selector
        ) and self.does_input_represent_batch(input_selector=selector):
            input_name = get_last_chunk_of_selector(selector=selector)
            input_data = self._runtime_parameters[input_name]
            # simplification: assumption that we can only request dim-1 batch from inputs
            requested_indices = {i[0] for i in indices}
            return [
                data_element if idx in requested_indices else None
                for idx, data_element in enumerate(input_data)
            ]
        elif is_step_selector(
            selector_or_value=potential_step_selector
        ) and self.is_step_simd(step_selector=potential_step_selector):
            step_name = get_last_chunk_of_selector(selector=potential_step_selector)
            if selector.endswith(".*"):
                return self._execution_cache.get_all_batch_step_outputs(
                    step_name=step_name,
                    batch_elements_indices=indices,
                )
            return self._execution_cache.get_batch_output(
                selector=selector,
                batch_elements_indices=indices,
            )
        else:
            raise ValueError(f"Invalid selector {selector}!")

    def is_step_simd(self, step_selector: str) -> bool:
        step_node_data: StepNode = self._execution_graph.nodes[step_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        return step_node_data.is_batch_oriented()

    def does_input_represent_batch(self, input_selector: str) -> bool:
        input_node: InputNode = self._execution_graph.nodes[input_selector][
            NODE_COMPILATION_OUTPUT_PROPERTY
        ]
        return input_node.is_batch_oriented()

    def _register_flow_control_output_for_non_simd_step(
        self,
        step_node: StepNode,
        output: FlowControl,
    ) -> None:
        if not step_node.child_execution_branches:
            raise ValueError(
                "This step is not flow-control, so cannot return FlowControl object"
            )
        if not output.context:
            raise ValueError("Step must decode on flow control!")
        selected_steps = output.context
        if not isinstance(selected_steps, list):
            selected_steps = {selected_steps}
        else:
            selected_steps = set(selected_steps)
        selected_execution_branches = set()
        for target_step, branch_name in step_node.child_execution_branches.items():
            if target_step in selected_steps:
                selected_execution_branches.add(branch_name)
        for branch_name in step_node.child_execution_branches.values():
            mask = branch_name in selected_execution_branches
            print(f"NON-SIMD flow control -> {branch_name}: {mask}")
            self._branching_manager.register_non_batch_mask(
                execution_branch=branch_name,
                mask=mask,
            )
        return None

    def _register_flow_control_output_for_simd_step(
        self,
        step_node: StepNode,
        indices: List[DynamicBatchIndex],
        outputs: List[FlowControl],
    ) -> None:
        all_branches_masks = {
            branch_name: set()
            for branch_name in step_node.child_execution_branches.values()
        }
        for output_index, output in zip(indices, outputs):
            selected_steps = output.context
            if selected_steps is None:
                selected_steps = []
            if not isinstance(selected_steps, list):
                selected_steps = {selected_steps}
            else:
                selected_steps = set(selected_steps)
            for selected_step in selected_steps:
                branch_for_step = step_node.child_execution_branches.get(selected_step)
                print("selected_step", selected_step, "branch", branch_for_step)
                if branch_for_step is None:
                    raise ValueError(
                        f"Cannot find execution branch for step {selected_step}"
                    )
                all_branches_masks[branch_for_step].add(output_index)
        for branch_name, mask in all_branches_masks.items():
            print(f"SIMD flow control -> {branch_name}: {mask}")
            self._branching_manager.register_batch_oriented_mask(
                execution_branch=branch_name,
                mask=mask,
            )


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


def flatten_nested_output(
    indices: List[DynamicBatchIndex],
    outputs: List[
        Union[List[Dict[str, Any]], Dict[str, Any], List[FlowControl], FlowControl]
    ],
) -> Tuple[List[DynamicBatchIndex], List[Union[Dict[str, Any], FlowControl]]]:
    flattened_index, flattened_output = [], []
    for index, output_element in zip(indices, outputs):
        if not isinstance(output_element, list):
            raise ValueError("Output missmatch")
        for nested_index, nested_element_of_output_element in enumerate(output_element):
            flattened_index.append(index + (nested_index,))
            flattened_output.append(nested_element_of_output_element)
    return flattened_index, flattened_output
