from collections import defaultdict
from copy import copy
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

from networkx import DiGraph

from inference.core.workflows.errors import (
    ExecutionEngineRuntimeError,
    InvalidBlockBehaviourError,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.v1.compiler.entities import StepNode
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_step_node,
    is_step_output_selector,
    node_as,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchIndex,
)


class ExecutionCache:

    @classmethod
    def init(
        cls,
        execution_graph: DiGraph,
    ) -> "ExecutionCache":
        cache = cls(
            cache_content={}, batches_compatibility={}, step_outputs_registered=set()
        )
        for node in execution_graph.nodes:
            if not is_step_node(execution_graph=execution_graph, node=node):
                continue
            node_data: StepNode = node_as(
                execution_graph=execution_graph,
                node=node,
                expected_type=StepNode,
            )
            step_name = node_data.step_manifest.name
            compatible_with_batches = node_data.is_batch_oriented()
            outputs = node_data.step_manifest.get_actual_outputs()
            cache.declare_step(
                step_name=step_name,
                compatible_with_batches=compatible_with_batches,
                outputs=outputs,
            )
        return cache

    def __init__(
        self,
        cache_content: Dict[str, Union["BatchStepCache", "NonBatchStepCache"]],
        batches_compatibility: Dict[str, bool],
        step_outputs_registered: Set[str],
    ):
        self._cache_content = cache_content
        self._batches_compatibility = batches_compatibility
        self._step_outputs_registered = step_outputs_registered

    def declare_step(
        self,
        step_name: str,
        compatible_with_batches: bool,
        outputs: List[OutputDefinition],
    ) -> None:
        if self.contains_step(step_name=step_name):
            return None
        if compatible_with_batches:
            step_cache = BatchStepCache.init(
                step_name=step_name,
                outputs=outputs,
            )
        else:
            step_cache = NonBatchStepCache.init(
                step_name=step_name,
                outputs=outputs,
            )
        self._cache_content[step_name] = step_cache
        self._batches_compatibility[step_name] = compatible_with_batches

    def register_batch_of_step_outputs(
        self,
        step_name: str,
        indices: List[DynamicBatchIndex],
        outputs: List[Dict[str, Any]],
    ) -> None:
        if not self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to register batch outputs for "
                f"step {step_name} which is not registered as batch-compatible. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        try:
            self._cache_content[step_name].register_outputs(
                indices=indices, outputs=outputs
            )
            self._step_outputs_registered.add(step_name)
        except (TypeError, AttributeError) as e:
            # checking this case defensively as there is no guarantee on block
            # meeting contract, and we want graceful error handling
            raise InvalidBlockBehaviourError(
                public_message=f"Block implementing step `{step_name}` should return outputs which are lists of "
                f"dicts, but the type of output does not match expectation.",
                context="workflow_execution | step_output_registration",
                inner_error=e,
            ) from e

    def register_non_batch_step_outputs(
        self, step_name: str, outputs: Dict[str, Any]
    ) -> None:
        if self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to register non-batch outputs for "
                f"step {step_name} which was registered in cache as batch compatible. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        try:
            self._cache_content[step_name].register_outputs(outputs=outputs)
        except (TypeError, AttributeError) as e:
            # checking this case defensively as there is no guarantee on block
            # meeting contract, and we want graceful error handling
            raise InvalidBlockBehaviourError(
                public_message=f"Block implementing step `{step_name}` `should return outputs which are "
                f"dicts, but the type of output does not match expectation.",
                context="workflow_execution | step_output_registration",
                inner_error=e,
            ) from e
        self._step_outputs_registered.add(step_name)

    def get_batch_output(
        self,
        selector: str,
        batch_elements_indices: List[DynamicBatchIndex],
        mask: Optional[Set[DynamicBatchIndex]] = None,
    ) -> List[Any]:
        if not self.is_step_output_declared(selector=selector):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output which is not registered using "
                f"step {selector}. That behavior should be prevented by workflows compiler, so "
                f"this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if not self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output in batch mode which is "
                f"not supported for step {selector}. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].get_outputs(
            property_name=property_name,
            indices=batch_elements_indices,
            mask=mask,
        )

    def get_non_batch_output(
        self,
        selector: str,
    ) -> Any:
        if not self.is_step_output_declared(selector=selector):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output which is not registered using "
                f"step {selector}. That behavior should be prevented by workflows compiler, so "
                f"this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get output in non-batch mode which is "
                f"not supported for step {selector} registered as batch-compatible. That behavior "
                f"should be prevented by workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].get_outputs(property_name=property_name)

    def get_all_batch_step_outputs(
        self,
        step_name: str,
        batch_elements_indices: List[DynamicBatchIndex],
        mask: Optional[Set[DynamicBatchIndex]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get all outputs from step {step_name} "
                f"which is not register in cache. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return self._cache_content[step_name].get_all_outputs(
            indices=batch_elements_indices,
            mask=mask,
        )

    def get_all_non_batch_step_outputs(
        self,
        step_name: str,
    ) -> Dict[str, Any]:
        if self.step_outputs_batches(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get all non-batch outputs from step {step_name} "
                f"which is registered as batch-compatible. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return self._cache_content[step_name].get_all_outputs()

    def step_outputs_batches(self, step_name: str) -> bool:
        if not self.contains_step(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to check outputs status from step {step_name} "
                f"which is not register in cache. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return self._batches_compatibility[step_name]

    def is_step_output_declared(self, selector: Any) -> bool:
        if not is_step_output_selector(selector_or_value=selector):
            return False
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if not self.contains_step(step_name=step_name):
            return False
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].is_property_defined(
            property_name=property_name
        )

    def is_step_output_data_registered(self, step_name: str) -> bool:
        if not self.contains_step(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to check if step {step_name} "
                f"registered outputs actual, but provided unknown step name. Behavior should "
                f"be prevented by workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return step_name in self._step_outputs_registered

    def contains_step(self, step_name: str) -> bool:
        return step_name in self._cache_content


class BatchStepCache:

    @classmethod
    def init(cls, step_name: str, outputs: List[OutputDefinition]) -> "BatchStepCache":
        return cls(
            step_name=step_name,
            outputs=outputs,
            cache_content=defaultdict(lambda: defaultdict()),
        )

    def __init__(
        self,
        step_name: str,
        outputs: List[OutputDefinition],
        cache_content: DefaultDict[str, DefaultDict[DynamicBatchIndex, Any]],
    ):
        self._step_name = step_name
        self._outputs = {o.name for o in outputs}
        self._cache_content = cache_content

    def register_outputs(
        self,
        indices: List[DynamicBatchIndex],
        outputs: List[Dict[str, Any]],
    ) -> None:
        if len(indices) != len(outputs):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to register batch of step outputs for step:"
                f"{self._step_name} providing not aligned indices (length of indices: {len(indices)}, "
                f" length of data: {len(outputs)}). This is most likely bug. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        for index, element in zip(indices, outputs):
            element_properties = set(element.keys())
            if element_properties != self._outputs:
                raise ExecutionEngineRuntimeError(
                    public_message=f"Step {self._step_name} did not produce required outputs. "
                    f"Expected: {self._outputs}. Got: {element_properties}. "
                    f"Contact Roboflow team through github issues "
                    f"(https://github.com/roboflow/inference/issues) providing full context of"
                    f"the problem - including workflow definition you use.",
                    context="workflow_execution | step_output_registration",
                )
            for property_name, property_value in element.items():
                self._cache_content[property_name][index] = property_value

    def get_outputs(
        self,
        property_name: str,
        indices: List[DynamicBatchIndex],
        mask: Optional[Set[DynamicBatchIndex]] = None,
    ) -> List[Any]:
        return [
            (
                self._cache_content.get(property_name, {}).get(index)
                if mask is None or index in mask
                else None
            )
            for index in indices
        ]

    def get_all_outputs(
        self,
        indices: List[DynamicBatchIndex],
        mask: Optional[Set[DynamicBatchIndex]] = None,
    ) -> List[Dict[str, Any]]:
        all_keys = list(self._cache_content.keys())
        if not all_keys:
            all_keys = self._outputs
        empty_value = {k: None for k in all_keys}
        return [
            (
                {k: self._cache_content.get(k, {}).get(index) for k in all_keys}
                if mask is None or index[: len(mask)] in mask
                else copy(empty_value)
            )
            for index in indices
        ]

    def is_property_defined(self, property_name: str) -> bool:
        return property_name in self._cache_content or property_name in self._outputs


class NonBatchStepCache:

    @classmethod
    def init(
        cls, step_name: str, outputs: List[OutputDefinition]
    ) -> "NonBatchStepCache":
        return cls(
            step_name=step_name,
            outputs=outputs,
            cache_content=dict(),
        )

    def __init__(
        self,
        step_name: str,
        outputs: List[OutputDefinition],
        cache_content: Dict[str, Any],
    ):
        self._step_name = step_name
        self._outputs = {o.name for o in outputs}
        self._cache_content = cache_content

    def register_outputs(self, outputs: Dict[str, Any]):
        if set(outputs.keys()) != self._outputs:
            raise ExecutionEngineRuntimeError(
                public_message=f"Step {self._step_name} did not produce required outputs. "
                f"Expected: {self._outputs}. Got: {outputs.keys()}. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        self._cache_content = outputs

    def get_outputs(
        self,
        property_name: str,
    ) -> Any:
        return self._cache_content.get(property_name)

    def get_all_outputs(self) -> Dict[str, Any]:
        if not self._cache_content:
            return {output: None for output in self._outputs}
        return self._cache_content

    def is_property_defined(self, property_name: str) -> bool:
        return property_name in self._cache_content or property_name in self._outputs
