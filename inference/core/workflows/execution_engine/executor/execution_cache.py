from typing import Any, Dict, List

from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.errors import (
    ExecutionEngineRuntimeError,
    InvalidBlockBehaviourError,
)
from inference.core.workflows.execution_engine.compiler.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_step_output_selector,
)


class StepCache:

    @classmethod
    def init(
        cls,
        step_name: str,
        output_definitions: List[OutputDefinition],
    ) -> "StepCache":
        cache_content = {output.name: [] for output in output_definitions}
        return cls(step_name=step_name, cache_content=cache_content)

    def __init__(
        self,
        step_name: str,
        cache_content: Dict[str, List[Any]],
    ):
        self._step_name = step_name
        self._cache_content = cache_content

    def register_outputs(
        self,
        outputs: List[Dict[str, Any]],
    ) -> None:
        for output in outputs:
            for key in output:
                if key not in self._cache_content:
                    # TODO: remove once parent_coordinates outputs
                    # are handled differently
                    self._cache_content[key] = []
                self._cache_content[key].append(output[key])

    def get_outputs(
        self,
        property_name: str,
    ) -> List[Any]:
        return self._cache_content[property_name]

    def get_all_outputs(self) -> List[Dict[str, Any]]:
        all_keys = list(self._cache_content.keys())
        values = list(self._cache_content.values())
        result = []
        for all_keys_values_pack in zip(*values):
            result.append(dict(zip(all_keys, all_keys_values_pack)))
        return result

    def is_property_defined(self, property_name: str) -> bool:
        return property_name in self._cache_content


class ExecutionCache:

    @classmethod
    def init(cls) -> "ExecutionCache":
        return cls(cache_content={}, batches_compatibility={})

    def __init__(
        self,
        cache_content: Dict[str, StepCache],
        batches_compatibility: Dict[str, bool],
    ):
        self._cache_content = cache_content
        self._batches_compatibility = batches_compatibility

    def register_step(
        self,
        step_name: str,
        output_definitions: List[OutputDefinition],
        compatible_with_batches: bool,
    ) -> None:
        if self.contains_step(step_name=step_name):
            return None
        step_cache = StepCache.init(
            step_name=step_name,
            output_definitions=output_definitions,
        )
        self._cache_content[step_name] = step_cache
        self._batches_compatibility[step_name] = compatible_with_batches

    def register_step_outputs(
        self, step_name: str, outputs: List[Dict[str, Any]]
    ) -> None:
        if not self.contains_step(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to register outputs for "
                f"step {step_name} which was not previously registered in cache. "
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        try:
            self._cache_content[step_name].register_outputs(outputs=outputs)
        except TypeError as e:
            # checking this case defensively as there is no guarantee on block
            # meeting contract and we want graceful error handling
            raise InvalidBlockBehaviourError(
                public_message=f"Block implementing step {step_name} should return outputs which are lists of "
                f"dicts, but the type of output does not much expectation.",
                context="workflow_execution | step_output_registration",
                inner_error=e,
            ) from e

    def get_output(self, selector: str) -> List[Any]:
        if not self.is_value_registered(selector=selector):
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
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].get_outputs(property_name=property_name)

    def get_all_step_outputs(self, step_name: str) -> List[Dict[str, Any]]:
        if not self.contains_step(step_name=step_name):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get all outputs from step {step_name} "
                f"which is not register in cache. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        return self._cache_content[step_name].get_all_outputs()

    def output_represent_batch(self, selector: str) -> bool:
        if not self.is_value_registered(selector=selector):
            raise ExecutionEngineRuntimeError(
                public_message=f"Error in execution engine. Attempted to get batches compatibility status "
                f"from step {selector} which is not register in cache. That behavior should be prevented by "
                f"workflows compiler, so this error should be treated as a bug."
                f"Contact Roboflow team through github issues "
                f"(https://github.com/roboflow/inference/issues) providing full context of"
                f"the problem - including workflow definition you use.",
                context="workflow_execution | step_output_registration",
            )
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        return self._batches_compatibility[step_name]

    def is_value_registered(self, selector: Any) -> bool:
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

    def contains_step(self, step_name: str) -> bool:
        return step_name in self._cache_content
