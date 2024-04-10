from enum import Enum
from typing import Any, Dict, List, Union

from inference.enterprise.workflows.complier.utils import (
    get_last_chunk_of_selector,
    get_step_selector_from_its_output,
    is_step_output_selector,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition


class OutputPlaceholder(Enum):
    EMPTY = "empty"


class StepCache:

    @classmethod
    def init(
        cls, step_name: str, output_definitions: List[OutputDefinition]
    ) -> "StepCache":
        cache_content = {output.name: [] for output in output_definitions}
        return cls(step_name=step_name, cache_content=cache_content)

    def __init__(
        self,
        step_name: str,
        cache_content: Dict[str, Union[List[Any], OutputPlaceholder]],
    ):
        self._step_name = step_name
        self._cache_content = cache_content

    def register_outputs(
        self,
        outputs: List[Dict[str, Any]],
    ) -> None:
        try:
            for output in outputs:
                for key in self._cache_content:
                    self._cache_content[key].append(output[key])
        except ValueError as e:
            raise e  # TODO: error handling
        except KeyError as e:
            raise e  # TODO: error handling

    def register_empty_output(self) -> None:
        for key in self._cache_content:
            self._cache_content[key] = OutputPlaceholder.EMPTY

    def get_outputs(
        self,
        property_name: str,
    ) -> Union[List[Any], OutputPlaceholder]:
        if property_name not in self._step_name:
            raise KeyError(f"{property_name} - TODO: error handling")
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
        return cls(cache_content={})

    def __init__(self, cache_content: Dict[str, StepCache]):
        self._cache_content = cache_content

    def register_step(
        self, step_name: str, output_definitions: List[OutputDefinition]
    ) -> None:
        if self.contains_step(step_name=step_name):
            return None
        step_cache = StepCache.init(
            step_name=step_name,
            output_definitions=output_definitions,
        )
        self._cache_content[step_name] = step_cache

    def register_step_outputs(
        self, step_name: str, outputs: List[Dict[str, Any]]
    ) -> None:
        if not self.contains_step(step_name=step_name):
            raise RuntimeError("TODO: error handling")
        self._cache_content[step_name].register_outputs(outputs=outputs)

    def register_empty_outputs(self, step_name: str) -> None:
        if not self.contains_step(step_name=step_name):
            raise RuntimeError("TODO: error handling")
        self._cache_content[step_name].register_empty_output()

    def is_output_empty(self, selector: str) -> bool:
        if not self.contains_value(selector_or_value=selector):
            raise RuntimeError("TODO: error handling")
        output = self.get_output(selector=selector)
        return output is OutputPlaceholder.EMPTY

    def get_output(self, selector: str) -> Union[List[Any], OutputPlaceholder]:
        if not self.contains_value(selector_or_value=selector):
            raise RuntimeError("TODO: error handling")
        step_selector = get_step_selector_from_its_output(step_output_selector=selector)
        step_name = get_last_chunk_of_selector(selector=step_selector)
        property_name = get_last_chunk_of_selector(selector=selector)
        return self._cache_content[step_name].get_outputs(property_name=property_name)

    def get_all_step_outputs(self, step_name: str) -> List[Dict[str, Any]]:
        if not self.contains_step(step_name=step_name):
            raise RuntimeError("TODO: error handling")
        return self._cache_content[step_name].get_all_outputs()

    def contains_value(self, selector_or_value: Any) -> bool:
        if not is_step_output_selector(selector_or_value=selector_or_value):
            return False
        step_selector = get_step_selector_from_its_output(
            step_output_selector=selector_or_value
        )
        step_name = get_last_chunk_of_selector(selector=step_selector)
        if not self.contains_step(step_name=step_name):
            return False
        property_name = get_last_chunk_of_selector(selector=selector_or_value)
        return self._cache_content[step_name].is_property_defined(
            property_name=property_name
        )

    def contains_step(self, step_name: str) -> bool:
        return step_name in self._cache_content
