from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple, Union

from networkx import DiGraph


class ExecutionDataManager:

    @classmethod
    def init(
        cls,
        execution_graph: DiGraph,
        runtime_parameters: Dict[str, Any],
    ) -> "ExecutionDataManager":
        return cls()

    def __init__(self):
        pass

    def should_step_be_run(self, step_selector: str) -> bool:
        return True

    def get_non_simd_step_input(self, step_selector: str) -> Dict[str, Any]:
        if self.is_step_simd(step_selector=step_selector):
            raise ValueError()
        return {}

    def register_non_simd_step_output(
        self, step_selector: str, output: Dict[str, Any]
    ) -> None:
        if self.is_step_simd(step_selector=step_selector):
            raise ValueError()

    def get_non_simd_step_output(
        self, output_selector: str
    ) -> Union[Any, Dict[str, Any]]:
        pass

    def get_simd_step_input(self, step_selector: str) -> Dict[str, Any]:
        if not self.is_step_simd(step_selector=step_selector):
            raise ValueError()
        return {}

    def iterate_over_simd_step_input(
        self, step_selector: str
    ) -> Generator[Dict[str, Any], None, None]:
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
        pass
