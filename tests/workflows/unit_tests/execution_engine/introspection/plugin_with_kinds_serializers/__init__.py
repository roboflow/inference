from typing import List, Type

from inference.core.workflows.execution_engine.entities.types import Kind
from inference.core.workflows.prototypes.block import WorkflowBlock

MY_KIND_1 = Kind(name="1")
MY_KIND_2 = Kind(name="2")
MY_KIND_3 = Kind(name="3")


def load_blocks() -> List[Type[WorkflowBlock]]:
    return []


def load_kinds() -> List[Kind]:
    return [
        MY_KIND_1,
        MY_KIND_2,
        MY_KIND_3,
    ]


KINDS_SERIALIZERS = {
    "1": lambda value: "1",
    "2": lambda value: "2",
    "3": lambda value: "3",
}

KINDS_DESERIALIZERS = {
    "1": lambda name, value: "1",
    "2": lambda name, value: "2",
    "3": lambda name, value: "3",
}
