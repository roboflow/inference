"""V2 blocks used by the capability examples beyond the 45 parity cases.

Each class shows one declaration feature an author can change and observe:
list-of-batch and mixed-dict parameters, compound groups, per-output layouts,
configured output names, input-free sources, vectorized and stateful calls,
kind codecs, in-place mutation and resource precedence.
"""

from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import Field
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    Select,
    StepRef,
    Stop,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Kind,
)
from roboflow_workflows.execution_engine.v2.resources import Factory

CAPABILITY_NAMESPACE = "capability"


def _parse_celsius(value: Any) -> Any:
    if isinstance(value, str) and value.endswith("C"):
        return float(value[:-1])

    return value


CELSIUS_KIND = Kind(
    name="celsius",
    description="Temperature in degrees Celsius; accepts '21.5C' at the input.",
    validate=lambda payload: isinstance(payload, (int, float)),
    deserialize=_parse_celsius,
    serialize=lambda payload: f"{payload}C",
)
FAHRENHEIT_KIND = Kind(
    name="fahrenheit",
    description="Temperature in degrees Fahrenheit; serialized as '70.7F'.",
    validate=lambda payload: isinstance(payload, (int, float)),
    serialize=lambda payload: f"{round(payload, 1)}F",
)


class Vote(Block):
    """Compare the same position of several batches: one call, ``list[Batch]``."""

    type = "capability/vote@v1"
    outputs = {"votes": Output(), "unanimous": Output()}

    class Params(BlockParams):
        predictions: List[Ref(batch="always")] = Field(
            min_length=1, description="Batches to compare position by position."
        )

    def run(self, *, predictions: List[Batch]) -> list:
        results = []
        for values in zip(*predictions):
            results.append({"votes": list(values), "unanimous": len(set(values)) == 1})

        return results


class CsvRow(Block):
    """Format one CSV line; literal columns stay scalar, varying columns batch."""

    type = "capability/csv_row@v1"
    outputs = {"csv": Output(STRING_KIND)}

    class Params(BlockParams):
        columns: Dict[str, Union[str, float, bool, None, Ref(batch="if_varying")]] = (
            Field(description="Column name to literal or selected value.")
        )

    def run(self, *, columns: Dict[str, Any]) -> Union[dict, list]:
        varying = [value for value in columns.values() if isinstance(value, Batch)]
        if not varying:
            return {"csv": ",".join(str(value) for value in columns.values())}

        lines = []
        for position in range(len(varying[0])):
            cells = [
                value[position] if isinstance(value, Batch) else value
                for value in columns.values()
            ]
            lines.append({"csv": ",".join(str(cell) for cell in cells)})

        return lines


class NamedGroups(Block):
    """Compound group: one child group per name, beside a parent value."""

    type = "capability/named_groups@v1"
    outputs = {"summary": Output(source="parent")}

    class Params(BlockParams):
        parent: Ref() = Field(description="Parent value the groups belong to.")
        groups: Dict[str, Group()] = Field(description="Named child groups.")

    def run(self, *, parent: Any, groups: Dict[str, Batch]) -> dict:
        summary = {
            "parent": parent,
            "groups": {
                name: {
                    "values": list(group),
                    "indices": [list(i) for i in group.indices],
                }
                for name, group in groups.items()
            },
        }

        return {"summary": summary}


class TotalAndShift(Block):
    """Two layouts from one call: a parent total and one shifted value per child."""

    type = "capability/total_and_shift@v1"
    outputs = {
        "total": Output(FLOAT_KIND, source="parent"),
        "shifted": Output(FLOAT_KIND, preserve="children"),
    }

    class Params(BlockParams):
        parent: Ref(FLOAT_KIND) = Field(description="Parent number.")
        children: Group(FLOAT_KIND) = Field(description="Children of the parent.")

    def run(self, *, parent: float, children: Batch) -> dict:
        shifted = Batch.of([child + parent for child in children])

        return {"total": parent + sum(children), "shifted": shifted}


class SplitFields(Block):
    """Split text on commas into outputs named by the ``names`` parameter."""

    type = "capability/split_fields@v1"
    output_fields = ("names",)

    class Params(BlockParams):
        text: Ref(STRING_KIND) | str = Field(description="Comma-separated text.")
        names: List[str] = Field(description="One output name per field.")

    @classmethod
    def describe_outputs(cls, params: BlockParams) -> Mapping[str, Output]:
        return {name: Output(STRING_KIND) for name in params.names}

    def run(self, *, text: str, names: List[str]) -> dict:
        return dict(zip(names, text.split(",")))


class Generate(Block):
    """Input-free source: creates a new top-level group from its settings."""

    type = "capability/generate@v1"
    outputs = {"value": Output(INTEGER_KIND, expand="generated")}

    class Params(BlockParams):
        values: List[int] = Field(description="Values to emit, in order.")

    def run(self, *, values: List[int]) -> dict:
        return {"value": Batch.of(values)}


class Pair(Block):
    """Combine two item values into a list."""

    type = "capability/pair@v1"
    outputs = {"pair": Output()}

    class Params(BlockParams):
        left: Ref() = Field(description="First value.")
        right: Ref() = Field(description="Second value.")

    def run(self, *, left: Any, right: Any) -> dict:
        return {"pair": [left, right]}


class Total(Block):
    """Sum a group of numbers."""

    type = "capability/total@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        values: Group(INTEGER_KIND, FLOAT_KIND) = Field(description="Numbers to sum.")

    def run(self, *, values: Batch) -> dict:
        return {"total": sum(values)}


class CountingBatchConsumer(Block):
    """Vectorized consumer that numbers its own calls."""

    type = "capability/counting_batch_consumer@v1"
    outputs = {"result": Output()}

    class Params(BlockParams):
        item: Ref(batch="always") = Field(description="Values of one call.")

    def __init__(self):
        self.calls = 0

    def run(self, *, item: Batch) -> list:
        self.calls += 1

        return [{"result": {"item": value, "call": self.calls}} for value in item]


class CountingGroupConsumer(Block):
    """Vectorized consumer of parents and their child groups."""

    type = "capability/counting_group_consumer@v1"
    outputs = {"result": Output()}

    class Params(BlockParams):
        parent: Ref(batch="always") = Field(description="Parent values.")
        value: Group(batch="always") = Field(description="Children per parent.")

    def __init__(self):
        self.calls = 0

    def run(self, *, parent: Batch, value: Batch) -> list:
        self.calls += 1
        results = [
            {"result": {"parent": p, "children": list(group), "call": self.calls}}
            for p, group in zip(parent, value)
        ]

        return results


class EmptyAcceptingGroupConsumer(CountingGroupConsumer):
    """The same consumer, also called for parents whose group is unavailable."""

    type = "capability/empty_accepting_group_consumer@v1"
    accepts_empty = True


class EvenGate(Block):
    """Vectorized control: admits even values, one decision per element."""

    type = "capability/even_gate@v1"

    class Params(BlockParams):
        value: Ref(INTEGER_KIND, batch="always") = Field(description="Values.")
        next_steps: List[StepRef] = Field(description="Governed steps.")

    def run(self, *, value: Batch, next_steps: List[str]) -> list:
        return [Select(next_steps) if item % 2 == 0 else Stop() for item in value]


class ToFahrenheit(Block):
    """Convert Celsius to Fahrenheit."""

    type = "capability/to_fahrenheit@v1"
    outputs = {"fahrenheit": Output(FAHRENHEIT_KIND)}

    class Params(BlockParams):
        celsius: Ref(CELSIUS_KIND) = Field(description="Temperature in Celsius.")

    def run(self, *, celsius: float) -> dict:
        return {"fahrenheit": celsius * 9 / 5 + 32}


class AppendTag(Block):
    """Append a tag to the selected list, in place."""

    type = "capability/append_tag@v1"
    outputs = {"tags": Output(LIST_OF_VALUES_KIND)}
    mutates = ("tags",)

    class Params(BlockParams):
        tags: Ref(LIST_OF_VALUES_KIND) = Field(description="List to extend.")
        tag: str = Field(description="Tag to append.")

    def run(self, *, tags: List[str], tag: str) -> dict:
        tags.append(tag)

        return {"tags": tags}


class CountTags(Block):
    """Count the tags of a list when this step runs."""

    type = "capability/count_tags@v1"
    outputs = {"count": Output(INTEGER_KIND)}

    class Params(BlockParams):
        tags: Ref(LIST_OF_VALUES_KIND) = Field(description="List to count.")

    def run(self, *, tags: List[str]) -> dict:
        return {"count": len(tags)}


class Greeting(Block):
    """Report which ``greeting`` resource the session injected."""

    type = "capability/greeting@v1"
    outputs = {"text": Output(STRING_KIND)}

    def __init__(self, *, greeting: str, suffix: Optional[str] = "!"):
        self.greeting = greeting
        self.suffix = suffix

    def run(self) -> dict:
        return {"text": f"{self.greeting}{self.suffix or ''}"}


CAPABILITY_BLOCKS = (
    Vote,
    CsvRow,
    NamedGroups,
    TotalAndShift,
    SplitFields,
    Generate,
    Pair,
    Total,
    CountingBatchConsumer,
    CountingGroupConsumer,
    EmptyAcceptingGroupConsumer,
    EvenGate,
    ToFahrenheit,
    AppendTag,
    CountTags,
    Greeting,
)


def create_capability_catalogue() -> Catalogue:
    """Collect the capability blocks, their kinds and a ``greeting`` provider.

    Returns:
        Catalogue in namespace ``capability``.
    """
    catalogue = Catalogue(
        CAPABILITY_BLOCKS,
        kinds=[CELSIUS_KIND, FAHRENHEIT_KIND],
        namespace=CAPABILITY_NAMESPACE,
        providers={"greeting": Factory(lambda: "hello from the catalogue")},
    )

    return catalogue
