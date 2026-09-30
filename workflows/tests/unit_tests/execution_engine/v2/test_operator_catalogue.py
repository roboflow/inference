"""The immutable Catalogue's separate operator registry.

Operators are class-owned like blocks and sources: a catalogue only collects
classes, there is no process-wide hidden registry, and a third-party operator
registers and compiles exactly like the built-in ones.
"""

import json
import sys
import types

import pytest
from pydantic import Field
from roboflow_workflows.execution_engine.v2 import (
    Catalogue,
    Source,
    SourceOutput,
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.catalogue import OperatorEntry
from roboflow_workflows.execution_engine.v2.data import Axis, EntryLayout
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    CatalogueError,
    UnknownBlockError,
)
from roboflow_workflows.execution_engine.v2.introspection import describe_catalogue
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.operators import (
    Operator,
    OperatorParams,
    OperatorPort,
)
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window


class PairUp(Operator):
    """Third-party operator: emits consecutive readings of one input as pairs."""

    type = "acme/pair_up@v1"
    aliases = ("PairUp",)
    input_roles = ("input",)

    class Params(OperatorParams):
        gap: int = Field(default=1, ge=1, description="Readings between members.")

    @classmethod
    def plan_ports(cls, name, params, inputs):
        axis = Axis(f"operators.{name}:members", "sample", stationary=True)
        ports = {"pair": OperatorPort(*inputs[0].kinds, layout=EntryLayout((axis,)))}

        return ports

    def push(self, arrivals):
        raise AssertionError("compilation must not construct an operator")


class OtherPairUp(Operator):
    """Claims the identity of ``PairUp``."""

    type = "acme/pair_up@v1"
    input_roles = ("input",)

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {}

    def push(self, arrivals):
        return []


class AbstractOperator(Operator):
    """No ``type``: an abstract base, not registrable."""


class FutureOperator(Operator):
    """Requires an engine this one is not."""

    type = "acme/future@v1"
    input_roles = ("input",)
    engine_compatibility = ">=3"

    @classmethod
    def plan_ports(cls, name, params, inputs):
        return {}

    def push(self, arrivals):
        return []


class Reading(Source):
    """One scalar reading per pulse."""

    type = "acme/reading@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    def open(self):
        raise AssertionError("compilation must not open a source")

    def read(self):
        raise AssertionError("compilation must not read a source")


class Spread(Block):
    """Uses the pair members as ordinary per-member values."""

    type = "acme/spread@v1"
    outputs = {"value": Output(FLOAT_KIND, source="value")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def run(self, *, value):
        return {"value": value}


def test_operators_are_a_separate_class_owned_registry() -> None:
    catalogue = Catalogue(
        [Spread], sources=[Reading], operators=[PairUp], namespace="acme"
    )

    entry = catalogue.resolve_operator("acme/pair_up@v1")
    assert isinstance(entry, OperatorEntry)
    assert entry.spec.operator_class is PairUp
    assert entry.namespace == "acme"
    assert catalogue.resolve_operator("PairUp") is entry
    assert catalogue.find_operator("acme/unknown@v1") is None
    assert catalogue.operator_types == ("acme/pair_up@v1",)
    assert "acme/pair_up@v1" not in catalogue  # blocks only
    assert catalogue.find("acme/pair_up@v1") is None
    assert "operators=['acme/pair_up@v1']" in repr(catalogue)

    with pytest.raises(CatalogueError, match="Unknown operator type 'nope'"):
        catalogue.resolve_operator("nope")


def test_a_block_and_an_operator_may_spell_the_same_identity() -> None:
    class SameName(Block):
        type = "acme/pair_up@v1"
        outputs = {"value": Output()}

        def run(self):
            return {"value": None}

    catalogue = Catalogue([SameName], operators=[PairUp])

    assert catalogue.entry("acme/pair_up@v1").spec.block_class is SameName
    assert catalogue.resolve_operator("acme/pair_up@v1").spec.operator_class is PairUp


@pytest.mark.parametrize(
    ("operators", "message"),
    [
        ([PairUp, OtherPairUp], "already registered by PairUp"),
        ([AbstractOperator], "is abstract"),
        ([Spread], "Expected an Operator subclass"),
        ([FutureOperator], r"requires engine >=3"),
    ],
    ids=["duplicate-identity", "abstract", "not-an-operator", "incompatible"],
)
def test_invalid_operator_registrations_are_rejected(operators, message) -> None:
    with pytest.raises(CatalogueError, match=message):
        Catalogue(operators=operators)


def test_merge_keeps_one_registration_and_catalogues_stay_immutable() -> None:
    first = Catalogue(operators=[PairUp], namespace="acme")
    second = Catalogue(operators=[PairUp, Window], namespace="acme")

    merged = Catalogue.merge(first, second, Catalogue(operators=[Align]))

    assert merged.operator_types == ("acme/pair_up@v1", "v2/window@v1", "v2/align@v1")
    assert first.operator_types == ("acme/pair_up@v1",)
    assert merged.with_blocks([Spread]).operator_types == merged.operator_types
    with pytest.raises(CatalogueError, match="already registered"):
        Catalogue.merge(first, Catalogue(operators=[OtherPairUp]))


def test_plugin_modules_can_expose_operators(monkeypatch) -> None:
    module = types.ModuleType("acme_operators_plugin")
    module.WORKFLOWS_V2_CATALOGUE = lambda: Catalogue(
        operators=[PairUp], namespace="acme"
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)

    catalogue = Catalogue.from_modules([module.__name__])

    assert catalogue.resolve_operator("PairUp").namespace == "acme"


def test_a_third_party_operator_compiles_like_a_built_in() -> None:
    catalogue = Catalogue(
        [Spread], sources=[Reading], operators=[PairUp], namespace="acme"
    )
    plan = compile_workflow(
        {
            "version": "2.0",
            "sources": [{"type": "acme/reading@v1", "name": "probe"}],
            "operators": [
                {
                    "type": "PairUp",
                    "name": "pairs",
                    "gap": 2,
                    "inputs": {"reading": "$sources.probe.value"},
                }
            ],
            "steps": [
                {
                    "type": "acme/spread@v1",
                    "name": "spread",
                    "value": "$operators.pairs.pair",
                }
            ],
            "outputs": [
                {
                    "type": "OutputGroup",
                    "name": "pairs",
                    "anchor": "$operators.pairs.pair",
                    "outputs": [
                        {
                            "type": "JsonField",
                            "name": "value",
                            "selector": "$steps.spread.value",
                        }
                    ],
                }
            ],
        },
        catalogue=catalogue,
    )

    operator = plan.operator("pairs")
    assert operator.spec.operator_class is PairUp
    assert operator.namespace == "acme"
    assert operator.params.gap == 2
    assert [axis.id for axis in operator.outputs["pair"].layout.axes] == [
        "operators.pairs:members"
    ]
    assert plan.step(("spread",)).domain == "pairs"
    assert plan.axis_origin("operators.pairs:members").describe() == "$operators.pairs"


def test_compiling_with_an_operator_the_catalogue_lacks_names_the_known_types() -> None:
    with pytest.raises(UnknownBlockError, match=r"known operator types: \[\]"):
        compile_workflow(
            {
                "version": "2.0",
                "sources": [{"type": "acme/reading@v1", "name": "probe"}],
                "operators": [
                    {
                        "type": "PairUp",
                        "name": "pairs",
                        "inputs": {"reading": "$sources.probe.value"},
                    }
                ],
                "steps": [],
                "outputs": [],
            },
            catalogue=Catalogue(sources=[Reading]),
        )


def test_catalogue_descriptions_list_operators_with_roles_and_parameters() -> None:
    catalogue = Catalogue(operators=[PairUp, Window], namespace="acme")

    described = catalogue.describe()
    assert [item["type"] for item in described["operators"]] == [
        "acme/pair_up@v1",
        "v2/window@v1",
    ]
    assert described["operators"][0]["namespace"] == "acme"
    assert described["operators"][0]["input_roles"] == ["input"]

    tooling = describe_catalogue(catalogue)
    json.dumps(tooling)
    pair_up = tooling["operators"][0]
    assert pair_up["identities"] == ["acme/pair_up@v1", "PairUp"]
    assert pair_up["parameters"] == {"gap": {"default": {"value": 1}}}
    window = tooling["operators"][1]
    assert window["input_roles"] == ["collect", "hold"]
    assert window["parameters"]["size"] == {"default": None}
    assert "operator_input_maps" in tooling["selector_grammar"]
