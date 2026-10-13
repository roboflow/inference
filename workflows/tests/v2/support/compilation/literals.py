"""Literal values and blocks shared by the literal-cast and boundary suites."""

from typing import Dict, Optional

from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)

from tests.v2.support.compilation.fixture_blocks import CATALOGUE


class LiteralBatch(Block):
    """Batch-only, scalar-or-batch and per-call fields that all accept literals."""

    type = "test/literal_batch@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: int | Ref(INTEGER_KIND, batch="always") = 1
        maybe: int | Ref(INTEGER_KIND, batch="if_varying") = 2
        plain: Optional[int] | Ref(INTEGER_KIND) = None

    def run(self, *, value, maybe, plain) -> list:
        return []


class LiteralGroups(Block):
    """Parent reference plus whole and compound groups accepting literals."""

    type = "test/literal_groups@v1"
    outputs = {"kept": Output(preserve="groups"), "whole": Output(preserve="single")}

    class Params(BlockParams):
        parent: Ref()
        groups: Dict[str, int | Group(INTEGER_KIND)]
        single: int | Group(INTEGER_KIND) = 5

    def run(self, *, parent, groups, single) -> dict:
        return {}


LITERALS = Catalogue.merge(CATALOGUE, Catalogue([LiteralBatch, LiteralGroups]))
