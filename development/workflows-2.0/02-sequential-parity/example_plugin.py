"""A small V2 plugin module, loaded with ``Catalogue.from_modules``.

It exposes ``WORKFLOWS_V2_CATALOGUE`` like any installable plugin:

* ``plugin/ledger@v1`` appends to a ``ledger`` list resource. The plugin
  provides it as ``Factory(list, scope="session")``: steps of one session share
  one list, a new session gets a new one.
* ``plugin/locate@v1`` returns a ``point`` whose kind converts it for each
  workflow output: ``"coordinates_system": "parent"`` adds the origin.
* ``plugin/load_model@v1`` declares the model it needs, so workload discovery
  can list it without running anything.
"""

from typing import Any, Dict, List, Mapping

from pydantic import Field
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Kind,
)
from roboflow_workflows.execution_engine.v2.resources import Factory


def _to_output(point: Dict[str, Any], options: Mapping[str, Any]) -> Dict[str, Any]:
    x, y = point["x"], point["y"]
    if options.get("coordinates_system") == "parent":
        origin_x, origin_y = point["origin"]
        x, y = x + origin_x, y + origin_y

    return {"x": x, "y": y}


POINT_KIND = Kind(
    name="point",
    description="Point in its own coordinates, with the origin of its parent.",
    validate=lambda payload: isinstance(payload, dict) and {"x", "y"} <= payload.keys(),
    convert_output=_to_output,
)


class Ledger(Block):
    """Append a value to the session's ledger and report its size."""

    type = "plugin/ledger@v1"
    outputs = {"size": Output(INTEGER_KIND)}

    class Params(BlockParams):
        value: Ref() = Field(description="Value to append.")

    def __init__(self, *, ledger: List[Any]):
        self.ledger = ledger

    def run(self, *, value: Any) -> dict:
        self.ledger.append(value)

        return {"size": len(self.ledger)}


class Locate(Block):
    """Place a point relative to an origin."""

    type = "plugin/locate@v1"
    outputs = {"point": Output(POINT_KIND)}

    class Params(BlockParams):
        x: float | Ref(FLOAT_KIND) = Field(description="X in own coordinates.")
        y: float | Ref(FLOAT_KIND) = Field(description="Y in own coordinates.")
        origin: List[float] = Field(description="Origin inside the parent.")

    def run(self, *, x: float, y: float, origin: List[float]) -> dict:
        return {"point": {"x": x, "y": y, "origin": origin}}


class LoadModel(Block):
    """Pretend to load a model; declares it as a dependent resource."""

    type = "plugin/load_model@v1"
    outputs = {"model": Output(STRING_KIND)}

    class Params(BlockParams):
        model_id: str | Ref(STRING_KIND) = Field(description="Model to load.")

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        return [DependentResource(resource_type="model", identifier=params.model_id)]

    def run(self, *, model_id: str) -> dict:
        return {"model": f"loaded {model_id}"}


def create_plugin_catalogue() -> Catalogue:
    """Collect the plugin blocks with the session-scoped ledger provider.

    Returns:
        Catalogue in namespace ``example_plugin``.
    """
    catalogue = Catalogue(
        [Ledger, Locate, LoadModel],
        kinds=[POINT_KIND],
        namespace="example_plugin",
        providers={"ledger": Factory(list, scope="session")},
    )

    return catalogue


WORKFLOWS_V2_CATALOGUE = create_plugin_catalogue
