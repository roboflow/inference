"""Contract digests of ``Json`` fields, and merging of re-read implementations.

Synthetic blocks only. The digest must keep the constraints of a ``Json[Model]``
field's decoded content and ignore its documentation and model class names.
Merging keeps one block class once only when both catalogues hold the same
validated declaration of every implementation, compared without calling
``==`` on user objects such as resource defaults.
"""

import dataclasses
from typing import Annotated, Any, Callable, Dict, Iterator, List, Optional

import pytest
from pydantic import BaseModel, ConfigDict, Field, Json, create_model
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    Implementation,
    ImplementationContribution,
    ImplementationSpec,
    Output,
    contract_digest,
    phase,
    spec_of,
)
from roboflow_workflows.v2.engine.contributions import _canonical_schema
from roboflow_workflows.v2.engine.errors import (
    CatalogueError,
    ParamsValidationError,
)

# --- Json[Model] parameters ---------------------------------------------------


def region_model(
    *,
    name: str = "Region",
    minimum: int = 0,
    field_name: str = "width",
    field_doc: str = "Width in pixels.",
    model_doc: Optional[str] = None,
    title: Optional[str] = None,
) -> type:
    inner = create_model(
        "Inner", __base__=BaseModel, depth=(int, Field(default=1, le=4 + minimum))
    )
    region = create_model(
        name,
        __base__=BaseModel,
        __config__=None if title is None else ConfigDict(title=title),
        __doc__=model_doc,
        inner=(inner, Field(default_factory=inner)),
        **{field_name: (int, Field(ge=minimum, description=field_doc))},
    )

    return region


def json_block(annotation: Any, *, params_name: str = "Params") -> type:
    params = create_model(params_name, __base__=BlockParams, config=(annotation, ...))

    class JsonConfigured(Block):
        type = "probe/json_config@v1"
        Params = params
        outputs = {"value": Output(INTEGER_KIND)}

        def run(self, *, config: Any) -> dict:
            return {"value": 1}

    return JsonConfigured


def region_block(**edit: Any) -> type:
    params_name = edit.pop("params_name", "Params")
    block = json_block(Json[region_model(**edit)], params_name=params_name)

    return block


def all_references(node: Any) -> Iterator[str]:
    """Every ``$ref`` value anywhere in a schema, whatever its keyword."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref":
                yield value
            else:
                yield from all_references(value)
    elif isinstance(node, list):
        for item in node:
            yield from all_references(item)


def test_json_model_constraints_change_the_digest() -> None:
    baseline = contract_digest(region_block())

    assert contract_digest(region_block(minimum=10)) != baseline


def test_json_nested_model_constraints_change_the_digest() -> None:
    # Inner's bound moves with ``minimum``; reaching it needs Region's $defs entry.
    unrestricted = spec_of(region_block())
    restricted = spec_of(region_block(minimum=10))
    deep = '{"width": 20, "inner": {"depth": 10}}'

    assert restricted.validate_params({"config": deep}).config.inner.depth == 10
    with pytest.raises(ParamsValidationError):
        unrestricted.validate_params({"config": deep})
    assert contract_digest(unrestricted) != contract_digest(restricted)


def test_contribution_pinned_to_another_json_constraint_is_rejected() -> None:
    unrestricted = region_block()
    restricted = region_block(minimum=10)

    class JsonImplementation(Implementation):
        name = "json-config"

        def run(self, *, config: Any) -> dict:
            return {"value": 1}

    pinned = ImplementationContribution(
        block_type=unrestricted.type,
        contract=contract_digest(unrestricted),
        implementations=(JsonImplementation,),
        provider="example-provider",
        version="1.0.0",
    )

    assert Catalogue([unrestricted]).contribution_problem(pinned) is None
    problem = Catalogue([restricted]).contribution_problem(pinned)
    assert problem is not None and "was built for contract" in problem
    assert problem.endswith(
        "use a provider release built for the current contract, or leave this "
        "contribution out so the block runs its own implementations"
    )


@pytest.mark.parametrize(
    "edit",
    [
        {"name": "Area"},
        {"params_name": "JsonParameters"},
        {"field_doc": "Another description."},
        {"model_doc": "A documented region."},
        {"title": "Documented region"},
    ],
    ids=["model-renamed", "params-renamed", "field-doc", "model-doc", "model-title"],
)
def test_json_model_names_and_documentation_keep_the_digest(edit: dict) -> None:
    assert contract_digest(region_block(**edit)) == contract_digest(region_block())


def test_inline_json_content_keeps_constraints_and_drops_documentation() -> None:
    def inline(*, minimum: int = 0, doc: str = "Count.") -> type:
        content = Annotated[int, Field(ge=minimum, description=doc)]
        block = json_block(Json[content])

        return block

    schema = spec_of(inline()).params_schema()
    assert "contentSchema" in schema["properties"]["config"]

    assert contract_digest(inline(doc="Another count.")) == contract_digest(inline())
    assert contract_digest(inline(minimum=3)) != contract_digest(inline())


def test_json_model_field_named_description_is_part_of_the_contract() -> None:
    digests = {
        contract_digest(region_block(field_name=name))
        for name in ("description", "title", "label")
    }

    assert len(digests) == 3
    assert contract_digest(
        region_block(field_name="description", field_doc="Other.")
    ) == contract_digest(region_block(field_name="description"))


@pytest.mark.parametrize(
    "annotation",
    [
        lambda region: Json[region],
        lambda region: Optional[Json[region]],
        lambda region: List[Json[region]],
        lambda region: Json[List[region]],
        lambda region: Dict[str, Json[region]],
    ],
    ids=["json", "optional-json", "list-of-json", "json-list", "dict-of-json"],
)
def test_canonical_references_resolve_to_retained_definitions(
    annotation: Callable[[type], Any],
) -> None:
    schema = spec_of(json_block(annotation(region_model()))).params_schema()
    canonical = _canonical_schema(schema)

    retained = canonical.get("$defs", {})
    assert len(retained) == len(schema["$defs"]) == 2
    references = list(all_references(canonical))
    assert references
    assert set(references) <= {f"#/$defs/{alias}" for alias in retained}


# --- merging re-read implementation declarations ------------------------------


class NoEquality:
    """A user object; merging must compare it by identity, never with ``==``."""

    def __eq__(self, other: object) -> bool:
        raise AssertionError("a catalogue merge called == on a user object")

    __hash__ = object.__hash__


CACHE = NoEquality()
MARKER = NoEquality()


class Public(Block):
    type = "probe/merge@v1"
    quality = ("fast",)
    outputs = {"value": Output(INTEGER_KIND)}

    class Params(BlockParams):
        pass

    def __init__(self, *, cache: Annotated[Any, MARKER] = CACHE):
        self.cache = cache

    def run(self) -> dict:
        return {"value": 1}


def provider_class() -> type:
    class Supplied(Implementation):
        name = "supplied"
        requires = ("cpu",)

        def __init__(self, *, cache: Annotated[Any, MARKER] = CACHE):
            self.cache = cache

        def run(self) -> dict:
            return {"value": 1}

    return Supplied


def contribution(
    supplied: type, *, version: str = "1.0.0"
) -> ImplementationContribution:
    declared = ImplementationContribution(
        block_type=Public.type,
        contract=contract_digest(Public),
        implementations=(supplied,),
        provider="example-provider",
        version=version,
    )

    return declared


def public() -> Catalogue:
    catalogue = Catalogue([Public], namespace="probe")

    return catalogue


def test_independently_read_equal_declarations_merge_without_user_equality() -> None:
    supplied = provider_class()
    first = public().with_contributions([contribution(supplied)])
    second = public().with_contributions([contribution(supplied)])

    for pair in ((first, second), (second, first), (public(), public())):
        merged = Catalogue.merge(*pair)
        assert merged.entry(Public.type).spec is pair[0].entry(Public.type).spec


def renamed(supplied: type) -> ImplementationContribution:
    supplied.name = "renamed"

    declared = contribution(supplied)

    return declared


def replaced(supplied: type) -> ImplementationContribution:
    declared = contribution(provider_class())

    return declared


def newer(supplied: type) -> ImplementationContribution:
    declared = contribution(supplied, version="1.1.0")

    return declared


def more_requirements(supplied: type) -> ImplementationContribution:
    supplied.requires = ("cuda",)

    declared = contribution(supplied)

    return declared


def labelled(supplied: type) -> ImplementationContribution:
    supplied.quality = ("fast",)

    declared = contribution(supplied)

    return declared


def phased(supplied: type) -> ImplementationContribution:
    def prepared(self: Any) -> dict:
        return {"value": 1}

    supplied.prepared = phase(prepared)

    declared = contribution(supplied)

    return declared


def serial(supplied: type) -> ImplementationContribution:
    supplied.phase_overlap = False

    declared = contribution(supplied)

    return declared


def other_default(supplied: type) -> ImplementationContribution:
    def __init__(self: Any, *, cache: Annotated[Any, MARKER] = NoEquality()):
        self.cache = cache

    supplied.__init__ = __init__

    declared = contribution(supplied)

    return declared


def other_annotation(supplied: type) -> ImplementationContribution:
    def __init__(self: Any, *, cache: Annotated[Any, NoEquality()] = CACHE):
        self.cache = cache

    supplied.__init__ = __init__

    declared = contribution(supplied)

    return declared


# ImplementationSpec field -> (re-declaration changing only that field,
# words of the expected merge error).
LISTED = "lists implementations"
DECLARED = r"declares implementation\(s\) \['supplied'\] differently"
CHANGES = {
    "name": ([renamed], LISTED),
    "implementation_class": ([replaced], DECLARED),
    "origin": ([newer], LISTED),
    "requires": ([more_requirements], DECLARED),
    "quality": ([labelled], DECLARED),
    "phases": ([phased], DECLARED),
    "phase_overlap": ([serial], DECLARED),
    "resources": ([other_default, other_annotation], DECLARED),
}
CASES = [
    pytest.param(change, message, id=f"{field_name}-{change.__name__}")
    for field_name, (changes, message) in CHANGES.items()
    for change in changes
]


def test_every_implementation_field_has_a_merge_case() -> None:
    fields = {item.name for item in dataclasses.fields(ImplementationSpec)}

    assert fields == set(CHANGES)


@pytest.mark.parametrize("change, message", CASES)
def test_one_changed_implementation_declaration_does_not_merge(
    change: Callable[[type], ImplementationContribution], message: str
) -> None:
    supplied = provider_class()
    first = public().with_contributions([contribution(supplied)])
    unchanged = public().with_contributions([contribution(supplied)])
    Catalogue.merge(first, unchanged)

    changed = public().with_contributions([change(supplied)])

    for pair in ((first, changed), (changed, first)):
        with pytest.raises(CatalogueError, match=message) as raised:
            Catalogue.merge(*pair)
        assert str(raised.value).endswith("apply contributions once, after merging")
