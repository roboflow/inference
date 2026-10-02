"""Class-owned source declarations, emissions, catalogue registration and selectors.

No test here opens or reads a source through the engine; the runtime suite
covers the lifecycle. These tests read declarations the way the compiler does.
"""

from fractions import Fraction

import pytest
from pydantic import Field
from roboflow_workflows.execution_engine.v2 import (
    Catalogue,
    Emission,
    Source,
    SourceDeclarationError,
    SourceOutput,
    SourceParams,
    spec_of_source,
)
from roboflow_workflows.execution_engine.v2.catalogue import SourceEntry
from roboflow_workflows.execution_engine.v2.data import (
    Axis,
    EntryLayout,
    EntryMetadata,
    InputValue,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
    StepRef,
    parse_selector,
)
from roboflow_workflows.execution_engine.v2.errors import (
    CatalogueError,
    ContractError,
    ParamsValidationError,
    ResolvedParameterError,
    SelectorError,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND


class CsvTemperature(Source):
    """Finite temperature readings; the compiler only reads this declaration."""

    type = "test/csv_temperature@v1"
    aliases = ("CsvTemperature",)
    outputs = {
        "temperature": SourceOutput(FLOAT_KIND, description="Celsius reading"),
        "probe": SourceOutput(STRING_KIND),
    }

    class Params(SourceParams):
        path: str
        probe: str | Ref(STRING_KIND) = "probe-1"
        scale: float | Ref(FLOAT_KIND) = Field(default=1.0, ge=0)

    def __init__(self, *, clock=None):
        self.clock = clock

    def open(self, *, path, probe, scale):
        self.rows = []

    def read(self):
        return None


class Frames(Source):
    """A grouped port: several frames per pulse under one local sample axis."""

    type = "test/frames@v1"
    outputs = {
        "frames": SourceOutput(FLOAT_KIND, layout=EntryLayout((Axis("f", "sample"),))),
        "labels": SourceOutput(STRING_KIND, layout=EntryLayout((Axis("f", "sample"),))),
    }

    def open(self):
        pass

    def read(self):
        return None


class Echo(Block):
    type = "test/csv_temperature@v1"
    outputs = {"value": Output()}

    class Params(BlockParams):
        value: Ref()

    def run(self, *, value):
        return {"value": value}


def _source(**body):
    namespace = {"type": "test/made@v1", "outputs": {"x": SourceOutput()}}
    namespace.update(body)
    namespace.setdefault("open", lambda self, **params: None)
    namespace.setdefault("read", lambda self: None)
    made = type("Made", (Source,), namespace)

    return made


# --- declarations -----------------------------------------------------------


def test_a_source_class_owns_its_whole_declaration() -> None:
    spec = spec_of_source(CsvTemperature)

    assert spec.type == "test/csv_temperature@v1"
    assert spec.identities == ("test/csv_temperature@v1", "CsvTemperature")
    assert list(spec.fields) == ["path", "probe", "scale"]
    assert spec.fields["probe"].role == "item"
    assert spec.fields["path"].role is None
    assert list(spec.outputs) == ["temperature", "probe"]
    assert spec.outputs["temperature"].kind_names == ("float",)
    assert spec.outputs["temperature"].layout == EntryLayout()
    assert [resource.name for resource in spec.resources] == ["clock"]
    assert [kind.name for kind in spec.kinds] == ["float", "string"]
    assert spec.description.startswith("Finite temperature readings")


def test_a_grouped_port_keeps_its_local_layout_in_the_spec() -> None:
    spec = spec_of_source(Frames)

    assert spec.outputs["frames"].layout.axis_ids == ("f",)
    assert spec.outputs["frames"].describe() == {
        "kinds": ["float"],
        "axes": ["f"],
        "description": "",
    }


def test_an_abstract_source_without_type_cannot_be_registered() -> None:
    class Base(Source):
        def open(self):
            pass

        def read(self):
            return None

    with pytest.raises(SourceDeclarationError, match="is abstract"):
        spec_of_source(Base)
    with pytest.raises(SourceDeclarationError, match="Expected a Source subclass"):
        spec_of_source(Echo)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"read": Source.read}, "does not implement read"),
        ({"open": Source.open}, "does not implement open"),
        ({"read": lambda self, chunk: None}, "read\\(\\) takes no required parameters"),
        (
            {
                "open": lambda self: None,
                "Params": type(
                    "P", (SourceParams,), {"__annotations__": {"path": str}}
                ),
            },
            "open\\(\\) does not accept Params field",
        ),
        ({"Params": BlockParams}, "Params must be a subclass of SourceParams"),
        ({"outputs": {}}, "non-empty mapping"),
        ({"outputs": {"bad name": SourceOutput()}}, "output name 'bad name'"),
        ({"outputs": {"x": Output()}}, "must be a SourceOutput"),
        ({"aliases": ("test/made@v1",)}, "alias repeats the canonical type"),
        ({"engine_compatibility": "not a specifier"}, "not a valid specifier"),
        ({"type": "bad type"}, "type must be a non-empty identity"),
    ],
)
def test_invalid_declarations_fail_when_the_class_is_created(body, message) -> None:
    with pytest.raises(SourceDeclarationError, match=message):
        _source(**body)


def test_source_parameters_are_static_configuration_only() -> None:
    with pytest.raises(SourceDeclarationError, match="is a StepRef"):
        _source(
            Params=type("P", (SourceParams,), {"__annotations__": {"next": StepRef}})
        )
    with pytest.raises(SourceDeclarationError, match="is a Group"):
        _source(
            Params=type("P", (SourceParams,), {"__annotations__": {"items": Group()}})
        )
    with pytest.raises(SourceDeclarationError, match="requests batch delivery"):
        _source(
            Params=type(
                "P", (SourceParams,), {"__annotations__": {"v": Ref(batch="always")}}
            )
        )


def test_conflicting_local_axis_declarations_are_rejected() -> None:
    with pytest.raises(SourceDeclarationError, match="declares axis 'f' differently"):
        _source(
            outputs={
                "a": SourceOutput(layout=EntryLayout((Axis("f", "sample"),))),
                "b": SourceOutput(layout=EntryLayout((Axis("f", "static_nesting"),))),
            }
        )


def test_source_output_rejects_time_axes_and_bad_kinds() -> None:
    with pytest.raises(SourceDeclarationError, match="cannot contain a time axis"):
        SourceOutput(layout=EntryLayout((Axis("t", "time"),)))
    with pytest.raises(SourceDeclarationError, match="must list Kind objects"):
        SourceOutput("float")
    with pytest.raises(SourceDeclarationError, match="layout must be an EntryLayout"):
        SourceOutput(layout=("f",))


# --- parameters --------------------------------------------------------------


def test_validate_params_keeps_selectors_and_ignores_type_and_name() -> None:
    spec = spec_of_source(CsvTemperature)

    params = spec.validate_params(
        {"type": "x", "name": "temp", "path": "a.csv", "probe": "$inputs.probe"},
        source_name="temp",
    )

    assert params.path == "a.csv" and params.probe == "$inputs.probe"
    uses = spec.find_selectors(params)
    assert [(use.field, use.selector) for use in uses] == [("probe", "$inputs.probe")]


def test_validate_params_reports_the_source_location() -> None:
    spec = spec_of_source(CsvTemperature)

    with pytest.raises(ParamsValidationError, match=r"\$sources\.temp .*scale"):
        spec.validate_params({"path": "a.csv", "scale": -1}, source_name="temp")
    with pytest.raises(SelectorError, match=r"\$sources\.temp .*malformed selector"):
        spec.validate_params({"path": "a.csv", "probe": "$inputs."}, source_name="temp")


def test_selector_looking_strings_at_literal_positions_stay_literal() -> None:
    spec = spec_of_source(CsvTemperature)

    params = spec.validate_params({"path": "$sources.cam.image"}, source_name="temp")

    assert params.path == "$sources.cam.image"
    assert spec.find_selectors(params) == ()


def test_resolved_arguments_are_checked_before_open() -> None:
    spec = spec_of_source(CsvTemperature)
    params = spec.validate_params(
        {"path": "a.csv", "scale": "$inputs.scale"}, source_name="temp"
    )

    spec.validate_resolved_arguments(
        params, {"path": "a.csv", "probe": "probe-1", "scale": 2.0}
    )
    with pytest.raises(ResolvedParameterError, match="scale"):
        spec.validate_resolved_arguments(
            params, {"path": "a.csv", "probe": "probe-1", "scale": -2.0}
        )


def test_describe_is_json_friendly_and_names_ports_and_resources() -> None:
    description = spec_of_source(CsvTemperature).describe()

    assert description["outputs"]["temperature"]["description"] == "Celsius reading"
    assert description["resources"] == [
        {"name": "clock", "required": False, "annotation": None}
    ]
    assert "probe" in description["params_schema"]["properties"]


# --- emissions ---------------------------------------------------------------


def test_emission_distinguishes_filtered_absent_and_present_values() -> None:
    pts = Timestamp(40, Fraction(1, 1000), "camera-clock")

    filtered = Emission({})
    partial = Emission({"temperature": []}, media=pts, source_metadata={"row": 3})

    assert filtered.is_filtered and not partial.is_filtered
    assert partial.payload("temperature") == []
    assert partial.media is pts and partial.capture is None
    assert dict(partial.source_metadata) == {"row": 3}
    assert "probe" not in partial.data


def test_emission_values_may_carry_indexed_metadata() -> None:
    metadata = EntryMetadata(temporal={(): None})
    emission = Emission({"temperature": InputValue(21.5, metadata)})

    assert emission.payload("temperature") == 21.5
    assert emission.data["temperature"].metadata is metadata


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"data": [1]}, "must map port names"),
        ({"data": {1: 2}}, "port names must be strings"),
        ({"data": {}, "media": 40}, "media must be a Timestamp"),
        ({"data": {}, "source_metadata": [1]}, "source_metadata must be a mapping"),
    ],
)
def test_invalid_emissions_are_rejected(kwargs, message) -> None:
    with pytest.raises(ContractError, match=message):
        Emission(**kwargs)


def test_emission_data_is_snapshotted() -> None:
    data = {"temperature": 1.0}
    emission = Emission(data)
    data["probe"] = "late"

    assert list(emission.data) == ["temperature"]
    with pytest.raises(TypeError):
        emission.data["x"] = 1


# --- catalogue ---------------------------------------------------------------


def test_catalogue_registers_sources_beside_blocks() -> None:
    catalogue = Catalogue([Echo], sources=[CsvTemperature, Frames], namespace="test")

    assert catalogue.source_types == ("test/csv_temperature@v1", "test/frames@v1")
    assert catalogue.block_types == ("test/csv_temperature@v1",)
    entry = catalogue.find_source("CsvTemperature")
    assert isinstance(entry, SourceEntry)
    assert entry.spec.source_class is CsvTemperature and entry.namespace == "test"
    assert catalogue.find_source("test/none@v1") is None
    assert catalogue.kinds["float"] is FLOAT_KIND
    assert [item["type"] for item in catalogue.describe()["sources"]] == list(
        catalogue.source_types
    )
    assert catalogue.describe()["sources"][0]["namespace"] == "test"


def test_a_block_and_a_source_may_share_an_identity() -> None:
    catalogue = Catalogue([Echo], sources=[CsvTemperature])

    assert catalogue.entry("test/csv_temperature@v1").spec.block_class is Echo
    assert (
        catalogue.source_entry("test/csv_temperature@v1").spec.source_class
        is CsvTemperature
    )


def test_unknown_or_conflicting_sources_are_catalogue_errors() -> None:
    catalogue = Catalogue(sources=[CsvTemperature])
    other = _source(type="test/csv_temperature@v1")

    with pytest.raises(CatalogueError, match="Unknown source type 'test/x@v1'"):
        catalogue.source_entry("test/x@v1")
    with pytest.raises(CatalogueError, match="Source identity .* already registered"):
        Catalogue(sources=[CsvTemperature, other])
    with pytest.raises(CatalogueError, match="Cannot register"):
        Catalogue(sources=[Echo])


def test_merge_keeps_sources_and_deduplicates_the_same_class() -> None:
    merged = Catalogue.merge(
        Catalogue(sources=[CsvTemperature], namespace="a"),
        Catalogue(sources=[CsvTemperature, Frames], namespace="a"),
    )

    assert merged.source_types == ("test/csv_temperature@v1", "test/frames@v1")
    assert merged.with_blocks([Echo]).source_types == merged.source_types


def test_incompatible_source_is_rejected_at_registration() -> None:
    incompatible = _source(engine_compatibility=">=9")

    with pytest.raises(CatalogueError, match="requires engine >=9"):
        Catalogue(sources=[incompatible])


# --- selectors ---------------------------------------------------------------


def test_source_selectors_parse_and_reject_wildcards() -> None:
    parsed = parse_selector("$sources.camera.image")

    assert (parsed.target, parsed.name, parsed.output) == (
        "source_output",
        "camera",
        "image",
    )
    for text in ("$sources.camera", "$sources.camera.*", "$sources..image"):
        with pytest.raises(SelectorError):
            parse_selector(text)


# --- synchronous lifecycle ---------------------------------------------------


async def _async_open(self, **params):
    return None


async def _async_read(self):
    return None


async def _async_generator_read(self):
    yield Emission({})


async def _async_close(self):
    return None


@pytest.mark.parametrize(
    ("body", "method"),
    [
        ({"open": _async_open}, "open"),
        ({"read": _async_read}, "read"),
        ({"read": _async_generator_read}, "read"),
        ({"close": _async_close}, "close"),
    ],
)
def test_asynchronous_lifecycle_methods_are_rejected_at_declaration(
    body, method
) -> None:
    with pytest.raises(
        SourceDeclarationError,
        match=f"{method}\\(\\) must be a plain synchronous method",
    ):
        _source(**body)


def test_a_synchronous_close_override_is_accepted() -> None:
    made = _source(close=lambda self: None)

    assert spec_of_source(made).type == "test/made@v1"
