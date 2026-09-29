import pytest
from roboflow_workflows.execution_engine.v2.contracts import (
    CONTEXT_POLICIES,
    INPUT_VIEWS,
    OUTPUT_TRANSFORMS,
    SUPPORTED_VIEW_TRANSFORMS,
    BlockContract,
    BlockRegistration,
    InputSpec,
    OutputSpec,
    Registry,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
    WorkflowExecutionError,
)


def _crop_contract() -> BlockContract:
    return BlockContract(
        reference="image",
        inputs={"image": InputSpec(kind="image")},
        outputs={
            "crops": OutputSpec(kind="image", transform="append", axis="regions"),
            "summary": OutputSpec(kind="summary"),
        },
    )


def _reducer_contract() -> BlockContract:
    return BlockContract(
        reference="images",
        inputs={
            "images": InputSpec(kind="image", view="batch"),
            "parent": InputSpec(kind="image"),
        },
        outputs={
            "mosaic": OutputSpec(kind="image", transform="collapse"),
            "count": OutputSpec(kind="number", transform="collapse"),
        },
    )


class _Block:
    def __init__(self, config):
        self.config = dict(config)

    def run(self, **named_inputs):
        return {"crops": Batch.of([]), "summary": {"count": 0}}


def _registry_with_kinds(*kinds: str) -> Registry:
    registry = Registry()
    for kind in kinds:
        registry.register_kind(kind)

    return registry


# --- errors --------------------------------------------------------------------


def test_error_hierarchy() -> None:
    assert issubclass(ContractError, ValueError)
    assert issubclass(WorkflowCompileError, ValueError)
    assert issubclass(WorkflowExecutionError, RuntimeError)
    assert not issubclass(ContractError, WorkflowCompileError)


# --- InputSpec -------------------------------------------------------------------


def test_input_spec_defaults_to_item_view() -> None:
    spec = InputSpec(kind="image")

    assert spec.view == "item"
    assert INPUT_VIEWS == ("item", "batch")


def test_input_spec_rejects_bad_kind_and_view() -> None:
    with pytest.raises(ContractError, match="kind must be a non-empty string"):
        InputSpec(kind="")
    with pytest.raises(ContractError, match="view must be one of"):
        InputSpec(kind="image", view="group")


# --- OutputSpec -----------------------------------------------------------------


def test_output_spec_defaults() -> None:
    spec = OutputSpec(kind="image")

    assert spec.transform == "preserve"
    assert spec.source is None
    assert spec.axis is None
    assert spec.stationary is False
    assert spec.context_policy == "common_or_none"
    assert OUTPUT_TRANSFORMS == ("preserve", "append", "collapse")
    assert CONTEXT_POLICIES == ("common_or_none",)


def test_output_spec_append_requires_axis_key() -> None:
    with pytest.raises(ContractError, match="requires an axis key"):
        OutputSpec(kind="image", transform="append")
    with pytest.raises(ContractError, match="axis must be a non-empty string"):
        OutputSpec(kind="image", transform="append", axis="")


def test_output_spec_axis_and_stationary_only_with_append() -> None:
    with pytest.raises(ContractError, match="only allowed with transform 'append'"):
        OutputSpec(kind="image", transform="preserve", axis="regions")
    with pytest.raises(ContractError, match="stationary=True is only allowed"):
        OutputSpec(kind="image", transform="collapse", stationary=True)


def test_output_spec_rejects_unknown_transform_policy_and_source() -> None:
    with pytest.raises(ContractError, match="transform must be one of"):
        OutputSpec(kind="image", transform="expand")
    with pytest.raises(ContractError, match="context_policy must be one of"):
        OutputSpec(kind="image", context_policy="first")
    with pytest.raises(ContractError, match="valid Python identifier"):
        OutputSpec(kind="image", source="my image")
    with pytest.raises(ContractError, match="stationary must be a bool"):
        OutputSpec(kind="image", transform="append", axis="a", stationary=1)


def test_output_spec_appended_axis_identity_derives_from_producer_and_key() -> None:
    dynamic = OutputSpec(kind="image", transform="append", axis="regions")
    static = OutputSpec(
        kind="image", transform="append", axis="regions", stationary=True
    )

    dynamic_axis = dynamic.appended_axis(producer="crop_a")
    static_axis = static.appended_axis(producer="crop_b")

    assert dynamic.appended_axis_kind == "dynamic_nesting"
    assert static.appended_axis_kind == "static_nesting"
    assert dynamic_axis.id == "crop_a/regions"
    assert dynamic_axis.kind == "dynamic_nesting"
    assert dynamic_axis.stationary is False
    assert static_axis.id == "crop_b/regions"
    assert static_axis.stationary is True
    assert dynamic.appended_axis(producer="a") != dynamic.appended_axis(producer="b")


def test_output_spec_appended_axis_rejects_non_append() -> None:
    with pytest.raises(ContractError, match="appends no axis"):
        OutputSpec(kind="image").appended_axis(producer="x")
    with pytest.raises(ContractError, match="appends no axis"):
        OutputSpec(kind="image", transform="collapse").appended_axis_kind
    with pytest.raises(ContractError, match="Producer identity"):
        OutputSpec(kind="image", transform="append", axis="a").appended_axis(
            producer=""
        )


# --- BlockContract --------------------------------------------------------------


def test_contract_snapshots_inputs_and_outputs_read_only() -> None:
    inputs = {"image": InputSpec(kind="image")}
    outputs = {"out": OutputSpec(kind="image")}

    contract = BlockContract(reference="image", inputs=inputs, outputs=outputs)
    inputs["extra"] = InputSpec(kind="image")
    outputs.clear()

    assert contract.input_names == ("image",)
    assert contract.output_names == ("out",)
    assert contract.mutates_inputs == ()
    assert contract.reference_input.view == "item"
    with pytest.raises(TypeError):
        contract.inputs["x"] = InputSpec(kind="image")
    with pytest.raises(TypeError):
        contract.outputs["x"] = OutputSpec(kind="image")
    with pytest.raises(AttributeError):
        contract.reference = "other"


def test_contract_output_source_defaults_to_reference() -> None:
    contract = _reducer_contract()
    explicit = BlockContract(
        reference="images",
        inputs={
            "images": InputSpec(kind="image", view="batch"),
            "parent": InputSpec(kind="image"),
        },
        outputs={
            "mosaic": OutputSpec(kind="image", transform="collapse"),
            "parent_copy": OutputSpec(kind="image", source="parent"),
        },
    )

    assert contract.output_source("mosaic") == "images"
    assert contract.output_source_view("mosaic") == "batch"
    assert explicit.output_source("parent_copy") == "parent"
    assert explicit.output_source_view("parent_copy") == "item"
    with pytest.raises(ContractError, match="Unknown output 'nope'"):
        contract.output_source("nope")


def test_contract_rejects_unknown_reference_and_source() -> None:
    with pytest.raises(
        ContractError, match="reference 'missing' is not a declared input"
    ):
        BlockContract(
            reference="missing",
            inputs={"image": InputSpec(kind="image")},
            outputs={"out": OutputSpec(kind="image")},
        )
    with pytest.raises(ContractError, match="unknown source input 'other'"):
        BlockContract(
            reference="image",
            inputs={"image": InputSpec(kind="image")},
            outputs={"out": OutputSpec(kind="image", source="other")},
        )


def test_contract_requires_ports_and_valid_names() -> None:
    with pytest.raises(ContractError, match="inputs must declare at least one port"):
        BlockContract(reference="image", inputs={}, outputs={"o": OutputSpec(kind="k")})
    with pytest.raises(ContractError, match="outputs must declare at least one port"):
        BlockContract(
            reference="image", inputs={"image": InputSpec(kind="k")}, outputs={}
        )
    with pytest.raises(ContractError, match="valid Python identifier"):
        BlockContract(
            reference="image",
            inputs={"image": InputSpec(kind="k"), "bad name": InputSpec(kind="k")},
            outputs={"o": OutputSpec(kind="k")},
        )
    with pytest.raises(ContractError, match="must be an instance of InputSpec"):
        BlockContract(
            reference="image",
            inputs={"image": OutputSpec(kind="k")},
            outputs={"o": OutputSpec(kind="k")},
        )
    with pytest.raises(ContractError, match="inputs must be a mapping"):
        BlockContract(
            reference="image", inputs=["image"], outputs={"o": OutputSpec(kind="k")}
        )


@pytest.mark.parametrize(
    "view, transform",
    [
        ("item", "preserve"),
        ("item", "append"),
        ("batch", "preserve"),
        ("batch", "collapse"),
    ],
)
def test_contract_accepts_supported_view_transform_combinations(
    view, transform
) -> None:
    axis = "children" if transform == "append" else None

    contract = BlockContract(
        reference="value",
        inputs={"value": InputSpec(kind="k", view=view)},
        outputs={"out": OutputSpec(kind="k", transform=transform, axis=axis)},
    )

    assert (view, transform) in SUPPORTED_VIEW_TRANSFORMS
    assert contract.output_source_view("out") == view


@pytest.mark.parametrize("view, transform", [("item", "collapse"), ("batch", "append")])
def test_contract_rejects_unsupported_view_transform_combinations(
    view, transform
) -> None:
    axis = "children" if transform == "append" else None

    with pytest.raises(
        ContractError,
        match=f"transform '{transform}' on source input 'value' with view '{view}'",
    ):
        BlockContract(
            reference="value",
            inputs={"value": InputSpec(kind="k", view=view)},
            outputs={"out": OutputSpec(kind="k", transform=transform, axis=axis)},
        )


def test_contract_checks_combination_against_explicit_source_not_reference() -> None:
    contract = BlockContract(
        reference="parent",
        inputs={
            "parent": InputSpec(kind="k"),
            "children": InputSpec(kind="k", view="batch"),
        },
        outputs={
            "total": OutputSpec(kind="k", transform="collapse", source="children")
        },
    )

    assert contract.output_source("total") == "children"
    with pytest.raises(ContractError, match="view 'item'"):
        BlockContract(
            reference="parent",
            inputs={
                "parent": InputSpec(kind="k"),
                "children": InputSpec(kind="k", view="batch"),
            },
            outputs={"total": OutputSpec(kind="k", transform="collapse")},
        )


def test_contract_shared_axis_key_must_agree_on_stationarity() -> None:
    BlockContract(
        reference="image",
        inputs={"image": InputSpec(kind="k")},
        outputs={
            "a": OutputSpec(kind="k", transform="append", axis="regions"),
            "b": OutputSpec(kind="k", transform="append", axis="regions"),
        },
    )
    with pytest.raises(ContractError, match="sharing axis key 'regions' must agree"):
        BlockContract(
            reference="image",
            inputs={"image": InputSpec(kind="k")},
            outputs={
                "a": OutputSpec(kind="k", transform="append", axis="regions"),
                "b": OutputSpec(
                    kind="k", transform="append", axis="regions", stationary=True
                ),
            },
        )


def test_contract_mutates_inputs_must_name_declared_inputs() -> None:
    contract = BlockContract(
        reference="image",
        inputs={"image": InputSpec(kind="k")},
        outputs={"o": OutputSpec(kind="k")},
        mutates_inputs=["image"],
    )

    assert contract.mutates_inputs == ("image",)
    with pytest.raises(ContractError, match="unknown input 'other'"):
        BlockContract(
            reference="image",
            inputs={"image": InputSpec(kind="k")},
            outputs={"o": OutputSpec(kind="k")},
            mutates_inputs=("other",),
        )
    with pytest.raises(ContractError, match="sequence of input names"):
        BlockContract(
            reference="image",
            inputs={"image": InputSpec(kind="k")},
            outputs={"o": OutputSpec(kind="k")},
            mutates_inputs="image",
        )


def test_contract_equality_is_structural() -> None:
    assert _crop_contract() == _crop_contract()
    assert _crop_contract() != _reducer_contract()


# --- BlockRegistration ----------------------------------------------------------


def test_registration_validates_contract_and_factory() -> None:
    registration = BlockRegistration(contract=_crop_contract(), factory=_Block)

    assert registration.contract == _crop_contract()
    assert registration.factory({}).run(image=None)["summary"] == {"count": 0}
    with pytest.raises(ContractError, match="contract must be a BlockContract"):
        BlockRegistration(contract={"reference": "image"}, factory=_Block)
    with pytest.raises(ContractError, match="factory must be callable"):
        BlockRegistration(contract=_crop_contract(), factory="Block")


# --- Registry --------------------------------------------------------------------


def test_registry_starts_empty_and_mutable() -> None:
    registry = Registry()

    assert registry.kind_names == ()
    assert registry.block_names == ()
    assert registry.is_frozen is False
    assert registry.has_kind("image") is False
    assert registry.has_block("v2/crop") is False


def test_registry_registers_kinds_and_blocks_in_order() -> None:
    registry = _registry_with_kinds("image", "summary")
    registry.register_block("v2/crop", contract=_crop_contract(), factory=_Block)

    assert registry.kind_names == ("image", "summary")
    assert registry.block_names == ("v2/crop",)
    assert registry.has_kind("summary") is True
    assert registry.has_block("v2/crop") is True
    assert registry.get_block("v2/crop").contract == _crop_contract()
    assert registry.get_block("v2/crop").factory is _Block


def test_registry_rejects_duplicates_and_bad_names() -> None:
    registry = _registry_with_kinds("image", "summary")
    registry.register_block("v2/crop", contract=_crop_contract(), factory=_Block)

    with pytest.raises(ContractError, match="Kind 'image' is already registered"):
        registry.register_kind("image")
    with pytest.raises(ContractError, match="Block 'v2/crop' is already registered"):
        registry.register_block("v2/crop", contract=_crop_contract(), factory=_Block)
    with pytest.raises(ContractError, match="Kind name must be a non-empty string"):
        registry.register_kind("")
    with pytest.raises(ContractError, match="Block name must be a non-empty string"):
        registry.register_block("", contract=_crop_contract(), factory=_Block)
    with pytest.raises(ContractError, match="validator must be callable"):
        registry.register_kind("boolean", validator=True)


def test_registry_rejects_blocks_with_unregistered_kinds() -> None:
    registry = _registry_with_kinds("image")

    with pytest.raises(
        ContractError, match="output 'summary' uses unregistered kind 'summary'"
    ):
        registry.register_block("v2/crop", contract=_crop_contract(), factory=_Block)
    with pytest.raises(
        ContractError, match="input 'images' uses unregistered kind 'image'"
    ):
        Registry().register_block(
            "v2/mosaic", contract=_reducer_contract(), factory=_Block
        )
    assert registry.block_names == ()


def test_registry_unknown_lookups_fail_with_names() -> None:
    registry = _registry_with_kinds("image")

    with pytest.raises(
        ContractError, match=r"Unknown block 'v2/nope'; registered blocks are \[\]"
    ):
        registry.get_block("v2/nope")
    with pytest.raises(
        ContractError, match=r"Unknown kind 'nope'; registered kinds are \['image'\]"
    ):
        registry.validate("nope", 1)


def test_registry_validate_delegates_to_kind_validator() -> None:
    registry = Registry()
    registry.register_kind("anything")
    registry.register_kind(
        "positive", validator=lambda payload: isinstance(payload, int) and payload > 0
    )

    registry.validate("anything", None)
    registry.validate("anything", [])
    registry.validate("positive", 3)
    with pytest.raises(
        ContractError, match="Payload of type int is not a valid 'positive'"
    ):
        registry.validate("positive", -3)
    with pytest.raises(
        ContractError, match="Payload of type str is not a valid 'positive'"
    ):
        registry.validate("positive", "3")


def test_registry_validate_wraps_raising_validator_and_rejects_non_bool() -> None:
    def exploding(payload):
        raise KeyError("shape")

    registry = Registry()
    registry.register_kind("exploding", validator=exploding)
    registry.register_kind("truthy", validator=lambda payload: payload)

    with pytest.raises(
        ContractError, match="Validator of kind 'exploding' failed"
    ) as info:
        registry.validate("exploding", object())
    assert isinstance(info.value.__cause__, KeyError)
    with pytest.raises(ContractError, match="must return a bool, got int"):
        registry.validate("truthy", 1)


def test_registry_validator_never_inspects_payload_type_itself() -> None:
    # The registry has no notion of images; an opaque object passes a
    # validator-less kind and only a plugin validator can reject it.
    class Opaque:
        pass

    registry = Registry()
    registry.register_kind("opaque")

    registry.validate("opaque", Opaque())
    registry.validate("opaque", Batch.of([Opaque()]))


def test_registry_snapshot_is_isolated_from_later_registrations() -> None:
    registry = _registry_with_kinds("image", "summary")
    registry.register_block("v2/crop", contract=_crop_contract(), factory=_Block)

    snapshot = registry.snapshot()
    registry.register_kind("number")
    registry.register_block("v2/mosaic", contract=_reducer_contract(), factory=_Block)

    assert snapshot.kind_names == ("image", "summary")
    assert snapshot.block_names == ("v2/crop",)
    assert registry.block_names == ("v2/crop", "v2/mosaic")
    assert snapshot.get_block("v2/crop") is registry.get_block("v2/crop")
    with pytest.raises(ContractError, match="Unknown block 'v2/mosaic'"):
        snapshot.get_block("v2/mosaic")
    with pytest.raises(ContractError, match="Unknown kind 'number'"):
        snapshot.validate("number", 1)


def test_registry_snapshot_is_frozen() -> None:
    snapshot = _registry_with_kinds("image").snapshot()

    assert snapshot.is_frozen is True
    with pytest.raises(ContractError, match="frozen snapshot"):
        snapshot.register_kind("summary")
    with pytest.raises(ContractError, match="frozen snapshot"):
        snapshot.register_block("v2/crop", contract=_crop_contract(), factory=_Block)
    assert snapshot.kind_names == ("image",)


def test_registry_snapshot_of_snapshot_is_independent_too() -> None:
    registry = _registry_with_kinds("image")
    first = registry.snapshot()
    second = first.snapshot()
    registry.register_kind("summary")

    assert first.kind_names == ("image",)
    assert second.kind_names == ("image",)
    assert second.is_frozen is True
    assert registry.is_frozen is False


def test_registry_lookup_of_kind_validator_survives_snapshot() -> None:
    registry = Registry()
    registry.register_kind("positive", validator=lambda payload: payload > 0)
    snapshot = registry.snapshot()

    snapshot.validate("positive", 1)
    with pytest.raises(ContractError, match="not a valid 'positive'"):
        snapshot.validate("positive", 0)


def test_factory_receives_config_and_instances_are_separate() -> None:
    registry = _registry_with_kinds("image", "summary")
    registry.register_block("v2/crop", contract=_crop_contract(), factory=_Block)
    registration = registry.get_block("v2/crop")

    first = registration.factory({"regions": [[0, 0, 1, 1]]})
    second = registration.factory({"regions": []})

    assert first is not second
    assert first.config == {"regions": [[0, 0, 1, 1]]}
    assert second.config == {"regions": []}
    assert isinstance(first.run(image=None)["crops"], Batch)
