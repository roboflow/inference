from typing import Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockManifestMetadata,
    PrimitiveTypeDefinition,
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def test_parse_block_manifest_when_manifest_only_defines_primitive_python_values() -> (
    None
):
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        some_integer: int
        some_string: str
        some_float: float
        some_boolean: bool

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "some_integer": PrimitiveTypeDefinition(
                property_name="some_integer",
                property_description="not available",
                type_annotation="int",
            ),
            "some_string": PrimitiveTypeDefinition(
                property_name="some_string",
                property_description="not available",
                type_annotation="str",
            ),
            "some_float": PrimitiveTypeDefinition(
                property_name="some_float",
                property_description="not available",
                type_annotation="float",
            ),
            "some_boolean": PrimitiveTypeDefinition(
                property_name="some_boolean",
                property_description="not available",
                type_annotation="bool",
            ),
        },
        selectors={},
    )


def test_parse_block_manifest_when_manifest_only_defines_compound_python_values() -> (
    None
):
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        some_list: list
        some_dict: dict
        some_set: set

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "some_list": PrimitiveTypeDefinition(
                property_name="some_list",
                property_description="not available",
                type_annotation="List[Any]",
            ),
            "some_dict": PrimitiveTypeDefinition(
                property_name="some_dict",
                property_description="not available",
                type_annotation="Dict[Any, Any]",
            ),
            "some_set": PrimitiveTypeDefinition(
                property_name="some_set",
                property_description="not available",
                type_annotation="Set[Any]",
            ),
        },
        selectors={},
    )


def test_parse_block_manifest_when_manifest_defines_list_with_detailed_type_annotation() -> (
    None
):
    # given

    class MyObject(BaseModel):
        a: str

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        some_list: List[
            Union[
                None, str, int, float, MyObject, List[str], List[Union[MyObject, bool]]
            ]
        ]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "some_list": PrimitiveTypeDefinition(
                property_name="some_list",
                property_description="not available",
                type_annotation="List[Optional[List[Union[MyObject, bool]], List[str], MyObject, float, int, str]]",
            ),
        },
        selectors={},
    )


def test_parse_block_manifest_when_manifest_defines_dict_with_detailed_type_annotation() -> (
    None
):
    # given

    class MyObject(BaseModel):
        a: str

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        some_dict: Dict[str, Optional[Union[str, MyObject]]]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "some_dict": PrimitiveTypeDefinition(
                property_name="some_dict",
                property_description="not available",
                type_annotation="Dict[str, Optional[MyObject, str]]",
            ),
        },
        selectors={},
    )


def test_parse_block_manifest_when_manifest_defines_set_with_detailed_type_annotation() -> (
    None
):
    # given

    class MyObject(BaseModel):
        a: str

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        some_set: Set[Optional[Union[str, MyObject]]]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "some_set": PrimitiveTypeDefinition(
                property_name="some_set",
                property_description="not available",
                type_annotation="Set[Optional[MyObject, str]]",
            ),
        },
        selectors={},
    )


def test_parse_block_manifest_when_manifest_defines_selectors_without_nesting() -> None:
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        image: WorkflowImageSelector
        input_parameter: WorkflowParameterSelector(
            kind=[BOOLEAN_KIND, STRING_KIND],
        )
        step_output_image: StepOutputImageSelector
        step_output_property: StepOutputSelector(
            kind=[BOOLEAN_KIND, OBJECT_DETECTION_PREDICTION_KIND]
        )
        step: StepSelector

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
        },
        selectors={
            "image": SelectorDefinition(
                property_name="image",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_image",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    )
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            ),
            "input_parameter": SelectorDefinition(
                property_name="input_parameter",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_parameter",
                        kind=[BOOLEAN_KIND, STRING_KIND],
                        points_to_batch={False},
                    )
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            ),
            "step_output_image": SelectorDefinition(
                property_name="step_output_image",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="step_output",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    )
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            ),
            "step_output_property": SelectorDefinition(
                property_name="step_output_property",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="step_output",
                        kind=[
                            BOOLEAN_KIND,
                            OBJECT_DETECTION_PREDICTION_KIND,
                        ],
                        points_to_batch={True},
                    )
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            ),
            "step": SelectorDefinition(
                property_name="step",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="step",
                        kind=[],
                        points_to_batch={False},
                    )
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            ),
        },
    )


def test_parse_block_manifest_when_manifest_defines_compound_selector() -> None:
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        compound: List[
            Union[
                WorkflowImageSelector,
                StepOutputImageSelector,
                List[StepOutputSelector()],
            ]
        ]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
        },
        selectors={
            "compound": SelectorDefinition(
                property_name="compound",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_image",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    ),
                    ReferenceDefinition(
                        selected_element="step_output",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    ),
                    # nested list is ignored
                ],
                is_list_element=True,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )


def test_parse_block_manifest_when_manifest_defines_union_of_selector_and_primitive_type() -> (
    None
):
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        compound: List[
            Union[WorkflowImageSelector, StepOutputImageSelector, str, float]
        ]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "compound": PrimitiveTypeDefinition(
                property_name="compound",
                property_description="not available",
                type_annotation="List[Union[float, str]]",
            ),
        },
        selectors={
            "compound": SelectorDefinition(
                property_name="compound",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_image",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    ),
                    ReferenceDefinition(
                        selected_element="step_output",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    ),
                    # nested list is ignored
                ],
                is_list_element=True,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )


def test_parse_block_manifest_when_manifest_defines_selector_inside_dictionary() -> (
    None
):
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        compound: Dict[
            str, Union[WorkflowImageSelector, StepOutputImageSelector, str, float]
        ]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "compound": PrimitiveTypeDefinition(
                property_name="compound",
                property_description="not available",
                type_annotation="Dict[str, Union[float, str]]",
            ),
        },
        selectors={
            "compound": SelectorDefinition(
                property_name="compound",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_image",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    ),
                    ReferenceDefinition(
                        selected_element="step_output",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    ),
                    # nested list is ignored
                ],
                is_list_element=False,
                is_dict_element=True,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )


def test_parse_block_manifest_when_manifest_defines_list_of_strings_and_string_selectors() -> (
    None
):
    """
    Test for mixed array with literal strings and WorkflowParameterSelector(STRING_KIND).
    This tests the registration_tags use case: ["literal", "$inputs.tag"]
    """
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        registration_tags: List[Union[WorkflowParameterSelector(kind=[STRING_KIND]), str]]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "registration_tags": PrimitiveTypeDefinition(
                property_name="registration_tags",
                property_description="not available",
                type_annotation="List[str]",
            ),
        },
        selectors={
            "registration_tags": SelectorDefinition(
                property_name="registration_tags",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_parameter",
                        kind=[STRING_KIND],
                        points_to_batch={False},
                    ),
                ],
                is_list_element=True,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )


def test_parse_block_manifest_when_manifest_defines_union_of_string_and_string_selector() -> (
    None
):
    """
    Test for a property that can be either a string OR a selector.
    This tests the case: registration_tags: Union[str, WorkflowParameterSelector]
    Expected to resolve to just a string OR a selector to a string.
    """
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        single_tag: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "single_tag": PrimitiveTypeDefinition(
                property_name="single_tag",
                property_description="not available",
                type_annotation="str",
            ),
        },
        selectors={
            "single_tag": SelectorDefinition(
                property_name="single_tag",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_parameter",
                        kind=[STRING_KIND],
                        points_to_batch={False},
                    ),
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )


def test_parse_block_manifest_when_manifest_defines_list_of_strings_and_list_selector() -> (
    None
):
    """
    Test for a property that can be a list of strings OR a selector to a list.
    This tests: Union[List[str], WorkflowParameterSelector]
    Expected to handle the case where $inputs.tags itself is an array.
    """
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        tags: Union[
            List[str],
            WorkflowParameterSelector(kind=[STRING_KIND]),
        ]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "tags": PrimitiveTypeDefinition(
                property_name="tags",
                property_description="not available",
                type_annotation="List[str]",
            ),
        },
        selectors={
            "tags": SelectorDefinition(
                property_name="tags",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_parameter",
                        kind=[STRING_KIND],
                        points_to_batch={False},
                    ),
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )


def test_parse_block_manifest_when_manifest_defines_only_list_of_strings() -> (
    None
):
    """
    Test baseline case: just a list of strings with no selectors.
    This should only create a primitive type definition.
    """
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        static_tags: List[str]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "static_tags": PrimitiveTypeDefinition(
                property_name="static_tags",
                property_description="not available",
                type_annotation="List[str]",
            ),
        },
        selectors={},
    )


def test_parse_block_manifest_when_manifest_defines_only_string_selector_list() -> (
    None
):
    """
    Test a list containing only selectors (no literal strings).
    This tests: List[WorkflowParameterSelector(STRING_KIND)]
    """
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        dynamic_tags: List[WorkflowParameterSelector(kind=[STRING_KIND])]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
        },
        selectors={
            "dynamic_tags": SelectorDefinition(
                property_name="dynamic_tags",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_parameter",
                        kind=[STRING_KIND],
                        points_to_batch={False},
                    ),
                ],
                is_list_element=True,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )


def test_parse_block_manifest_when_manifest_defines_mixed_selectors_and_step_output_selectors() -> (
    None
):
    """
    Test mixed array with different selector types (WorkflowParameter and StepOutput).
    This tests: List[Union[WorkflowParameterSelector, StepOutputSelector, str]]
    """
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        mixed_references: List[
            Union[
                WorkflowParameterSelector(kind=[STRING_KIND]),
                StepOutputSelector(kind=[STRING_KIND]),
                str,
            ]
        ]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    # when
    manifest_metadata = parse_block_manifest(manifest_type=Manifest)

    # then
    assert manifest_metadata == BlockManifestMetadata(
        primitive_types={
            "name": PrimitiveTypeDefinition(
                property_name="name",
                property_description="name field",
                type_annotation="str",
            ),
            "mixed_references": PrimitiveTypeDefinition(
                property_name="mixed_references",
                property_description="not available",
                type_annotation="List[str]",
            ),
        },
        selectors={
            "mixed_references": SelectorDefinition(
                property_name="mixed_references",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_parameter",
                        kind=[STRING_KIND],
                        points_to_batch={False},
                    ),
                    ReferenceDefinition(
                        selected_element="step_output",
                        kind=[STRING_KIND],
                        points_to_batch={True},
                    ),
                ],
                is_list_element=True,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            )
        },
    )
