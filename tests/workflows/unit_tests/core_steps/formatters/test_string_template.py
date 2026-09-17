import pytest

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    OperationsChain,
)
from inference.core.workflows.core_steps.formatters.string_template.v1 import (
    StringTemplateBlockV1,
)


def parse_operations(operations: list) -> list:
    return OperationsChain.model_validate({"operations": operations}).operations


def test_block_run_when_template_has_no_placeholders() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(template="static prompt", data={}, data_operations={})

    # then
    assert result == {"output": "static prompt"}


def test_block_run_when_placeholders_filled_from_data() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(
        template="Found {count} instances of {label}",
        data={"count": 3, "label": "person"},
        data_operations={},
    )

    # then
    assert result == {"output": "Found 3 instances of person"}


def test_block_run_when_extra_variables_declared_in_data() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(
        template="Hello {name}",
        data={"name": "world", "unused": "value"},
        data_operations={},
    )

    # then
    assert result == {"output": "Hello world"}


def test_block_run_when_data_operations_transform_variable_before_substitution() -> (
    None
):
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(
        template="This facing contains one of: {sku_list}. Answer with the product name or NONE.",
        data={"sku_list": ["SKU-1", "SKU-2", "SKU-3"]},
        data_operations={
            "sku_list": parse_operations([{"type": "SequenceJoin", "separator": ", "}])
        },
    )

    # then
    assert result == {
        "output": "This facing contains one of: SKU-1, SKU-2, SKU-3. "
        "Answer with the product name or NONE."
    }


def test_block_run_when_template_contains_literal_braces() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(
        template='Answer in JSON: {{"class": "{label}"}}',
        data={"label": "person"},
        data_operations={},
    )

    # then
    assert result == {"output": 'Answer in JSON: {"class": "person"}'}


def test_block_run_when_format_spec_used() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(
        template="Confidence: {confidence:.2f}",
        data={"confidence": 0.98765},
        data_operations={},
    )

    # then
    assert result == {"output": "Confidence: 0.99"}


def test_block_run_when_nested_format_spec_variable_is_declared() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(
        template="Confidence: {confidence:{width}}",
        data={"confidence": 0.98765, "width": ".2f"},
        data_operations={},
    )

    # then
    assert result == {"output": "Confidence: 0.99"}


def test_block_run_when_nested_format_spec_variable_is_undeclared() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="Confidence: {confidence:{width}}",
            data={"confidence": 0.98765},
            data_operations={},
        )

    # then
    assert "width" in str(error.value)


def test_block_run_when_nested_format_spec_uses_attribute_access() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="{x:{y.__class__}}",
            data={"x": 1, "y": "world"},
            data_operations={},
        )

    # then
    assert "attribute or index access" in str(error.value)


def test_block_run_when_nested_format_spec_uses_positional_placeholder() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="{x:{0}}",
            data={"x": 1},
            data_operations={},
        )

    # then
    assert "positional" in str(error.value)


def test_block_run_when_template_references_undeclared_variable() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(template="Hello {name}", data={}, data_operations={})

    # then
    assert "name" in str(error.value)


def test_block_run_when_template_uses_positional_placeholder() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(template="Hello {}", data={"name": "world"}, data_operations={})

    # then
    assert "positional" in str(error.value)


def test_block_run_when_template_uses_indexed_placeholder() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(template="Hello {0}", data={"name": "world"}, data_operations={})

    # then
    assert "positional" in str(error.value)


def test_block_run_when_template_uses_attribute_access() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="{name.__class__}", data={"name": "world"}, data_operations={}
        )

    # then
    assert "attribute or index access" in str(error.value)


def test_block_run_when_template_uses_index_access() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(template="{names[0]}", data={"names": ["world"]}, data_operations={})

    # then
    assert "attribute or index access" in str(error.value)


def test_block_run_when_template_has_unbalanced_brace() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(template="Hello {name", data={"name": "world"}, data_operations={})

    # then
    assert "could not parse template" in str(error.value)


def test_block_run_when_operations_declared_for_undeclared_variable() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="Hello {name}",
            data={"name": "world"},
            data_operations={"other": [{"type": "SequenceJoin", "separator": ", "}]},
        )

    # then
    assert "other" in str(error.value)


def test_block_run_when_template_is_not_a_string() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(template=42, data={}, data_operations={})

    # then
    assert "expected template to be a string" in str(error.value)


def test_block_run_when_format_spec_declares_huge_literal_width() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(template="{value:1000000000}", data={"value": "x"}, data_operations={})

    # then
    assert "10000" in str(error.value)


def test_block_run_when_format_spec_declares_huge_width_at_runtime() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="{value:{width}}",
            data={"value": "x", "width": "999999999"},
            data_operations={},
        )

    # then
    assert "10000" in str(error.value)


def test_block_run_when_format_spec_declares_huge_precision() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="{value:.999999999f}", data={"value": 1.5}, data_operations={}
        )

    # then
    assert "10000" in str(error.value)


def test_block_run_when_rendered_output_exceeds_length_limit() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    with pytest.raises(ValueError) as error:
        step.run(
            template="{value}{value}",
            data={"value": "a" * 600_000},
            data_operations={},
        )

    # then
    assert "1000000" in str(error.value)


def test_block_run_when_format_specs_are_within_limits() -> None:
    # given
    step = StringTemplateBlockV1()

    # when
    result = step.run(
        template="{confidence:.2f} {value:>10}",
        data={"confidence": 0.12345, "value": "ok"},
        data_operations={},
    )

    # then
    assert result == {"output": "0.12         ok"}
