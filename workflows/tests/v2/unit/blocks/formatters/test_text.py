"""``roboflow_core/string_template@v1`` on plain values."""

import pytest
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.formatters.string_template.v1 import MAX_RENDERED_TEMPLATE_LENGTH, StringTemplateBlock, render_template

JOIN = [{"type": "SequenceJoin", "separator": ", "}]


def _render(template: str, data: dict, data_operations=None) -> str:
    params = StringTemplateBlock.Params(
        template=template,
        data={name: f"$inputs.{name}" for name in data},
        data_operations=data_operations or {},
    )
    result = StringTemplateBlock().run(
        template=template, data=data, data_operations=params.data_operations
    )

    return result["output"]


def test_placeholders_take_format_specs_conversions_and_literal_braces() -> None:
    rendered = _render(
        "{count} at {score:.2f} {{raw}} {name!r} {labels}",
        {"count": 3, "score": 0.8765, "name": "gate", "labels": ["car", "dog"]},
    )

    assert rendered == "3 at 0.88 {raw} 'gate' ['car', 'dog']"


def test_data_operations_transform_a_variable_before_rendering() -> None:
    rendered = _render(
        "Answer with one of: {skus}.",
        {"skus": ["A-1", "B-2", "C-3"]},
        data_operations={"skus": JOIN},
    )

    assert rendered == "Answer with one of: A-1, B-2, C-3."


def test_a_placeholder_may_repeat_and_a_variable_may_stay_unused() -> None:
    rendered = _render("{a}-{a}", {"a": 1, "unused": 2})

    assert rendered == "1-1"


def test_a_literal_template_with_an_undeclared_placeholder_fails_validation() -> None:
    with pytest.raises(ValidationError, match=r"uses \['missing'\]"):
        StringTemplateBlock.Params(template="{missing}", data={"a": "$inputs.a"})


def test_a_literal_template_starting_with_a_dollar_is_still_checked() -> None:
    with pytest.raises(ValidationError, match=r"uses \['item'\]"):
        StringTemplateBlock.Params(template="$5 for {item}")


def test_render_template_checks_placeholders_for_direct_callers() -> None:
    params = StringTemplateBlock.Params(template="$inputs.template")

    assert params.template == "$inputs.template"
    with pytest.raises(ValueError, match=r"uses \['x'\]"):
        render_template("{x}", variables={})


def test_operations_for_an_unknown_variable_fail_validation() -> None:
    with pytest.raises(ValidationError, match="data_operations names `skus`"):
        StringTemplateBlock.Params(template="hi", data_operations={"skus": JOIN})


@pytest.mark.parametrize(
    ("template", "message"),
    [
        ("{}", "positional placeholders"),
        ("{0}", "positional placeholders"),
        ("{a.real}", "attribute or index access"),
        ("{a[0]}", "attribute or index access"),
        ("{a", "could not parse"),
        ("{a:{missing}}", r"uses \['missing'\]"),
    ],
)
def test_unsupported_placeholders_are_rejected(template: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StringTemplateBlock.Params(template=template, data={"a": "$inputs.a"})


def test_a_format_spec_not_matching_its_value_names_the_cause() -> None:
    with pytest.raises(ValueError, match="Unknown format code 'd'"):
        _render("{a:d}", {"a": "text"})


def test_widths_above_the_limit_are_rejected_before_allocation() -> None:
    with pytest.raises(ValueError, match="uses 20000"):
        _render("{a:>20000}", {"a": 1})


def test_a_runtime_width_from_a_nested_placeholder_is_also_limited() -> None:
    with pytest.raises(ValueError, match="uses 50000"):
        _render("{a:>{width}}", {"a": 1, "width": 50000})


def test_a_rendered_string_above_the_limit_fails() -> None:
    with pytest.raises(ValueError, match="more than the maximum"):
        _render("{a}{a}", {"a": "x" * (MAX_RENDERED_TEMPLATE_LENGTH // 2 + 1)})


def test_a_format_spec_exactly_at_the_limit_is_accepted() -> None:
    rendered = _render("{a:>10000}", {"a": "x"})

    assert rendered == " " * 9_999 + "x"


def test_a_rendered_string_exactly_at_the_limit_is_accepted() -> None:
    value = "x" * MAX_RENDERED_TEMPLATE_LENGTH

    assert _render("{a}", {"a": value}) == value
