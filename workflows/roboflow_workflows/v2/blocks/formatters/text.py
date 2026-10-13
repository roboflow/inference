"""String template block: fill named placeholders from workflow data.

Port of V1 ``roboflow_core/string_template@v1``::

    data[name] ── data_operations[name] ──▶ variables ──▶ template ──▶ output

Placeholders follow ``str.format``: ``{name}``, ``{name:.2f}``, ``{name!r}``
and doubled braces for literal ones. Positional (``{}``, ``{0}``), attribute
(``{a.b}``) and index (``{a[0]}``) placeholders are rejected; extract nested
values with ``data_operations`` instead.

As in V1, format-spec widths/precisions above 10 000 and rendered strings
above 1 000 000 characters fail the step. Unlike V1, a literal template and the
``data_operations`` keys are checked while the workflow compiles; a template
bound to a selector is checked by the same Params validator after resolution.
Present ``None`` variables render as ``None``; unavailable or gated rows still
skip the invocation.
"""

import re
from string import Formatter
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from pydantic import (
    Field,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from roboflow_workflows.v2.blocks.query import (
    CompiledOperations,
    Operation,
    compile_operations,
)
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.engine.declaration import (
    DATA_SELECTOR_PATTERN,
    Block,
    BlockParams,
    Output,
    Ref,
)

MAX_FORMAT_SPEC_NUMBER = 10_000
MAX_RENDERED_TEMPLATE_LENGTH = 1_000_000
BLOCK_TYPE = "v2/string_template@v1"


class StringTemplateBlock(Block):
    """Render a template whose named placeholders come from ``data``.

    Example: ``template="{count} cars on {camera}"`` with
    ``data={"count": "$steps.count.output", "camera": "$inputs.camera"}``
    outputs ``"3 cars on gate"``.
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "String Template",
        "port_of": "roboflow_core/string_template@v1",
        "section": "advanced",
    }
    outputs = {"output": Output(STRING_KIND, description="The rendered string.")}

    class Params(BlockParams):
        template: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Template with named placeholders in curly braces; double the "
                "braces for literal ones."
            ),
            examples=["{count} cars on {camera}", "$inputs.prompt_template"],
        )
        data: Dict[str, Any | Ref(allow_none=True)] = Field(
            default_factory=dict,
            description=(
                "Values of the placeholders, by name: literals or selectors. "
                "A present None renders as 'None'; use "
                "first_non_empty_or_default for a fallback."
            ),
            examples=[{"count": "$steps.count.output", "camera": "$inputs.camera"}],
        )
        data_operations: Dict[str, List[Operation]] = Field(
            default_factory=dict,
            description=(
                "UQL operations applied to a variable before rendering, by "
                "variable name, e.g. SequenceJoin to turn a list into text. "
                "Every key must be a data key; the operations name no runtime "
                "values."
            ),
            examples=[{"labels": [{"type": "SequenceJoin", "separator": ", "}]}],
        )

        # Not a model validator: the engine reruns those per invocation.
        # This literal-only field is validated once per definition, and
        # selectors replace data values, never keys.
        @field_validator("data_operations")
        @classmethod
        def _operations_compile(
            cls, value: Dict[str, List[Operation]], info: ValidationInfo
        ) -> Dict[str, List[Operation]]:
            if "data" in info.data:
                compile_data_operations(
                    value, variable_names=frozenset(info.data["data"])
                )

            return value

        @model_validator(mode="after")
        def _template_matches_data(self) -> "StringTemplateBlock.Params":
            # The engine reruns this validator after resolving selectors.
            # render_template also checks direct callers outside the engine.
            if not re.fullmatch(DATA_SELECTOR_PATTERN, self.template):
                require_declared_placeholders(
                    self.template, variable_names=frozenset(self.data)
                )

            return self

    def __init__(self) -> None:
        self._compiled: Optional[Tuple[Any, Dict[str, CompiledOperations]]] = None

    def run(
        self,
        *,
        template: str,
        data: Dict[str, Any],
        data_operations: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Transform the variables and render the template.

        Args:
            template: Resolved template text.
            data: Resolved variables by name.
            data_operations: Validated operations by variable name.

        Returns:
            ``output``: the rendered string.

        Raises:
            ValueError: On an invalid template, a placeholder without a
                variable, a format spec not matching its value, or a resource
                limit violation.
            QueryRuntimeError: When an operation cannot process its value.
        """
        cached = self._compiled
        key = (frozenset(data), data_operations)
        if cached is None or cached[0] != key:
            operations = compile_data_operations(
                data_operations, variable_names=frozenset(data)
            )
            cached = (key, operations)
            self._compiled = cached
        variables = dict(data)
        for name, chain in cached[1].items():
            variables[name] = chain.apply(variables[name], parameters={})
        rendered = render_template(template, variables=variables)

        return {"output": rendered}


def compile_data_operations(
    data_operations: Dict[str, List[Any]], *, variable_names: FrozenSet[str]
) -> Dict[str, CompiledOperations]:
    """Compile per-variable operation chains that name no runtime values.

    Args:
        data_operations: Operations by variable name.
        variable_names: Keys of ``data``.

    Returns:
        Compiled chains by variable name.

    Raises:
        ValueError: When a key is not a ``data`` key.
        QueryDefinitionError: On an invalid chain.
    """
    compiled = {}
    for name, operations in data_operations.items():
        if name not in variable_names:
            raise ValueError(
                f"{BLOCK_TYPE}: data_operations names `{name}`, which is not a "
                f"data variable (data: {sorted(variable_names)})"
            )
        compiled[name] = compile_operations(
            operations, path=f"$.data_operations.{name}", declared_names=frozenset()
        )

    return compiled


def render_template(template: str, *, variables: Dict[str, Any]) -> str:
    """Render ``template`` with ``variables`` under V1's limits.

    The placeholder check also protects direct callers; a workflow checks
    resolved templates in ``Params._template_matches_data`` first.

    Args:
        template: Template text.
        variables: Values by placeholder name.

    Returns:
        The rendered string.

    Raises:
        ValueError: On an invalid template, an undeclared placeholder, a
            format spec not matching its value, or a resource limit violation.
    """
    require_declared_placeholders(template, variable_names=frozenset(variables))
    try:
        rendered = _LimitedFormatter().vformat(template, (), variables)
    except _LimitError as error:
        raise ValueError(str(error)) from None
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"{BLOCK_TYPE} could not render the template: {error}. Check that "
            f"format specs match the types of the substituted values."
        ) from error
    if len(rendered) > MAX_RENDERED_TEMPLATE_LENGTH:
        raise ValueError(
            f"{BLOCK_TYPE} rendered {len(rendered)} characters, more than the "
            f"maximum of {MAX_RENDERED_TEMPLATE_LENGTH}. Reduce the substituted "
            f"values or the number of placeholders."
        )

    return rendered


def require_declared_placeholders(
    template: str, *, variable_names: FrozenSet[str]
) -> None:
    """Check that every placeholder of ``template`` is a declared variable.

    Args:
        template: Template text.
        variable_names: Names the template may use.

    Raises:
        ValueError: On unparsable text, a positional, attribute or index
            placeholder, or a placeholder that is not a declared variable.
    """
    missing = _placeholders(template) - variable_names
    if missing:
        raise ValueError(
            f"{BLOCK_TYPE}: the template uses {sorted(missing)}, which are not "
            f"data variables (data: {sorted(variable_names)})"
        )


def _placeholders(template: str) -> set:
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(
            f"{BLOCK_TYPE} could not parse the template: {error}. Double the "
            f"braces to produce literal ones."
        ) from error
    names = set()
    for _, field_name, format_spec, _ in parsed:
        if field_name is None:
            continue

        if field_name == "" or field_name.isdigit():
            raise ValueError(
                f"{BLOCK_TYPE} does not support positional placeholders; use "
                f"named ones, e.g. `{{variable_name}}`."
            )
        if "." in field_name or "[" in field_name:
            raise ValueError(
                f"{BLOCK_TYPE} does not support attribute or index access in "
                f"placeholders (got `{{{field_name}}}`). Extract nested values "
                f"with data_operations."
            )
        names.add(field_name)
        # str.format resolves placeholders nested in a spec, e.g. {value:{width}}.
        if format_spec and "{" in format_spec:
            names.update(_placeholders(format_spec))

    return names


class _LimitError(ValueError):
    pass


class _LimitedFormatter(Formatter):
    """Reject format-spec widths and precisions that would allocate too much.

    ``format_field`` runs after nested spec placeholders were expanded, so a
    runtime width such as ``{value:{width}}`` is checked too.
    """

    def format_field(self, value: Any, format_spec: str) -> str:
        for digits in re.findall(r"\d+", format_spec):
            if int(digits) > MAX_FORMAT_SPEC_NUMBER:
                raise _LimitError(
                    f"{BLOCK_TYPE}: format spec `{format_spec}` uses {digits}, "
                    f"more than the maximum width or precision of "
                    f"{MAX_FORMAT_SPEC_NUMBER}."
                )

        formatted = super().format_field(value, format_spec)

        return formatted
