"""Formatter blocks: UQL properties and expressions, text, CSV, values and JSON.

``create_catalogue()`` registers every block in ``BLOCKS``. ``TIMESTAMP_KIND``
is the output kind of ``roboflow_core/current_time@v1``.
"""

from roboflow_workflows.v2.blocks.formatters.csv_formatter.v1 import CsvFormatterBlock
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.formatters.expression.v1 import ExpressionBlock
from roboflow_workflows.v2.blocks.formatters.property_definition.v1 import PropertyDefinitionBlock
from roboflow_workflows.v2.blocks.formatters.string_template.v1 import StringTemplateBlock
from roboflow_workflows.v2.blocks.formatters._shared.values import TIMESTAMP_KIND
from roboflow_workflows.v2.blocks.formatters.current_time.v1 import CurrentTimeBlock
from roboflow_workflows.v2.blocks.formatters.first_non_empty_or_default.v1 import FirstNonEmptyOrDefaultBlock

BLOCKS = (
    PropertyDefinitionBlock,
    ExpressionBlock,
    StringTemplateBlock,
    CsvFormatterBlock,
    FirstNonEmptyOrDefaultBlock,
    CurrentTimeBlock,
    JsonParserBlock,
)

__all__ = [
    "BLOCKS",
    "CsvFormatterBlock",
    "CurrentTimeBlock",
    "ExpressionBlock",
    "FirstNonEmptyOrDefaultBlock",
    "JsonParserBlock",
    "PropertyDefinitionBlock",
    "StringTemplateBlock",
    "TIMESTAMP_KIND",
]
