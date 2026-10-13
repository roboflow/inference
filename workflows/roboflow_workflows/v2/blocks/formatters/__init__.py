"""Formatter blocks: UQL properties and expressions, text, CSV, values and JSON.

``create_catalogue()`` registers every block in ``BLOCKS``. ``TIMESTAMP_KIND``
is the output kind of ``v2/current_time@v1``.
"""

from roboflow_workflows.v2.blocks.formatters.csv import CsvFormatterBlock
from roboflow_workflows.v2.blocks.formatters.json import JsonParserBlock
from roboflow_workflows.v2.blocks.formatters.properties import (
    ExpressionBlock,
    PropertyDefinitionBlock,
)
from roboflow_workflows.v2.blocks.formatters.text import StringTemplateBlock
from roboflow_workflows.v2.blocks.formatters.values import (
    TIMESTAMP_KIND,
    CurrentTimeBlock,
    FirstNonEmptyOrDefaultBlock,
)

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
