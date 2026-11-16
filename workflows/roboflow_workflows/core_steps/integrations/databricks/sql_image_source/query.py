"""The image reader's only SQL template and bounded identifier inputs."""

from typing import Dict, List, Optional, Tuple, Union


def quote_identifier(value: str) -> str:
    """Quote one identifier component, including embedded dots or backticks.

    Args:
        value (str): A single, unquoted catalog, schema, table or column name.

    Returns:
        A delimited identifier for a native IDENTIFIER parameter.

    Raises:
        ValueError: The name is empty, oversized or contains control characters.
    """
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 255
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(
            "Databricks identifiers must contain 1-255 printable characters"
        )

    quoted = "`" + value.replace("`", "``") + "`"

    return quoted


def build_image_query(
    *,
    catalog: str,
    schema: str,
    table: str,
    image_column: str,
    image_name_column: Optional[str],
    metadata_columns: List[str],
    max_rows: int,
) -> Tuple[str, Dict[str, Union[str, int]]]:
    """Build a fixed SELECT with native identifier and row-limit parameters.

    Args:
        catalog (str): Catalog name, without SQL quoting.
        schema (str): Schema name, without SQL quoting.
        table (str): Table or view name, without SQL quoting.
        image_column (str): Encoded image column name.
        image_name_column (Optional[str]): Optional name column.
        metadata_columns (List[str]): Explicit metadata column names, at most 32.
        max_rows (int): Positive row limit, at most 1000.

    Returns:
        The SQL template and its separate native parameters.

    Raises:
        ValueError: A name, list or row limit is invalid.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= 1000:
        raise ValueError("max_rows must be an integer between 1 and 1000")

    if not isinstance(metadata_columns, list) or len(metadata_columns) > 32:
        raise ValueError("metadata_columns must be a list of at most 32 names")

    source = ".".join(quote_identifier(part) for part in (catalog, schema, table))
    columns = [image_column]
    if image_name_column is not None:
        columns.append(image_name_column)
    columns.extend(metadata_columns)
    parameters: Dict[str, Union[str, int]] = {"source": source, "row_limit": max_rows}
    projection = []
    seen = set()
    for column in columns:
        quoted = quote_identifier(column)
        if quoted in seen:
            continue

        seen.add(quoted)
        parameter = f"column_{len(projection)}"
        parameters[parameter] = quoted
        projection.append(f"IDENTIFIER(:{parameter})")
    statement = (
        "SELECT " + ", ".join(projection) + " FROM IDENTIFIER(:source) LIMIT :row_limit"
    )

    return statement, parameters
