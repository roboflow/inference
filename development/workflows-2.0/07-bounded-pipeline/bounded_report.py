"""One self-contained ``index.html`` from the examples' JSON evidence.

A probe timeline is drawn as a grid: one row per call, one column per event
in logical order (not time); a filled cell means the call was inside.
"""

import html
import json
from pathlib import Path
from typing import Any, Dict, List

STYLE = """
body { font-family: sans-serif; margin: 2em; max-width: 72em; }
table { border-collapse: collapse; margin: 0.5em 0 1.5em; font-size: 13px; }
td, th { border: 1px solid #ccc; padding: 2px 6px; text-align: left; }
td.in { background: #4a7bd0; }
td.grid { width: 10px; padding: 0; }
pre { background: #f4f4f4; padding: 0.5em; overflow-x: auto; font-size: 12px; }
.fail { color: #b00; }
"""


def write_report(evidence: Dict[str, Any], *, destination: Path) -> Path:
    """Write the report page.

    Args:
        evidence: Evidence per example name, as written to ``evidence.json``.
        destination: Directory; created if missing.

    Returns:
        Path of ``index.html``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    sections = [f"<h1>Bounded pipeline examples</h1>{_legend()}"]
    for name, item in evidence.items():
        sections.append(f"<h2>{html.escape(name)}</h2>")
        sections.append(_section(item))
    page = destination / "index.html"
    page.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Bounded pipeline examples</title><style>{STYLE}</style></head>"
        f"<body>{''.join(sections)}</body></html>\n"
    )

    return page


def _legend() -> str:
    return (
        "<p>Scheduling examples use a SYNTHETIC event-driven workload: every "
        "overlap shown is forced by probe events and ordered logically, not "
        "timed. Model examples use the trained ResNet-18; their seconds are "
        "single observations on this host, not performance claims, and say "
        "nothing about CUDA.</p>"
    )


def _section(item: Any) -> str:
    if not isinstance(item, dict):
        return f"<pre>{html.escape(json.dumps(item, indent=2, default=str))}</pre>"
    if "failed" in item:
        return f"<p class='fail'>FAILED: {html.escape(str(item['failed']))}</p>"

    parts: List[str] = []
    rest: Dict[str, Any] = {}
    for key, value in item.items():
        if key == "timeline":
            parts.append(_timeline(value))
        elif isinstance(value, dict) and "timeline" in value:
            parts.append(f"<h3>{html.escape(key)}</h3>{_section(value)}")
        else:
            rest[key] = value
    if rest:
        parts.append(
            f"<pre>{html.escape(json.dumps(rest, indent=2, default=str))}</pre>"
        )

    return "".join(parts)


def _timeline(records: List[Dict[str, Any]]) -> str:
    calls: List[str] = []
    for record in records:
        if record["call"] not in calls:
            calls.append(record["call"])
    inside = {call: False for call in calls}
    columns = []
    for record in records:
        inside[record["call"]] = record["event"] == "enter"
        columns.append(dict(inside))
        if record["event"] == "leave":
            columns[-1][record["call"]] = True

    rows = [
        "<tr><th>call</th>"
        + "".join(f"<th class='grid'>{record['order']}</th>" for record in records)
        + "</tr>"
    ]
    for call in calls:
        cells = "".join(
            f"<td class='grid{' in' if column[call] else ''}'></td>"
            for column in columns
        )
        rows.append(f"<tr><td>{html.escape(call)}</td>{cells}</tr>")

    table = f"<table>{''.join(rows)}</table>"

    return table
