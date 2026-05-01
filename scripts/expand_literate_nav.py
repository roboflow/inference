"""Expand literate-nav SUMMARY.md references in mkdocs.yml.

Zensical does not run MkDocs plugins, so the mkdocs-literate-nav plugin
never processes SUMMARY.md files. This script runs as a pre-build step
and replaces directory references (e.g. ``Examples: workflows/gallery/``)
in the nav section of mkdocs.yml with the tree described by the
corresponding SUMMARY.md file.

The replacement is done via targeted text manipulation so that the rest
of the YAML file (comments, special tags, formatting) is preserved.
"""

import os
import re
import sys


def parse_summary_md(summary_path: str, dir_prefix: str) -> list[tuple[str, str | list]]:
    """Parse a literate-nav SUMMARY.md into a structured nav.

    Returns a list of (title, value) tuples where value is either a path
    string or a list of (title, path) tuples for nested sections.
    """
    with open(summary_path, encoding="utf-8") as f:
        lines = f.readlines()

    entries: list[tuple[str, str | list]] = []
    current_section: str | None = None
    current_children: list[tuple[str, str]] = []

    for line in lines:
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue

        indent = len(stripped) - len(stripped.lstrip())
        link_match = re.match(r"\s*\*\s+\[(.+?)\]\((.+?)\)", stripped)

        if indent == 0 and link_match:
            # Top-level link (flat list)
            if current_section:
                entries.append((current_section, current_children))
                current_section = None
                current_children = []
            title, url = link_match.group(1), link_match.group(2)
            entries.append((title, f"{dir_prefix}{url}"))

        elif indent == 0 and not link_match:
            # Section header
            if current_section:
                entries.append((current_section, current_children))
            section_match = re.match(r"\s*\*\s+(.+)", stripped)
            if section_match:
                current_section = section_match.group(1).strip()
                current_children = []

        elif indent > 0 and link_match:
            # Child link
            title, url = link_match.group(1), link_match.group(2)
            if current_section:
                current_children.append((title, f"{dir_prefix}{url}"))
            else:
                entries.append((title, f"{dir_prefix}{url}"))

    if current_section:
        entries.append((current_section, current_children))

    return entries


def _quote_if_needed(s: str) -> str:
    """Quote a YAML string if it contains special characters."""
    if any(c in s for c in ":#{}[]&*!|>'\",?@`"):
        return f'"{s}"'
    return s


def format_nav_yaml(entries: list, base_indent: str) -> str:
    """Format parsed SUMMARY entries as YAML nav lines."""
    lines = []
    for title, value in entries:
        if isinstance(value, str):
            lines.append(f"{base_indent}- {_quote_if_needed(title)}: {value}")
        elif isinstance(value, list):
            lines.append(f"{base_indent}- {_quote_if_needed(title)}:")
            for child_title, child_url in value:
                lines.append(f"{base_indent}    - {_quote_if_needed(child_title)}: {child_url}")
    return "\n".join(lines)


def expand_mkdocs_yml(config_path: str, docs_dir: str) -> None:
    """Find directory references in nav and expand them using SUMMARY.md."""
    with open(config_path, encoding="utf-8") as f:
        content = f.read()

    # Match nav entries like "      - Block Gallery: workflows/blocks/"
    # Captures: (leading spaces)(- )(Title)(: )(directory/path/)
    pattern = re.compile(
        r"^( +)(- )(.+?):\s+([\w/]+/)\s*$",
        re.MULTILINE,
    )

    replacements = []
    for match in pattern.finditer(content):
        leading_spaces = match.group(1)
        dash = match.group(2)
        title = match.group(3).strip().strip('"').strip("'")
        dir_path = match.group(4)

        summary_path = os.path.join(docs_dir, dir_path, "SUMMARY.md")
        if not os.path.exists(summary_path):
            continue

        entries = parse_summary_md(summary_path, dir_path)
        if not entries:
            continue

        # Children are indented one level deeper than the parent "- "
        child_indent = leading_spaces + "    "
        # Build the replacement: parent as a section, children expanded.
        # If an index.md exists, include it so the section title links to it.
        replacement_lines = [f"{leading_spaces}{dash}{_quote_if_needed(title)}:"]
        index_path = os.path.join(docs_dir, dir_path, "index.md")
        if os.path.exists(index_path):
            replacement_lines.append(f"{child_indent}- {_quote_if_needed(title)}: {dir_path}index.md")
        replacement_lines.append(format_nav_yaml(entries, child_indent))
        replacement = "\n".join(replacement_lines)

        replacements.append((match.start(), match.end(), replacement))
        print(f"  expanding '{title}' from {summary_path} ({len(entries)} entries)")

    if not replacements:
        print("No directory references found to expand.")
        return

    # Apply replacements in reverse order to preserve offsets
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + content[end:]

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "mkdocs.yml",
    )
    config_path = os.path.abspath(config_path)
    docs_dir = os.path.join(os.path.dirname(config_path), "docs")

    print(f"Expanding literate-nav SUMMARY.md references in {config_path} ...")
    expand_mkdocs_yml(config_path, docs_dir)
    print("Done.")


if __name__ == "__main__":
    main()
