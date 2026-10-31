"""Validate that generated documentation files do not contain Jinja2 syntax
errors.

mkdocs-macros processes all Markdown files through Jinja2 before rendering.
Patterns like ``{{ $parameters.xxx }}`` cause "unexpected char '$'" errors
because ``$`` is not valid inside Jinja2 expressions.  The doc build pipeline
(``build_block_docs.py``) escapes these patterns, but this script acts as a
safety net to catch any regressions.

Usage:
    python -m development.docs.validate_docs_jinja2 [docs_dir]

If no directory is given, it defaults to ``docs/``.  The script exits with
code 1 if any Jinja2 parse errors are found.
"""

import glob
import os
import sys

from jinja2 import Environment


def validate_markdown_files(docs_dir: str) -> list[tuple[str, str]]:
    """Parse every ``.md`` file under *docs_dir* as a Jinja2 template.

    Returns a list of ``(relative_path, error_message)`` tuples for files that
    fail to parse.
    """
    env = Environment()
    errors: list[tuple[str, str]] = []

    md_files = sorted(glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True))
    for md_file in md_files:
        with open(md_file, encoding="utf-8") as fh:
            content = fh.read()
        try:
            env.parse(content)
        except Exception as exc:
            rel_path = os.path.relpath(md_file, docs_dir)
            errors.append((rel_path, str(exc)))

    return errors


def main() -> None:
    if len(sys.argv) > 1:
        docs_dir = sys.argv[1]
    else:
        docs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "docs")

    docs_dir = os.path.abspath(docs_dir)
    if not os.path.isdir(docs_dir):
        print(f"Error: {docs_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    print(f"Validating Jinja2 syntax in {docs_dir} ...")
    errors = validate_markdown_files(docs_dir)

    if errors:
        print(f"\n{len(errors)} file(s) with Jinja2 syntax errors:\n", file=sys.stderr)
        for rel_path, err_msg in errors:
            print(f"  {rel_path}: {err_msg}", file=sys.stderr)
        print(
            "\nThese files will cause 'Macro Syntax Error' when rendered by "
            "mkdocs-macros.  Ensure that {{ ... }} expressions containing '$' "
            "are escaped (e.g. {{ '{{' }} $parameters.xxx {{ '}}' }}).",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        md_count = len(
            glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)
        )
        print(f"All {md_count} Markdown files pass Jinja2 syntax validation.")


if __name__ == "__main__":
    main()
