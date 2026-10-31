import glob
import os
import sys
from pathlib import Path


def get_version():
    """Read version from inference/core/version.py"""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / 'inference').is_dir():
            repo_root = parent
            break
    else:
        raise FileNotFoundError("Could not find repository root with 'inference' directory")

    version_file_path = repo_root.joinpath('inference', 'core', 'version.py')

    try:
        namespace = {}
        with open(version_file_path, 'r') as f:
            exec(f.read(), namespace)
        return namespace['__version__']
    except Exception as e:
        print(f"Warning: Could not read version from {version_file_path}: {e}")
        return "unknown"


def define_env(env):
    """Hook function to define macros for MkDocs."""
    env.macro(get_version)
    env.variables['VERSION'] = get_version()


def replace_in_docs(docs_dir: str) -> int:
    """Replace ``{{ VERSION }}`` in all Markdown files under *docs_dir*.

    Zensical does not run MkDocs plugins, so this can be called as a
    pre-build step to perform the substitution that mkdocs-macros would
    normally handle.

    Returns the number of files modified.
    """
    version = get_version()
    replacements = {
        "{{ VERSION }}": version,
    }

    modified = 0
    for md_file in sorted(
        glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)
    ):
        with open(md_file, encoding="utf-8") as fh:
            content = fh.read()

        new_content = content
        for placeholder, value in replacements.items():
            new_content = new_content.replace(placeholder, value)

        if new_content != content:
            with open(md_file, "w", encoding="utf-8") as fh:
                fh.write(new_content)
            modified += 1
            print(f"  replaced variables in {os.path.relpath(md_file, docs_dir)}")

    return modified


if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..",
    )
    docs_dir = os.path.abspath(docs_dir)

    version = get_version()
    print(f"Replacing template variables in {docs_dir} (VERSION={version}) ...")
    count = replace_in_docs(docs_dir)
    print(f"Done - {count} file(s) updated.")