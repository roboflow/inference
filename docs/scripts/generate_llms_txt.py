"""Generate llms.txt, llms-full.txt, and per-page .md files for the built docs site.

Run after `zensical build` to make documentation LLM-friendly.
Parses the nav structure from mkdocs.yml automatically — no manual file lists needed.
New docs added to mkdocs.yml are picked up automatically.

Directory-based nav entries (e.g. "workflows/blocks/") are included in llms.txt
as links but excluded from llms-full.txt to keep the full dump focused on
curated documentation.

Usage:
    python docs/scripts/generate_llms_txt.py
"""

import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = REPO_ROOT / "site"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

MARKDOWN_DESCRIPTION = (
    "Roboflow Inference is a high-performance computer vision inference server. "
    "Deploy object detection, classification, segmentation, and foundation models "
    "on-device or in the cloud."
)


class _PermissiveLoader(yaml.SafeLoader):
    """YAML loader that ignores unknown tags (e.g. !!python/name)."""

_PermissiveLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, suffix, node: None,
)


def load_config() -> dict:
    with open(MKDOCS_YML) as f:
        return yaml.load(f, Loader=_PermissiveLoader)


def _expand_dir(docs_dir: Path, dir_path: str) -> list[tuple[str, str, bool]]:
    """Expand a directory nav entry into (title, src_path, from_dir) tuples."""
    full = docs_dir / dir_path
    if not full.is_dir():
        return []
    return [
        ("", str(md.relative_to(docs_dir)), True)
        for md in sorted(full.rglob("*.md"))
    ]


def extract_nav_pages(nav_item, docs_dir: Path) -> list[tuple[str, str, bool]]:
    """Recursively extract (title, src_path, from_dir) tuples from a nav item.

    from_dir=True for pages discovered via directory expansion.
    """
    pages = []

    if isinstance(nav_item, str):
        if nav_item.endswith(".md"):
            pages.append(("", nav_item, False))
        elif nav_item.endswith("/"):
            pages.extend(_expand_dir(docs_dir, nav_item))

    elif isinstance(nav_item, dict):
        for title, value in nav_item.items():
            if isinstance(value, str):
                if value.startswith(("http://", "https://")):
                    continue
                if value.endswith(".md"):
                    pages.append((title, value, False))
                elif value.endswith("/"):
                    pages.extend(_expand_dir(docs_dir, value))
            elif isinstance(value, list):
                for child in value:
                    pages.extend(extract_nav_pages(child, docs_dir))

    return pages


def parse_nav(config: dict, docs_dir: Path) -> list[tuple[str, list[tuple[str, str, bool]]]]:
    """Parse the top-level nav into sections.

    Returns list of (section_name, [(title, src_path, from_dir), ...]).
    """
    nav = config.get("nav", [])
    sections = []

    for item in nav:
        if isinstance(item, dict):
            for section_name, children in item.items():
                pages = []
                if isinstance(children, list):
                    for child in children:
                        pages.extend(extract_nav_pages(child, docs_dir))
                elif isinstance(children, str):
                    pages.extend(extract_nav_pages(item, docs_dir))
                if pages:
                    sections.append((section_name, pages))

    return sections


def src_to_site_path(src_path: str) -> Path:
    """Convert docs source path to site directory path.

    'start/overview.md' -> 'start/overview/index.md'
    'install/cloud/index.md' -> 'install/cloud/index.md'
    """
    p = Path(src_path)
    if p.stem == "index":
        return p
    return p.parent / p.stem / "index.md"


def src_to_url(src_path: str, site_url: str) -> str:
    return f"{site_url.rstrip('/')}/{src_to_site_path(src_path)}"


def extract_title(md_content: str) -> str:
    for line in md_content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def process_pages(
    docs_dir: Path, site_dir: Path, sections: list, site_url: str
) -> list[tuple[str, list[dict]]]:
    """Copy .md files into site/ and build page metadata per section."""
    result = []
    seen = set()

    for section_name, nav_pages in sections:
        section_data = []
        for nav_title, src_path, from_dir in nav_pages:
            if src_path in seen:
                continue
            seen.add(src_path)

            full_src = docs_dir / src_path
            if not full_src.exists():
                continue

            content = full_src.read_text(encoding="utf-8")
            title = nav_title or extract_title(content) or Path(src_path).stem.replace("_", " ").title()

            # Copy .md file into site/
            site_path = src_to_site_path(src_path)
            dest = site_dir / site_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full_src, dest)

            section_data.append({
                "title": title,
                "content": content,
                "url": src_to_url(src_path, site_url),
                "from_dir": from_dir,
            })

        if section_data:
            result.append((section_name, section_data))

    return result


def generate_llms_txt(
    site_dir: Path, site_name: str, site_description: str, sections: list[tuple[str, list[dict]]]
) -> None:
    """Generate llms.txt — includes all pages (explicit + directory-expanded)."""
    lines = [f"# {site_name}", "", f"> {site_description}", "", MARKDOWN_DESCRIPTION, ""]
    for section_name, pages in sections:
        lines.append(f"## {section_name}")
        lines.append("")
        for p in pages:
            lines.append(f"- [{p['title']}]({p['url']})")
        lines.append("")
    (site_dir / "llms.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def generate_llms_full_txt(
    site_dir: Path, site_name: str, site_description: str, sections: list[tuple[str, list[dict]]]
) -> None:
    """Generate llms-full.txt — only includes explicitly listed pages, not directory-expanded."""
    lines = [f"# {site_name}", "", f"> {site_description}", "", MARKDOWN_DESCRIPTION, ""]
    for section_name, pages in sections:
        explicit_pages = [p for p in pages if not p["from_dir"]]
        if not explicit_pages:
            continue
        lines.append(f"# {section_name}")
        lines.append("")
        for p in explicit_pages:
            lines.append(f"## {p['title']}")
            lines.append("")
            lines.append(p["content"])
            lines.append("")
    (site_dir / "llms-full.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main():
    config = load_config()
    site_name = config.get("site_name", "Documentation")
    site_description = config.get("site_description", "")
    site_url = config.get("site_url", "")

    sections = parse_nav(config, DOCS_DIR)
    processed = process_pages(DOCS_DIR, SITE_DIR, sections, site_url)

    total_pages = sum(len(pages) for _, pages in processed)
    explicit_pages = sum(1 for _, pages in processed for p in pages if not p["from_dir"])
    dir_pages = total_pages - explicit_pages
    generate_llms_txt(SITE_DIR, site_name, site_description, processed)
    generate_llms_full_txt(SITE_DIR, site_name, site_description, processed)

    print(f"Generated llms.txt with {total_pages} pages across {len(processed)} sections")
    print(f"  - {explicit_pages} explicit pages (in llms.txt + llms-full.txt)")
    print(f"  - {dir_pages} directory-expanded pages (in llms.txt only)")
    print(f"Generated llms-full.txt ({(SITE_DIR / 'llms-full.txt').stat().st_size:,} bytes)")
    print(f"Copied {total_pages} .md files into site/")


if __name__ == "__main__":
    main()
