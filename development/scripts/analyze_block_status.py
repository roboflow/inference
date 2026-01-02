#!/usr/bin/env python3
"""
Analyze which workflow blocks need description updates.

This script categorizes blocks based on whether they need updates or not.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def extract_long_description(content: str) -> str:
    """Extract LONG_DESCRIPTION content from file."""
    # Handle both regular and f-string LONG_DESCRIPTION
    patterns = [
        r'LONG_DESCRIPTION = f"""(.*?)"""',
        r'LONG_DESCRIPTION = """(.*?)"""',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1)
    
    return ""


def categorize_block(content: str, file_path: Path) -> Tuple[str, List[str]]:
    """Categorize a block based on its description format."""
    description = extract_long_description(content)
    issues = []
    category = "unknown"
    
    if not description:
        return "no_description", ["No LONG_DESCRIPTION found"]
    
    # Check length
    if len(description.strip()) < 100:
        return "simple_format", ["Very short description - may not need template format"]
    
    # Check for old format sections
    has_what_is = bool(re.search(r'## What is ', description, re.IGNORECASE))
    has_inputs_outputs = bool(re.search(r'## Inputs and Outputs', description, re.IGNORECASE))
    has_config_options = bool(re.search(r'## Key Configuration Options', description, re.IGNORECASE))
    has_how_it_works = bool(re.search(r'## How This Block Works', description, re.IGNORECASE))
    
    # Check for formatting issues
    missing_blank_line = bool(re.search(r':\n-', description))
    grammar_issue = 'You will need to set' in description
    
    # Categorize
    if has_what_is or has_inputs_outputs or has_config_options:
        category = "needs_update"
        if has_what_is:
            issues.append("Has 'What is...' section")
        if has_inputs_outputs:
            issues.append("Has 'Inputs and Outputs' section")
        if has_config_options:
            issues.append("Has 'Key Configuration Options' section")
        if missing_blank_line:
            issues.append("Missing blank lines before bullets")
        if grammar_issue:
            issues.append("Grammar issue: 'You will need'")
    elif has_how_it_works and not (has_what_is or has_inputs_outputs or has_config_options):
        category = "already_updated"
        if missing_blank_line:
            issues.append("Minor: Missing blank lines before bullets")
        if grammar_issue:
            issues.append("Minor: Grammar issue")
    else:
        category = "simple_format"
        if not has_how_it_works:
            issues.append("Doesn't follow template format (may be intentional)")
    
    # Check if f-string (dynamic content)
    if 'LONG_DESCRIPTION = f"""' in content:
        issues.append("Uses f-string (dynamic content)")
    
    return category, issues


def main():
    """Main function."""
    core_steps_dir = project_root / "inference" / "core" / "workflows" / "core_steps"
    block_files = []
    
    for py_file in core_steps_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if 'LONG_DESCRIPTION = """' in content or 'LONG_DESCRIPTION = f"""' in content:
                block_files.append(py_file)
        except Exception:
            pass
    
    categories = {
        "needs_update": [],
        "already_updated": [],
        "simple_format": [],
        "no_description": [],
        "unknown": [],
    }
    
    for file_path in sorted(block_files):
        try:
            content = file_path.read_text(encoding="utf-8")
            category, issues = categorize_block(content, file_path)
            relative_path = file_path.relative_to(project_root)
            categories[category].append((relative_path, issues))
        except Exception as e:
            categories["unknown"].append((file_path, [f"Error: {e}"]))
    
    # Print report
    print("=" * 80)
    print("WORKFLOW BLOCK DESCRIPTION STATUS REPORT")
    print("=" * 80)
    print()
    
    total = sum(len(files) for files in categories.values())
    
    print(f"📊 SUMMARY")
    print(f"  Total blocks: {total}")
    print(f"  Need updates: {len(categories['needs_update'])}")
    print(f"  Already updated: {len(categories['already_updated'])}")
    print(f"  Simple format: {len(categories['simple_format'])}")
    print(f"  No description: {len(categories['no_description'])}")
    print(f"  Unknown/Errors: {len(categories['unknown'])}")
    print()
    
    # Details for each category
    if categories["needs_update"]:
        print("=" * 80)
        print(f"🔴 NEEDS UPDATE ({len(categories['needs_update'])} blocks)")
        print("=" * 80)
        for file_path, issues in categories["needs_update"][:20]:  # Show first 20
            print(f"\n  {file_path}")
            for issue in issues:
                print(f"    - {issue}")
        if len(categories["needs_update"]) > 20:
            print(f"\n  ... and {len(categories['needs_update']) - 20} more")
        print()
    
    if categories["already_updated"]:
        print("=" * 80)
        print(f"✅ ALREADY UPDATED ({len(categories['already_updated'])} blocks)")
        print("=" * 80)
        for file_path, issues in categories["already_updated"][:10]:  # Show first 10
            print(f"  {file_path}")
            if issues:
                for issue in issues:
                    print(f"    - {issue}")
        if len(categories["already_updated"]) > 10:
            print(f"\n  ... and {len(categories['already_updated']) - 10} more")
        print()
    
    if categories["simple_format"]:
        print("=" * 80)
        print(f"ℹ️  SIMPLE FORMAT ({len(categories['simple_format'])} blocks)")
        print("=" * 80)
        print("  These blocks have simple descriptions that may not need template format")
        for file_path, issues in categories["simple_format"][:10]:
            print(f"  {file_path}")
        if len(categories["simple_format"]) > 10:
            print(f"  ... and {len(categories['simple_format']) - 10} more")
        print()
    
    if categories["no_description"]:
        print("=" * 80)
        print(f"⚠️  NO DESCRIPTION ({len(categories['no_description'])} blocks)")
        print("=" * 80)
        for file_path, issues in categories["no_description"]:
            print(f"  {file_path}")
        print()
    
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    if categories["needs_update"]:
        print(f"Run batch update script on {len(categories['needs_update'])} blocks that need updates.")
        print("Blocks already updated will be skipped automatically.")
    else:
        print("All blocks appear to be up to date or use simple formats.")


if __name__ == "__main__":
    main()

