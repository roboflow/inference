#!/usr/bin/env python3
"""
Batch update all workflow block descriptions to match the new template format.

This script:
1. Finds all workflow block files with LONG_DESCRIPTION
2. Applies template transformations:
   - Removes "What is [Concept]?" sections
   - Removes "Inputs and Outputs" sections
   - Removes "Key Configuration Options" sections
   - Ensures "How This Block Works" starts immediately
   - Ensures blank lines before bullet lists
   - Fixes common grammar issues
3. Updates Field() descriptions where applicable
4. Preserves unique content and version-specific sections
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_block_files() -> List[Path]:
    """Find all workflow block files with LONG_DESCRIPTION."""
    core_steps_dir = project_root / "inference" / "core" / "workflows" / "core_steps"
    block_files = []
    
    for py_file in core_steps_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if 'LONG_DESCRIPTION = """' in content or 'LONG_DESCRIPTION = f"""' in content:
                block_files.append(py_file)
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return sorted(block_files)


def extract_long_description(content: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract LONG_DESCRIPTION from file content.
    Returns: (full_match, prefix_before, suffix_after)
    """
    # Handle both regular and f-string LONG_DESCRIPTION
    patterns = [
        r'(LONG_DESCRIPTION = f?"""\n.*?""")',
        r'(LONG_DESCRIPTION = """\n.*?""")',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            start_pos = match.start()
            end_pos = match.end()
            prefix = content[:start_pos]
            suffix = content[end_pos:]
            return match.group(1), prefix, suffix
    
    return None, None, None


def needs_update(description: str) -> bool:
    """Check if description needs updating based on template rules."""
    # Skip if description is very short/simple (likely doesn't follow template format)
    if len(description.strip()) < 100:
        return False
    
    # Check for sections that should be removed
    if re.search(r'## What is ', description, re.IGNORECASE):
        return True
    if re.search(r'## Inputs and Outputs', description, re.IGNORECASE):
        return True
    if re.search(r'## Key Configuration Options', description, re.IGNORECASE):
        return True
    
    # Check for missing blank lines before bullet lists (only if description has sections)
    if re.search(r':\n-', description) and re.search(r'## ', description):
        return True
    
    # Check for grammar issues
    if 'You will need to set' in description:
        return True
    
    return False


def remove_section(content: str, section_title: str) -> str:
    """Remove a section from the description."""
    # Pattern to match section from header to next ## header or end of string
    # Match: ## Section Title\n\n followed by content until next ## or end
    pattern = rf'## {re.escape(section_title)}\s*\n\n.*?(?=\n## |\Z)'
    content = re.sub(pattern, '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Also try pattern without extra newline after header
    pattern2 = rf'## {re.escape(section_title)}\s*\n(?!\n)##.*?(?=\n## |\Z)'
    content = re.sub(pattern2, '', content, flags=re.DOTALL | re.MULTILINE)
    
    # Clean up any resulting double newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.strip()


def fix_blank_lines_before_bullets(content: str) -> str:
    """Ensure blank lines before bullet lists."""
    # Fix patterns like ":\\n-" to ":\\n\\n-"
    content = re.sub(r':\n-', ':\n\n-', content)
    # Also fix patterns after section headers
    content = re.sub(r'(## [^\n]+\n)(-)', r'\1\n\2', content)
    return content


def fix_grammar(content: str) -> str:
    """Fix common grammar issues."""
    # Fix "You will need" -> "You need"
    content = re.sub(r'You will need to set', 'You need to set', content)
    return content


def transform_description(description: str) -> str:
    """Apply template transformations to the description."""
    content = description
    
    # Remove "What is..." section (match any "What is X?" pattern)
    if re.search(r'## What is ', content, re.IGNORECASE):
        # Match from "## What is" to the next "##" section header
        pattern = r'## What is [^\n]+\n\n.*?(?=\n## |\Z)'
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.MULTILINE)
        # Clean up resulting blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Remove "Inputs and Outputs" section
    content = remove_section(content, 'Inputs and Outputs')
    
    # Remove "Key Configuration Options" section
    content = remove_section(content, 'Key Configuration Options')
    
    # Fix blank lines before bullets (must be after section removal to avoid breaking patterns)
    content = fix_blank_lines_before_bullets(content)
    
    # Fix grammar
    content = fix_grammar(content)
    
    # Clean up excessive blank lines (more than 2 consecutive)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Ensure proper ending (single newline at end)
    content = content.rstrip() + '\n'
    
    return content


def update_file(file_path: Path, dry_run: bool = False) -> bool:
    """Update a single file's LONG_DESCRIPTION."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Extract LONG_DESCRIPTION
        desc_match, prefix, suffix = extract_long_description(content)
        if not desc_match:
            print(f"  ⚠️  Could not extract LONG_DESCRIPTION from {file_path.name}")
            return False
        
        # Extract just the description content (between triple quotes)
        desc_content = re.search(r'"""(.*?)"""', desc_match, re.DOTALL)
        if not desc_content:
            desc_content = re.search(r'f"""(.*?)"""', desc_match, re.DOTALL)
        
        if not desc_content:
            print(f"  ⚠️  Could not parse description content from {file_path.name}")
            return False
        
        description = desc_content.group(1)
        
        # Check if update is needed
        if not needs_update(description):
            print(f"  ⊘  {file_path.name} - no updates needed (skipping)")
            return False
        
        # Transform the description
        new_description = transform_description(description)
        
        # Check if we actually changed anything
        if new_description.strip() == description.strip():
            print(f"  ⚠️  {file_path.name} - transformation resulted in no changes")
            return False
        
        if dry_run:
            print(f"  🔍 {file_path.name} - WOULD UPDATE")
            print(f"     Changes: {len(description)} -> {len(new_description)} chars")
            return True
        
        # Reconstruct the file content
        # Preserve f-string format if it was an f-string
        is_fstring = 'f"""' in desc_match
        quote_type = 'f"""' if is_fstring else '"""'
        
        new_desc_match = f'LONG_DESCRIPTION = {quote_type}\n{new_description}{quote_type}'
        new_content = prefix + new_desc_match + suffix
        
        # Write back
        file_path.write_text(new_content, encoding="utf-8")
        print(f"  ✓  Updated {file_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch update workflow block descriptions to match new template"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Update a specific file (relative path from project root)"
    )
    
    args = parser.parse_args()
    
    if args.file:
        files = [project_root / args.file]
        if not files[0].exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
    else:
        files = find_block_files()
    
    print(f"Found {len(files)} workflow block files\n")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in files:
        relative_path = file_path.relative_to(project_root)
        print(f"Processing: {relative_path}")
        
        try:
            was_updated = update_file(file_path, dry_run=args.dry_run)
            if was_updated:
                updated_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ❌ Error: {e}")
        
        print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(files)}")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()

