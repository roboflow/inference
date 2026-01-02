#!/usr/bin/env python3
"""
Enhance simple format block descriptions to match the new template format.

This script takes blocks with simple descriptions and enhances them with the
template structure while preserving existing content.
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_simple_format_blocks() -> List[Path]:
    """Find all blocks with simple format descriptions."""
    from development.scripts.analyze_block_status import categorize_block
    
    core_steps_dir = project_root / "inference" / "core" / "workflows" / "core_steps"
    simple_blocks = []
    
    for py_file in core_steps_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if 'LONG_DESCRIPTION = """' in content or 'LONG_DESCRIPTION = f"""' in content:
                category, _ = categorize_block(content, py_file)
                if category == "simple_format":
                    simple_blocks.append(py_file)
        except Exception:
            pass
    
    return sorted(simple_blocks)


def extract_long_description(content: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract LONG_DESCRIPTION from file content."""
    patterns = [
        r'(LONG_DESCRIPTION = f"""\n.*?""")',
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


def extract_block_info(content: str, file_path: Path) -> dict:
    """Extract block information from the file."""
    info = {
        "name": file_path.stem,
        "block_type": "unknown",
        "has_how_it_works": False,
        "existing_content": "",
    }
    
    # Try to extract block name from manifest
    name_match = re.search(r'"name":\s*"([^"]+)"', content)
    if name_match:
        info["name"] = name_match.group(1)
    
    # Try to extract block type
    block_type_match = re.search(r'"block_type":\s*"([^"]+)"', content)
    if block_type_match:
        info["block_type"] = block_type_match.group(1)
    
    # Extract existing description
    desc_match = re.search(r'LONG_DESCRIPTION = (?:f)?"""(.*?)"""', content, re.DOTALL)
    if desc_match:
        info["existing_content"] = desc_match.group(1).strip()
        info["has_how_it_works"] = "## How This Block Works" in info["existing_content"]
    
    return info


def enhance_description(existing_content: str, block_info: dict) -> str:
    """Enhance a simple description with template structure."""
    content = existing_content.strip()
    
    # If it already has "How This Block Works", it might already be enhanced
    if "## How This Block Works" in content:
        # Just ensure proper formatting
        return ensure_template_formatting(content)
    
    # Build enhanced description
    lines = []
    
    # Try to extract a good intro line - look for first line or first sentence
    # But be careful not to break code references like `sv.RoundBoxAnnotator`
    first_line = content.split('\n')[0].strip()
    
    # If first line ends with period and is a complete sentence, use it
    # Otherwise, use the whole first line or paragraph
    if first_line.endswith('.'):
        intro_line = first_line
        # Get remaining content
        remaining_content = content[len(intro_line):].strip()
    else:
        # Use first line/paragraph as intro, but don't force a period
        intro_line = first_line
        if not intro_line.endswith('.'):
            # Try to find the first sentence by looking for period followed by space
            sentence_match = re.match(r'([^.]*\.)\s+', content)
            if sentence_match:
                intro_line = sentence_match.group(1)
                remaining_content = content[len(intro_line):].strip()
            else:
                # No clear sentence break, use first line as-is
                remaining_content = content[len(intro_line):].strip()
        else:
            remaining_content = content[len(intro_line):].strip()
    
    lines.append(intro_line)
    lines.append("")
    lines.append("## How This Block Works")
    lines.append("")
    
    # Use remaining content for "How This Block Works"
    if remaining_content:
        # Clean up the content
        remaining_content = re.sub(r'\n{3,}', '\n\n', remaining_content)
        lines.append(remaining_content)
    else:
        # If we used the whole content as intro, repeat it in "How This Block Works"
        lines.append(intro_line)
    
    # Add template sections (with placeholders that make sense)
    lines.append("")
    lines.append("## Common Use Cases")
    lines.append("")
    lines.append("- Use this block to [purpose based on block type]")
    
    # Add connecting to other blocks section
    lines.append("")
    lines.append("## Connecting to Other Blocks")
    lines.append("")
    lines.append("The outputs from this block can be connected to other blocks in your workflow.")
    
    return "\n".join(lines) + "\n"


def ensure_template_formatting(content: str) -> str:
    """Ensure existing content follows template formatting rules."""
    # Fix blank lines before bullets
    content = re.sub(r':\n-', ':\n\n-', content)
    content = re.sub(r'(## [^\n]+\n)(-)', r'\1\n\2', content)
    
    # Fix grammar
    content = re.sub(r'You will need to set', 'You need to set', content)
    
    # Clean up excessive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    return content.rstrip() + '\n'


def update_file(file_path: Path, dry_run: bool = False) -> bool:
    """Update a single file's LONG_DESCRIPTION."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Extract LONG_DESCRIPTION
        desc_match, prefix, suffix = extract_long_description(content)
        if not desc_match:
            print(f"  ⚠️  Could not extract LONG_DESCRIPTION from {file_path.name}")
            return False
        
        # Extract description content
        desc_content_match = re.search(r'(?:f)?"""(.*?)"""', desc_match, re.DOTALL)
        if not desc_content_match:
            print(f"  ⚠️  Could not parse description content from {file_path.name}")
            return False
        
        existing_description = desc_content_match.group(1)
        block_info = extract_block_info(content, file_path)
        
        # Enhance the description
        enhanced_description = enhance_description(existing_description, block_info)
        
        # Check if we actually changed anything meaningful
        if enhanced_description.strip() == existing_description.strip():
            print(f"  ⊘  {file_path.name} - no significant changes needed")
            return False
        
        if dry_run:
            print(f"  🔍 {file_path.name} - WOULD ENHANCE")
            print(f"     {block_info['name']} ({block_info['block_type']})")
            print(f"     Changes: {len(existing_description)} -> {len(enhanced_description)} chars")
            return True
        
        # Reconstruct the file content
        is_fstring = 'f"""' in desc_match
        quote_type = 'f"""' if is_fstring else '"""'
        
        new_desc_match = f'LONG_DESCRIPTION = {quote_type}\n{enhanced_description}{quote_type}'
        new_content = prefix + new_desc_match + suffix
        
        # Write back
        file_path.write_text(new_content, encoding="utf-8")
        print(f"  ✓  Enhanced {file_path.name}")
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
        description="Enhance simple format block descriptions to match new template"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be enhanced without making changes"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Enhance a specific file (relative path from project root)"
    )
    
    args = parser.parse_args()
    
    if args.file:
        files = [project_root / args.file]
        if not files[0].exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
    else:
        files = find_simple_format_blocks()
    
    print(f"Found {len(files)} simple format blocks\n")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")
    
    enhanced_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in files:
        relative_path = file_path.relative_to(project_root)
        print(f"Processing: {relative_path}")
        
        try:
            was_enhanced = update_file(file_path, dry_run=args.dry_run)
            if was_enhanced:
                enhanced_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ❌ Error: {e}")
        
        print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Enhanced: {enhanced_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(files)}")
    
    if args.dry_run:
        print("\nRun without --dry-run to apply enhancements")
    else:
        print("\n⚠️  NOTE: Enhanced descriptions may need manual review and refinement")
        print("   Please review the changes and add specific use cases and connections.")


if __name__ == "__main__":
    # Import the categorize function from analyze script
    try:
        from development.scripts.analyze_block_status import categorize_block
    except ImportError:
        print("Error: Could not import categorize_block from analyze_block_status.py")
        sys.exit(1)
    
    main()

