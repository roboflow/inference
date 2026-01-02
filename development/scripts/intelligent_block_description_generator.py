#!/usr/bin/env python3
"""
Intelligently generate block descriptions using the same logic as /updateblockdesc command.

This script analyzes each block and generates comprehensive descriptions following the template.
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def find_simple_format_blocks() -> List[Path]:
    """Find all blocks that need intelligent description generation."""
    from development.scripts.analyze_block_status import categorize_block
    
    core_steps_dir = project_root / "inference" / "core" / "workflows" / "core_steps"
    blocks_to_update = []
    
    for py_file in core_steps_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            if 'LONG_DESCRIPTION = """' in content or 'LONG_DESCRIPTION = f"""' in content:
                category, _ = categorize_block(content, py_file)
                # Update simple format blocks or any that don't have "How This Block Works"
                if category == "simple_format" or (category != "already_updated" and "## How This Block Works" not in content):
                    blocks_to_update.append(py_file)
        except Exception:
            pass
    
    return sorted(blocks_to_update)


def extract_block_metadata(content: str, file_path: Path) -> dict:
    """Extract metadata about the block."""
    metadata = {
        "name": file_path.stem.replace("_", " ").title(),
        "block_type": "unknown",
        "version": "v1",
        "short_description": "",
        "existing_long_description": "",
        "inputs": [],
        "outputs": [],
        "fields": [],
    }
    
    # Extract block name
    name_match = re.search(r'"name":\s*"([^"]+)"', content)
    if name_match:
        metadata["name"] = name_match.group(1)
    
    # Extract version
    version_match = re.search(r'"version":\s*"([^"]+)"', content)
    if version_match:
        metadata["version"] = version_match.group(1)
    
    # Extract block type
    block_type_match = re.search(r'"block_type":\s*"([^"]+)"', content)
    if block_type_match:
        metadata["block_type"] = block_type_match.group(1)
    
    # Extract short description
    short_desc_match = re.search(r'"short_description":\s*"([^"]+)"', content)
    if short_desc_match:
        metadata["short_description"] = short_desc_match.group(1)
    
    # Extract existing long description
    desc_match = re.search(r'LONG_DESCRIPTION = (?:f)?"""(.*?)"""', content, re.DOTALL)
    if desc_match:
        metadata["existing_long_description"] = desc_match.group(1).strip()
    
    # Try to extract outputs from describe_outputs method
    outputs_match = re.search(r'describe_outputs\(\)[^:]*:\s*return\s*\[(.*?)\]', content, re.DOTALL)
    if outputs_match:
        outputs_content = outputs_match.group(1)
        output_defs = re.findall(r'OutputDefinition\([^)]*name=["\']([^"\']+)["\']', outputs_content)
        metadata["outputs"] = output_defs
    
    return metadata


def generate_description(metadata: dict) -> str:
    """
    Generate a comprehensive description following the /updateblockdesc template.
    
    This is a simplified version - in practice, you'd want more sophisticated
    content generation based on block type, purpose, etc.
    """
    lines = []
    name = metadata["name"]
    block_type = metadata["block_type"]
    existing_desc = metadata["existing_long_description"]
    short_desc = metadata["short_description"]
    
    # Use short description or existing description as base for intro
    intro = short_desc if short_desc else existing_desc.split('\n')[0] if existing_desc else f"{name} block."
    if not intro.endswith('.'):
        intro += '.'
    
    lines.append(intro)
    lines.append("")
    lines.append("## How This Block Works")
    lines.append("")
    
    # Use existing description content if it's good, otherwise create basic explanation
    if existing_desc and len(existing_desc) > 50 and "##" not in existing_desc:
        # Clean up and use existing content
        existing_clean = existing_desc.strip()
        existing_clean = re.sub(r'\n{3,}', '\n\n', existing_clean)
        lines.append(existing_clean)
    else:
        # Generate basic explanation based on block type and name
        explanation = generate_basic_explanation(metadata)
        lines.append(explanation)
    
    # Add Common Use Cases
    lines.append("")
    lines.append("## Common Use Cases")
    lines.append("")
    use_cases = generate_use_cases(metadata)
    for use_case in use_cases:
        lines.append(f"- **{use_case['category']}**: {use_case['description']}")
    
    # Add Connecting to Other Blocks
    lines.append("")
    lines.append("## Connecting to Other Blocks")
    lines.append("")
    connections = generate_connections(metadata)
    lines.append(f"The {block_type} results from this block can be connected to:")
    lines.append("")
    for connection in connections:
        lines.append(f"- **{connection['type']}** blocks to {connection['purpose']}")
    
    return "\n".join(lines) + "\n"


def generate_basic_explanation(metadata: dict) -> str:
    """Generate a basic explanation based on metadata."""
    name = metadata["name"].lower()
    block_type = metadata["block_type"]
    
    # Try to infer purpose from name and type
    if "visualization" in name or block_type == "visualization":
        return f"This block takes input data and visualizes it on images, drawing annotations, shapes, or overlays based on the provided data."
    elif "filter" in name or "filter" in block_type:
        return f"This block filters input data based on specified criteria, returning only items that meet the conditions."
    elif "transformation" in block_type or "transform" in name:
        return f"This block transforms or processes input data, modifying it in some way before passing it to the next step."
    elif "formatter" in block_type or "format" in name or "parser" in name:
        return f"This block formats or parses input data, converting it from one format to another or extracting specific information."
    elif "analytics" in block_type:
        return f"This block performs analytics or calculations on input data, generating metrics, counts, or measurements."
    elif "sink" in block_type or "notification" in name:
        return f"This block outputs data to external systems, sends notifications, or stores results."
    else:
        return f"This block processes input data and produces output based on its configuration."


def generate_use_cases(metadata: dict) -> List[dict]:
    """Generate use cases based on block metadata."""
    name = metadata["name"].lower()
    block_type = metadata["block_type"]
    use_cases = []
    
    # Generate generic use cases based on block type
    if block_type == "visualization":
        use_cases.extend([
            {"category": "Annotation", "description": "Draw annotations on images to highlight detected objects or regions"},
            {"category": "Debugging", "description": "Visualize workflow results for inspection and debugging"},
            {"category": "Reporting", "description": "Generate annotated images for reports or documentation"},
        ])
    elif block_type == "analytics":
        use_cases.extend([
            {"category": "Metrics", "description": "Calculate metrics and statistics from workflow data"},
            {"category": "Tracking", "description": "Track objects, events, or patterns over time"},
        ])
    elif "filter" in name or "filter" in block_type:
        use_cases.extend([
            {"category": "Data Quality", "description": "Filter out low-quality or unwanted data"},
            {"category": "Conditional Processing", "description": "Route data based on specific conditions"},
        ])
    else:
        use_cases.append({
            "category": "Workflow Processing",
            "description": f"Process data in workflows to achieve specific goals"
        })
    
    return use_cases[:4]  # Limit to 4 use cases


def generate_connections(metadata: dict) -> List[dict]:
    """Generate connection suggestions based on block type."""
    block_type = metadata["block_type"]
    outputs = metadata["outputs"]
    connections = []
    
    if block_type == "visualization":
        connections.append({"type": "Data storage", "purpose": "save annotated images"})
        connections.append({"type": "Sink", "purpose": "output visualized results"})
    elif block_type == "analytics":
        connections.append({"type": "Filter", "purpose": "filter based on calculated metrics"})
        connections.append({"type": "Conditional logic", "purpose": "route workflow execution based on analytics results"})
    elif "filter" in block_type:
        connections.append({"type": "Downstream processing", "purpose": "continue workflow with filtered data"})
    elif "model" in block_type:
        connections.append({"type": "Visualization", "purpose": "visualize model predictions"})
        connections.append({"type": "Filter", "purpose": "filter predictions based on criteria"})
    else:
        connections.append({"type": "Other workflow blocks", "purpose": "continue processing in your workflow"})
    
    return connections[:4]  # Limit connections


def update_file(file_path: Path, dry_run: bool = False) -> bool:
    """Update a single file's LONG_DESCRIPTION."""
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Extract metadata
        metadata = extract_block_metadata(content, file_path)
        
        # Generate new description
        new_description = generate_description(metadata)
        
        # Extract LONG_DESCRIPTION location
        desc_match, prefix, suffix = extract_long_description(content)
        if not desc_match:
            print(f"  ⚠️  Could not extract LONG_DESCRIPTION from {file_path.name}")
            return False
        
        if dry_run:
            print(f"  🔍 {file_path.name} - WOULD UPDATE")
            print(f"     {metadata['name']} ({metadata['block_type']})")
            return True
        
        # Reconstruct file
        is_fstring = 'f"""' in desc_match
        quote_type = 'f"""' if is_fstring else '"""'
        new_desc_match = f'LONG_DESCRIPTION = {quote_type}\n{new_description}{quote_type}'
        new_content = prefix + new_desc_match + suffix
        
        file_path.write_text(new_content, encoding="utf-8")
        print(f"  ✓  Updated {file_path.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Intelligently generate block descriptions using /updateblockdesc logic"
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--file", type=str, help="Process single file")
    
    args = parser.parse_args()
    
    if args.file:
        files = [project_root / args.file]
    else:
        files = find_simple_format_blocks()
    
    print(f"Found {len(files)} blocks to update\n")
    
    updated = 0
    for file_path in files:
        print(f"Processing: {file_path.relative_to(project_root)}")
        if update_file(file_path, dry_run=args.dry_run):
            updated += 1
        print()
    
    print(f"Summary: Updated {updated}/{len(files)} blocks")


if __name__ == "__main__":
    main()

