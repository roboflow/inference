# Batch Update Block Descriptions

This script automates the process of updating workflow block descriptions to match the new template format defined in `.cursor/commands/updateblockdesc.md`.

## What It Does

The script automatically:

1. ✅ Removes "What is [Concept]?" sections
2. ✅ Removes "Inputs and Outputs" sections  
3. ✅ Removes "Key Configuration Options" sections
4. ✅ Ensures blank lines before bullet lists (`:\n-` → `:\n\n-`)
5. ✅ Fixes grammar ("You will need" → "You need")
6. ✅ Preserves f-strings and dynamic content

## Important Notes

⚠️ **This script applies structural transformations only**. Some blocks may need manual review after automated updates, especially:
- Blocks with complex f-strings with variables
- Blocks with unique content structures
- Blocks that need content reorganization

## Analysis

You can analyze which blocks need updates first:

```bash
python3 development/scripts/analyze_block_status.py
```

This will show you:
- Blocks that need updates (have old format sections)
- Blocks already updated (match new template)
- Blocks with simple formats (may not need template)

## Usage

### Dry Run (Recommended First)

See what would be changed without modifying files:

```bash
python3 development/scripts/batch_update_block_descriptions.py --dry-run
```

### Update All Files

Update all workflow block files that need updates:

```bash
python3 development/scripts/batch_update_block_descriptions.py
```

**Note**: The script automatically skips blocks that already match the template or use simple formats.

### Update a Specific File

Update a single file for testing:

```bash
python3 development/scripts/batch_update_block_descriptions.py --file inference/core/workflows/core_steps/models/foundation/openai/v1.py
```

## Recommendations

1. **Start with a dry run** to see what will be changed
2. **Test on a few files first** using `--file` before running on all files
3. **Review changes** after the script runs - use git diff to see what changed
4. **Manual review** may still be needed for complex blocks (especially VLM blocks with f-strings)

## Current Status

Based on analysis of all 135 workflow blocks:
- **9 blocks need updates** (VLM blocks: anthropic_claude v1/v2, google_gemini v1/v2, openai v1-v4, florence2 v1)
- **10 blocks already updated** (Roboflow model blocks)
- **116 blocks use simple format** (will be skipped)

## Example Output

```
Found 135 workflow block files

Processing: inference/core/workflows/core_steps/models/foundation/openai/v1.py
  ✓  Updated openai/v1.py

Processing: inference/core/workflows/core_steps/models/roboflow/object_detection/v1.py
  ⊘  object_detection/v1.py - no updates needed (skipping)

...

============================================================
Summary:
  Updated: 9
  Skipped: 126
  Errors: 0
  Total: 135
```

