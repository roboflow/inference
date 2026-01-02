# How to Rebuild Workflow Block Documentation

After making changes to workflow block descriptions (LONG_DESCRIPTION), you need to rebuild the documentation.

## Quick Method (Recommended)

1. Open a terminal (Terminal.app, iTerm, VS Code integrated terminal, etc.)
2. Navigate to the project root directory:
   ```bash
   cd /path/to/inference
   ```
3. Run the script:
   ```bash
   ./rebuild_block_docs.sh
   ```

**Note:** The script works in any terminal (bash, zsh, etc.) on Mac/Linux. Make sure you're in the project root directory where the script is located.

## Manual Method

1. Make sure you're in the project root directory
2. Use the Python environment that has dependencies installed (usually your conda/pyenv inference environment)
3. Install jinja2 if needed: `pip install jinja2`
4. Run the build script:
   ```bash
   python -m development.docs.build_block_docs
   ```

## What Gets Rebuilt

The script regenerates all workflow block documentation files in `docs/workflows/blocks/` based on the current code.

## When to Rebuild

- After changing `LONG_DESCRIPTION` in any workflow block file
- After changing block metadata in `model_config.json_schema_extra`
- After adding/removing blocks
- Before committing documentation changes

## Notes

- The documentation is generated from code, not loaded dynamically
- Changes to code won't appear in docs until you rebuild
- If viewing docs locally (mkdocs), refresh your browser after rebuilding

