# Update Block Description

Enhance the LONG_DESCRIPTION for the current workflow block file following the standardized template format.

## Task

When this command is invoked, you should:

1. **Identify the block type**: Read the current file to understand what type of workflow block this is (e.g., object detection, classification, transformation, etc.)

2. **Analyze the current description**: Check if there's an existing LONG_DESCRIPTION and what sections it currently has.

3. **Check Field() descriptions**: Review the Field() definitions in the BlockManifest class. If any Field() descriptions are missing or inadequate, update them in the model_config block manifest. Field() descriptions should contain input/output documentation and key configuration details.

4. **Generate/enhance the description** following this template structure:

   - **Brief introduction**: One-line summary of what the block does
   - **How This Block Works**: High-level explanation of the workflow - what inputs → processing → outputs (START HERE, no "What is" section)
   - **Common Use Cases**: 4-6 real-world examples of when to use this block
   - **Connecting to Other Blocks**: Suggestions for how this block connects to others in workflows
   - **Additional sections** as needed (Requirements, Model Sources, Version Differences for v2+, etc.)
   - **DO NOT include** "Inputs and Outputs" or "Key Configuration Options" sections in LONG_DESCRIPTION - these belong only in Field() descriptions in the manifest

5. **Update the LONG_DESCRIPTION** in the file, replacing the existing one with the enhanced version.

6. **Update Field() descriptions** in the BlockManifest class if they need improvement (add missing descriptions, clarify existing ones).

## Guidelines

- Use beginner-friendly language - assume the reader is new to computer vision/workflows
- Keep paragraphs short and focused
- Use bullet points and clear section headers
- **CRITICAL**: Always include a blank line (double newline) before bullet lists - markdown requires this for proper rendering
- Bold key terms for emphasis
- Include practical examples and use cases
- Ensure proper grammar and spelling throughout
- Do not duplicate information that belongs in Field() descriptions

## Template Reference

The description should follow this structure:

```python
LONG_DESCRIPTION = """
[Brief one-line introduction]

## How This Block Works

[Step-by-step explanation: inputs → processing → outputs. Focus on "what" and "why", not deep technical implementation. Mention key inputs/outputs in context, but detailed documentation belongs in Field() descriptions.]

## Common Use Cases

- **[Category]**: [Specific example]
- [More examples...]

## Connecting to Other Blocks

The [results] from this block can be connected to:

- **[Block Type]** blocks to [purpose]
- [More connections...]

[Additional sections as needed: Requirements, Model Sources, Version Differences for v2+, etc.]
"""
```

## Important Notes

- **NO "What is [Concept]?" section** - start directly with "How This Block Works"
- **NO "Inputs and Outputs" section** in LONG_DESCRIPTION - this information belongs in Field() descriptions in the BlockManifest class
- **NO "Key Configuration Options" section** in LONG_DESCRIPTION - this information belongs in Field() descriptions in the BlockManifest class
- Extract actual field names from the BlockManifest class for reference, but document them in Field() descriptions, not LONG_DESCRIPTION
- Extract actual output names from the describe_outputs() method for reference, but document them in Field() descriptions, not LONG_DESCRIPTION
- Use the block name and version to tailor the description
- For v2+ blocks, include a "Version Differences" section explaining what's different from previous versions
- **Always ensure blank lines before bullet lists** - use `:\n\n-` not `:\n-` for lists following text ending with colons
- Check grammar and spelling carefully
- Keep the description comprehensive but accessible
