# Update Block Description

Enhance the LONG_DESCRIPTION for the current workflow block file following the standardized template format.

## Task

When this command is invoked, you should:

1. **Identify the block type**: Read the current file to understand what type of workflow block this is (e.g., object detection, classification, transformation, etc.)

2. **Analyze the current description**: Check if there's an existing LONG_DESCRIPTION and what sections it currently has.

3. **Generate/enhance the description** following this template structure:

   - **Brief introduction**: One-line summary of what the block does
   - **What is [Concept]?**: Beginner-friendly explanation of the core concept, what it does, how it differs from related concepts
   - **How This Block Works**: High-level explanation of the workflow - what inputs → processing → outputs
   - **Inputs and Outputs**: Clear documentation listing all inputs and outputs with descriptions
   - **Key Configuration Options**: Explanation of important parameters (extract from Field() definitions in the manifest)
   - **Common Use Cases**: 4-6 real-world examples of when to use this block
   - **Connecting to Other Blocks**: Suggestions for how this block connects to others in workflows
   - **Additional sections** as needed (Requirements, Model Sources, Version Differences for v2+, etc.)

4. **Update the LONG_DESCRIPTION** in the file, replacing the existing one with the enhanced version.

## Guidelines

- Use beginner-friendly language - assume the reader is new to computer vision/workflows
- Keep paragraphs short and focused
- Use bullet points and clear section headers
- Bold key terms for emphasis
- Include practical examples and use cases
- Reference actual field names and outputs from the block code

## Template Reference

The description should follow this structure:

```python
LONG_DESCRIPTION = """
[Brief one-line introduction]

## What is [Task/Concept Name]?

[Clear explanation for beginners - what is this concept, how does it differ from related tasks, what information does it provide]

## How This Block Works

[Step-by-step explanation: inputs → processing → outputs. Focus on "what" and "why", not deep technical implementation]

## Inputs and Outputs

**Input:**
- **[parameter_name]**: [Description]

**Output:**
- **[output_name]**: [Description]

## Key Configuration Options

- **[option_name]**: [Clear explanation of what it does, typical values, when to adjust]

## Common Use Cases

- **[Category]**: [Specific example]
- [More examples...]

## Connecting to Other Blocks

The [results] from this block can be connected to:
- **[Block Type]** blocks to [purpose]
- [More connections...]

[Additional sections as needed]
"""
```

## Important Notes

- Extract actual field names from the BlockManifest class
- Extract actual output names from the describe_outputs() method
- Use the block name and version to tailor the description
- For v2+ blocks, include a "Version Differences" section explaining what's different from previous versions
- Keep the description comprehensive but accessible
