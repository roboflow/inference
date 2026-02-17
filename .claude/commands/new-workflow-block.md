---
description: Spin up a team of agents to build a new workflow block for the inference project
argument-hint: <block description, e.g. "a rate limiter that throttles based on API key">
---

# New Workflow Block Builder

You are orchestrating a team of agents to build a new workflow block for the Roboflow inference project at /Users/jenniferkuchta/github/inference.

**Block request:** $ARGUMENTS

---

## Phase 1: Discovery

**Goal**: Understand what block needs to be built.

1. Parse the block request and identify:
   - **Block name** (snake_case, e.g. `event_store`, `rate_limiter`)
   - **Block category** (one of: analytics, cache, classical_cv, flow_control, formatters, fusion, math, models, sampling, sinks, trackers, transformations, visualizations)
   - **Block type** for manifest (e.g. `sink`, `transformation`, `analytics`, `model`, `fusion`)
   - **Key inputs** the block needs
   - **Key outputs** the block produces
   - **Whether it needs state** (stateful across calls vs stateless)
   - **Whether it needs file system access**
   - **Whether it needs external dependencies**

2. If the request is ambiguous, ask the user clarifying questions using AskUserQuestion before proceeding. Key questions to consider:
   - What data types should the inputs accept?
   - What should happen on error?
   - Does it need cooldown/throttling?
   - Does it need a disable flag?
   - Should it support batch inputs?

3. Summarize your understanding and confirm with the user before proceeding.

---

## Phase 2: Team Setup

**Goal**: Create the team and task list.

1. Create a team using TeamCreate with name `workflow-block-<block_name>`

2. Create tasks using TaskCreate:
   - **Task A**: "Research patterns for <block_name>" — Research relevant patterns and best practices
   - **Task B**: "Explore existing codebase patterns" — Explore similar blocks in the inference codebase
   - **Task C**: "Build <block_name> block" — Implement the block (manifest + class + supporting modules). Blocked by A, B.
   - **Task D**: "Register block in loader" — Add import and registration to loader.py. Blocked by C.
   - **Task E**: "Write tests for <block_name>" — Comprehensive test suite. Blocked by C.
   - **Task F**: "Write documentation for <block_name>" — LONG_DESCRIPTION and usage examples. Blocked by C.

3. Set up task dependencies with TaskUpdate (addBlockedBy).

---

## Phase 3: Research & Exploration (Parallel)

**Goal**: Gather all context needed for implementation.

Launch these agents in parallel using the Task tool:

### Agent 1: block-researcher
- **subagent_type**: `block-researcher`
- **Task**: Research best practices, similar implementations in other systems, and any libraries needed for the block
- Assign to Task A

### Agent 2: inference-workflow-architect
- **subagent_type**: `inference-workflow-architect`
- **Task**: Explore existing blocks in the inference project that are similar to what we're building. Read the base classes, 2-3 similar existing blocks, the loader, and the types/kinds system. Return a summary of patterns to follow and key file paths.
- Assign to Task B

Wait for both agents to complete and review their findings.

---

## Phase 4: Implementation

**Goal**: Build the block.

### Agent 3: block-builder
- **subagent_type**: `block-builder`
- **Task**: Implement the complete workflow block based on the research findings. This includes:
  - Creating the block directory and `__init__.py`
  - Creating any supporting modules (e.g., data stores, utilities)
  - Implementing `BlockManifest` with all inputs, outputs, and configuration
  - Implementing the `WorkflowBlock` class with the `run()` method
  - Following the exact patterns discovered in Phase 3
- Assign to Task C
- Provide the full block specification (name, type, inputs, outputs, behavior) and the research findings from Phase 3

Wait for the block-builder to complete.

---

## Phase 5: Registration, Testing & Documentation (Parallel)

**Goal**: Register the block, write tests, and document it.

Launch these agents in parallel:

### Agent 4: block-builder (or inference-workflow-architect)
- **subagent_type**: `inference-workflow-architect`
- **Task**: Register the new block in `inference/core/workflows/core_steps/loader.py`:
  - Add the import statement (alphabetically among similar blocks)
  - Add the block class to the `load_blocks()` list
  - Verify init parameter mappings if the block uses `get_init_parameters()`
- Assign to Task D

### Agent 5: block-tester
- **subagent_type**: `block-tester`
- **Task**: Write comprehensive tests and run them. Cover:
  - Supporting modules (if any)
  - BlockManifest validation
  - Block.run() with various inputs
  - Error handling
  - Any special features (cooldown, disable, file I/O, etc.)
- Assign to Task E

### Agent 6: block-documenter
- **subagent_type**: `block-documenter`
- **Task**: Write/enhance documentation:
  - Review and improve the LONG_DESCRIPTION in the block's v1.py
  - Ensure it follows the style of existing blocks
  - Add comprehensive usage examples and integration guidance
- Assign to Task F

Wait for all agents to complete.

---

## Phase 6: Verification & Cleanup

**Goal**: Verify everything works and clean up.

1. Check test results from block-tester — all tests must pass
2. If tests failed, send the failures back to block-builder or block-tester to fix
3. Verify the block is properly registered by checking the loader file
4. Shut down all agents with shutdown_request messages
5. Delete the team with TeamDelete

---

## Phase 7: Summary

**Goal**: Report what was built.

Present a summary to the user:

| | File | Purpose |
|---|---|---|
| **New** | `core_steps/<category>/<block_name>/__init__.py` | Package init |
| **New** | `core_steps/<category>/<block_name>/v1.py` | Block implementation |
| **New** | `core_steps/<category>/<block_name>/<supporting>.py` | Supporting modules (if any) |
| **New** | `tests/.../test_<block_name>.py` | Test suite |
| **Modified** | `core_steps/loader.py` | Block registration |

Include:
- Block type identifier (e.g., `roboflow_core/<block_name>@v1`)
- Input/output summary
- Test results (pass/fail count)
- Any notable design decisions

---

## Team Agent Reference

| Role | Agent Type | Tools | Purpose |
|------|-----------|-------|---------|
| Researcher | `block-researcher` | Read, Grep, Glob, WebSearch, WebFetch | Research patterns and best practices |
| Explorer | `inference-workflow-architect` | All | Explore existing codebase patterns |
| Builder | `block-builder` | Read, Write, Edit, Grep, Glob, Bash | Implement the block |
| Tester | `block-tester` | Read, Write, Edit, Grep, Glob, Bash | Write and run tests |
| Documenter | `block-documenter` | Read, Write, Edit, Grep, Glob, WebSearch | Write documentation |
| Registrar | `inference-workflow-architect` | All | Register block in loader |
