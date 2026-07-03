# Profiling Snippet Extraction

Use this skill when the user asks to create an isolated profiling target from a
function, snippet, module, or line range.

## Workflow

1. Read the referenced production code and nearby imports, types, helpers, and
   call sites needed to understand inputs, outputs, side effects, and device
   behavior.
2. Create `inference_profiling/snippets/<profile-name>/target.py`.
3. Copy or adapt only the code needed for the experiment into `target.py`.
   Generated snippets are local experiments; do not edit production code unless
   the user separately asks for a production change.
4. Add NVTX ranges with `development.profiling.nvtx.profiling_range_if_cuda` or
   `profiling_range` around meaningful operations such as preprocessing, model
   execution, decoding, filtering, NMS, or postprocessing.
5. Implement a `ProfileTarget`-compatible class with:
   - `name`
   - `prepare(record, *, device)`
   - `run(prepared)`
   - `validate(output)`
   - `summarize(output)`
6. Expose a `target` object or zero-argument factory from `target.py`.
7. Create `inference_profiling/snippets/<profile-name>/config.yaml` with:
   - profile name
   - target name
   - `target.import_path`
   - data source settings
   - device, warmup, iterations, repetitions
   - capture range and CUDA synchronization settings when needed
8. Create a short README in the generated snippet directory that documents:
   - provenance of the copied code
   - assumptions and intentional differences from production
   - required local files, models, or environment variables
   - smoke command without Nsight
   - full `nsys profile` command

## Sanity Checks

When practical, compare the generated snippet output against the production
function on a tiny fixture. Preserve dtype and device behavior unless the
experiment is intentionally profiling an alternative implementation. Document any
known differences in the generated README.

## Boundaries

Generated snippets may import `development.profiling.*` helpers because they are
local development artifacts. Production code must not import these helpers.
