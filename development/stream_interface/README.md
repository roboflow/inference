# Stream Interface Benchmark Harness

## Manual-Only CI Execution

The stream pipeline benchmark harness tests (`test_benchmark_pipeline_extraction.py`) run in CI only via manual dispatch with the `run_stream_benchmark_checks` input set to `true`. This step runs harness correctness tests only, not the long Mac/L4 performance comparisons, which are separate manual runs on dedicated hardware.

**Temporary tooling**: The harness and its tests/fixtures will be removed after extraction verification is accepted.
