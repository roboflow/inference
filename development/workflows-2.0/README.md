# Workflows 2.0 development examples

Runnable examples for the explicitly selected V2 engine in
`workflows/roboflow_workflows/execution_engine/v2/`.

| Examples | What you can try |
| --- | --- |
| [Finite sources](04-source-lifecycle/README.md) | Independent image and CSV signal sources, timestamps, nested conditionals, output groups and stop/drain |
| [Passive foundation](01-passive-foundation/README.md) | Nested image crops, mosaics, a filtering gate, custom blocks, invalid definitions and metadata measurements |
| [Tensor-native media](03-tensor-native/README.md) | Nested crop/resize predictions, native family round-trips, sparse filtering, provenance and rendered own/root outputs |
| [Sequential parity](02-sequential-parity/README.md) | 45 cases run on the real V1 engine and on V2 with call-by-call comparison, plus 20 V2 capability examples |

Run commands from the repository root with
`PYTHONPATH=workflows:inference_models:stream_vision` and the existing
`roboflow-inference-new` environment. Each directory contains its code,
workflow definitions and usage instructions.
