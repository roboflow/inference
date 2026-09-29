# Passive V2 examples

Run these examples from the repository root using the `roboflow-inference-new`
conda environment (NumPy, Pillow and click). Fixtures are generated locally;
the examples need no models, credentials, network, camera or GPU.

```bash
conda activate roboflow-inference-new
PYTHONPATH=workflows python development/workflows-2.0/01-passive-foundation/run_demo.py \
    --scenario all --output-dir /tmp/workflows-2.0-demo
```

Expect 45 `[PASS]` lines, no `[FAIL]`, and exit code 0. Run one example with
`--scenario <name>`. The examples import the V2 engine explicitly through
`PYTHONPATH=workflows`.

| Scenario | What it exercises | Outputs to inspect |
| --- | --- | --- |
| `nested` | Ragged crops `[2,0,1]`, per-crop inversion, per-image and all-image mosaics | `nested/result.json`, `nested/outputs/*.png` |
| `filtered` | Partial, empty and fully filtered groups using a `when` gate | `filtered/result.json`, `filtered/outputs/mosaic_*.png` |
| `invalid-bindings` | Invalid workflow definitions and incorrect block results | `invalid-bindings/cases.json` |
| `author-block` | Custom blocks at several nesting depths; engine results compared with direct calls | `author-block/comparison.json`, `author-block/result_*.json` |
| `metadata-cost` | Inherited versus per-leaf metadata measurements | `metadata-cost/metadata_cost.json` |

All paths above are relative to the chosen output directory. Each scenario also
writes `report.json`, which contains the demo's pass/fail checks. Engine outputs,
layouts, contexts, statuses and traces are in `result.json` or `result_*.json`.
PNG names carry logical indices: `crops_2_0.png` is crop 0 of image 2.

In `filtered`, image 1 has no crops and produces a blank mosaic. Image 2 has
crops removed by its gate, so its downstream mosaic is absent. This exposes the
difference between an empty group and a filtered group.

To change the examples, edit the [workflow definitions](workflows/) or inspect
[author_blocks.py](author_blocks.py) for custom block declarations.
[run_demo.py](run_demo.py) provides the CLI; [scenarios.py](scenarios.py)
assembles the cases and [fixtures.py](fixtures.py) creates their inputs.
