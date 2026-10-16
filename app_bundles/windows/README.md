# Windows

To build from scratch:

 - create venv with python 3.12
 - install inference with wanted optionals
 - run build.py
 

## Native operator packaging and validation

`install_dependencies.py --flavor cpu|gpu` replaces the build environment's
PyPI torch packages with an explicit Windows build pair, constrains the final
local-wheel installation, runs `pip check`, and executes real CPU NMS and PNG
encode/decode. `packaging_checks.py` owns the pair versions. CPU currently uses
2.14.1+cpu / 0.29.1+cpu; GPU uses 2.14.1+cu126 / 0.29.1+cu126 because the cu128
Windows index stops at torch 2.11, which conflicts with this repository's
setuptools floor. The CUDA compiler setup also targets 12.6. Model-specific
compatibility still requires qualification; wheel identity is not a guarantee.

The custom torchvision hook collects native binaries and their DLL dependencies
at their package-relative locations. This covers legacy `_C` / `image` and newer
`_C_stable` / `image_stable` layouts without importing torchvision during analysis.

Both production workflows validate the frozen executable before signing/upload.
`validate-windows-bundle.yml` builds both variants on Windows CPU runners without
signing credentials, release publication, or installer artifact upload. It tests
the production dist directory, then silently installs the unsigned installer and
tests the installed executable. The smoke test strips build/Python/CUDA PATH
entries and inherited credentials, uses isolated caches and offline mode, executes
bundled NMS and image decoding, and starts the actual server twice. It runs a local
image resize workflow in tensor mode and records cold/warm readiness and request
times. These are diagnostic timings, not a performance improvement claim.

Local regressions (no actual Windows execution):

```bash
uv run --no-project --python 3.12 --with pytest --with click --with pyinstaller --with pillow \
  python -m pytest -q -c app_bundles/windows/tests/pytest.ini app_bundles/windows/tests
```

On a built Windows executable:

```powershell
python app_bundles/windows/smoke_test.py --flavor cpu --exe app_bundles/windows/dist/inference/inference.exe
# On a Windows NVIDIA machine only:
.\inference.exe --packaging-smoke-test --flavor gpu --require-gpu
```

A CPU runner can certify CUDA wheel identity and CPU ops in a CUDA package, not
CUDA execution. NVIDIA qualification additionally requires a supported GPU/driver,
actual CUDA NMS, real representative model/workflow inference with GPU provider
verification, and an installed-executable restart. The model-free smoke does not
certify detection/anomaly model weights, anomaly heatmaps or external integrations.

For offline dependency installation, a future bundle would need a versioned,
hash-verified Windows/Python/flavor-specific wheelhouse and system prerequisites.
That is separate from model/workflow artifact provisioning, entitlement and leases,
telemetry and managed remote blocks. No offline installer bundle or truly offline
native deployment support is delivered by this repair.
