# OSX

## Recover a missing release DMG

`Build macOS Bundle` uses GitHub's ARM64 `macos-15` runner. To rebuild v1.8.6,
run the updated workflow from `main` (or the recovery PR branch while validating):

```bash
gh workflow run build-osx-bundle.yml --repo roboflow/inference --ref main \
  -f release_tag=v1.8.6
```

This checks out **the exact release tag for the application and local wheels**.
Only `build.py`, the torchvision packaging hook, and the DMG verification script
come from the selected workflow revision; both commit IDs appear in the run summary.
The build signs the app, notarizes it using the same explicit temporary keychain
that holds the notary profile, staples the ticket, and produces the DMG. The
verification step mounts that DMG read-only, verifies the app signature and ticket,
and requires the frozen server's `/readiness` endpoint to return HTTP 200.

By default, manual runs only retain an Actions artifact. After inspecting a
successful signed build, recover the missing asset on the existing release with:

```bash
gh workflow run build-osx-bundle.yml --repo roboflow/inference --ref main \
  -f release_tag=v1.8.6 -f publish_release=true
```

Publication requires an existing, published release and refuses to overwrite its
DMG. A failed signature, ticket, or startup check blocks release upload; the
Actions artifact is retained for diagnosis. Release-triggered builds still upload
automatically after these checks. `manual_version` remains an artifact-only label
for builds from the selected workflow ref and cannot be combined with `release_tag`.

The native build, signing, notarization, and mounted-DMG check require macOS and
the repository's Apple secrets. Linux checks cover recovery selection and failure
handling only. A successful build on macOS 15 does not establish compatibility
with older macOS versions.

## Local build

To build from scratch:

 - create venv with python 3.12
 - pip install -r requirements.txt
 - install inference in venv via  `pip install -e .[sam,transformers,clip,http,yolo-world,gaze,grounding-dino] ` in parent folder
 - for code signing, install developer ap signing cert in xcode keychain and create `roboflow-notary` notary profile (see github action for steps)
 - python build.py
 
 ```bash
uv venv --python 3.12
uv pip install pip
pip install -r requirements.txt
cd ../..
make create_wheels
WHEEL_FILE=$(ls dist/inference-*.whl | head -n 1)
pip install --find-links=./dist/ dist/roboflow_workflows-*.whl dist/streamvision-*.whl dist/inference_sdk-*.whl "$WHEEL_FILE[sam,transformers,clip,http,yolo-world,gaze,grounding-dino]"
cd app_bundles/osx
# skip code sign and notarize for local testing
python build.py --skip-sign
```

You can test the build by running the app bundle:
```bash
open dist/Roboflow\ Inference.app
```

For development, you can run the app bundle directly from the source code:
```bash
python run_inference.py
```




