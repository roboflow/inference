# OSX

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
pip install --find-links=./dist/ "$WHEEL_FILE[sam,transformers,clip,http,yolo-world,gaze,grounding-dino]"
cd app_bundles/osx
# skip code sign and notarize for local testing
python build.py --skip-sign --skip-notarize
```






