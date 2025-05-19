# OSX

To build from scratch:

 - create venv with python 3.12
 - pip install -r requirements.txt
 - install inference in venv via  `pip install -e .[sam,transformers,clip,http,yolo-world,gaze,grounding-dino] ` in parent folder
 - for code signing, install developer ap signing cert in xcode keychain and create `roboflow-notary` notary profile (see github action for steps)
 - python build.py
 