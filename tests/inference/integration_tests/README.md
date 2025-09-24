This directory is intended to run with pytest. The tests will run automatically via github actions.

To run files in this directory, the scripts need access to Roboflow API keys via env variables with the naming convention <PROJECT SLUG>_API_KEY=<ROBOFLOW API KEY>. For example, for a test with the project `asl-poly-instance-seg` to run succesfully, there needs to be an environment variable `asl_poly_instance_seg_API_KEY=<API KEY>`. If adding a test, be sure to add this environment variable to the github actions secrets via the github console and the file `.github/workflows/test.yml`.

The tests run using `tests.json`.  To add to this file, add an entry in the root list with all keys except for the `expected_response` key.  To add this key, run the `populate_expected_responses.py` script.