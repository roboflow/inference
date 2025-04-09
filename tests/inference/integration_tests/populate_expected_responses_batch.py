import json
import os
from pathlib import Path

import requests
from batch_regression_test import INFER_RESPONSE_FUNCTIONS
from PIL import Image

PORT = os.getenv("PORT", 9001)
BASE_URL = os.getenv("BASE_URL", "http://localhost")


def main():
    # Utility function to populate the expected responses for the tests. This likely shouldn't be run very often and should only be run when hosted inference is in working order.

    # Load tests.json
    with open(
        os.path.join(Path(__file__).resolve().parent, "batch_tests.json"), "r"
    ) as f:
        tests = json.load(f)

    # Iterate through list of tests
    for test in tests:
        if "expected_response" not in test:
            pil_image = Image.open(
                requests.get(test["image_url"], stream=True).raw
            ).convert("RGB")
            test["pil_image"] = pil_image
            api_key = os.getenv(test["project"].replace("-", "_") + "_API_KEY")

            test["expected_response"] = dict()
            for response_function in INFER_RESPONSE_FUNCTIONS:
                response, image_type = response_function(
                    test,
                    port=PORT,
                    api_key=api_key,
                    base_url=BASE_URL,
                    batch_size=test["batch_size"],
                )
                try:
                    response.raise_for_status()
                    test["expected_response"][image_type] = response.json()
                except Exception as e:
                    print(response.text)
                    # raise e
                    if "expected_response" in test:
                        del test["expected_response"]

            del test["pil_image"]

    # Save the response to a file
    with open(
        os.path.join(Path(__file__).resolve().parent, "batch_tests.json"), "w"
    ) as f:
        json.dump(tests, f, indent=4)


if __name__ == "__main__":
    main()
