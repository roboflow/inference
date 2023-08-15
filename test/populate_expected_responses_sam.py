import json
import os
import requests

from pathlib import Path
from PIL import Image

PORT = os.getenv("PORT", 9001)
BASE_URL = os.getenv("BASE_URL", "http://localhost")


def main():
    # Utility function to populate the expected responses for the tests. This likely shouldn't be run very often and should only be run when hosted inference is in working order.

    # Load tests.json
    with open(
        os.path.join(Path(__file__).resolve().parent, "sam_tests.json"), "r"
    ) as f:
        tests = json.load(f)

    # Iterate through list of tests
    for test in tests:
        if "expected_response" not in test:
            response = requests.post(
                f"{BASE_URL}:{PORT}/sam/{test['type']}",
                json=test["payload"],
            )
            try:
                response.raise_for_status()
            except Exception as e:
                print(response.text)
                print("PAYLOAD", test["payload"])
                raise e
            test["expected_response"] = response.json()

    # Save the response to a file
    with open(
        os.path.join(Path(__file__).resolve().parent, "sam_tests.json"), "w"
    ) as f:
        json.dump(tests, f)


if __name__ == "__main__":
    main()
