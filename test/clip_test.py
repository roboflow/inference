import json
import os
import requests
from copy import deepcopy
from pathlib import Path
import pytest
import time

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

with open(os.path.join(Path(__file__).resolve().parent, "clip_tests.json"), "r") as f:
    TESTS = json.load(f)


@pytest.mark.parametrize("test", TESTS)
def test_clip(test):
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key
    response = requests.post(
        f"{base_url}:{port}/clip/{test['type']}",
        json=payload,
    )
    try:
        response.raise_for_status()
        data = response.json()
        if test["type"] == "embed_image" or test["type"] == "embed_text":
            try:
                assert "embeddings" in data
            except:
                print(f"Invalid response: {data}, expected 'embeddings' in response")
            try:
                assert len(data["embeddings"]) == len(
                    test["expected_response"]["embeddings"]
                )
            except:
                print(
                    f"Invalid response: {data}, expected length of embeddings to be {len(test['expected_response']['embeddings'])}, got {len(data['embeddings'])}"
                )
        if test["type"] == "compare":
            try:
                assert "similarity" in data
            except:
                print(f"Invalid response: {data}, expected 'similarity' in response")
            try:
                assert data["similarity"] == test["expected_response"]["similarity"]
            except:
                print(
                    f"Invalid response: {data}, expected similarity to be {test['expected_response']['similarity']}, got {data['similarity']}"
                )
    except Exception as e:
        raise e


@pytest.fixture(scope="session", autouse=True)
def setup():
    try:
        res = requests.get(f"{base_url}:{port}")
        res.raise_for_status()
        success = True
    except:
        success = False

    waited = 0
    while not success:
        print("Waiting for server to start...")
        time.sleep(5)
        waited += 5
        try:
            res = requests.get(f"{base_url}:{port}")
            res.raise_for_status()
            success = True
        except:
            success = False
        if waited > 30:
            raise Exception("Test server failed to start")


if __name__ == "__main__":
    test_clip()
