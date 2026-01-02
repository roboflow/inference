#!/usr/bin/env python3
"""Standalone SAM3 test script - no pytest needed."""

import requests

API_KEY = "zaRavHwbvIXpGerDM3wi"
BASE_URL = "http://localhost"
PORT = 9101

def test_image_embedding():
    print("Testing image embedding...")
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
    }
    response = requests.post(
        f"{BASE_URL}:{PORT}/sam3/embed_image",
        json=payload,
        params={"api_key": API_KEY}
    )
    response.raise_for_status()
    data = response.json()
    assert "image_id" in data
    assert "time" in data
    print(f"✓ Embedding succeeded: {data}")

def test_visual_segmentation():
    print("Testing visual segmentation...")
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        }
    }
    response = requests.post(
        f"{BASE_URL}:{PORT}/sam3/visual_segment",
        json=payload,
        params={"api_key": API_KEY}
    )
    response.raise_for_status()
    print(f"✓ Visual segmentation succeeded")

def test_concept_segmentation():
    print("Testing concept segmentation...")
    payload = {
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/dog.jpeg",
        },
        "prompts": [
            {
                "type": "text",
                "text": "dog",
            }
        ]
    }
    response = requests.post(
        f"{BASE_URL}:{PORT}/sam3/concept_segment",
        json=payload,
        params={"api_key": API_KEY}
    )
    response.raise_for_status()
    print(f"✓ Concept segmentation succeeded")

if __name__ == "__main__":
    test_image_embedding()
    test_visual_segmentation()
    test_concept_segmentation()
    print("\nAll SAM3 tests passed!")