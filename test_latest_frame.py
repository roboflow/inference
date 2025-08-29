#!/usr/bin/env python3
"""
Test script for the new /inference_pipelines/{pipeline_id}/latest_frame endpoint.

This script:
1. Initializes a pipeline with a video source
2. Retrieves the latest frame
3. Verifies the response format matches device-manager expectations
"""

import requests
import json
import time
import base64
from io import BytesIO
from PIL import Image

# Configuration
API_BASE_URL = "http://localhost:9001"  # Default inference server port
API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
WORKSPACE_NAME = "YOUR_WORKSPACE"  # Replace with your workspace

def test_latest_frame_endpoint():
    """Test the latest frame endpoint implementation."""
    
    print("🔧 Testing Latest Frame Endpoint Implementation")
    print("=" * 50)
    
    # Step 1: Initialize a test pipeline
    print("\n1. Initializing test pipeline...")
    
    init_payload = {
        "api_key": API_KEY,
        "workspace_name": WORKSPACE_NAME,
        "model_id": "yolov8n-640",  # Using a small model for testing
        "video_reference": "https://media.roboflow.com/inference/people-walking.mp4",  # Test video
        "video_configuration": {
            "type": "VideoConfiguration",
            "video_reference": "https://media.roboflow.com/inference/people-walking.mp4",
            "max_fps": 5,
            "source_buffer_filling_strategy": "DROP_OLDEST",
            "source_buffer_consumption_strategy": "EAGER"
        },
        "processing_configuration": {
            "type": "WorkflowConfiguration",
            "workspace_name": WORKSPACE_NAME,
            "workflow_id": "simple-detection",
            "image_input_name": "image",
            "workflows_parameters": {}
        },
        "sink_configuration": {
            "type": "InMemorySinkConfiguration",
            "results_buffer_size": 10
        }
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/inference_pipelines/initialise",
            json=init_payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if "context" in result and "pipeline_id" in result["context"]:
            pipeline_id = result["context"]["pipeline_id"]
            print(f"✅ Pipeline initialized: {pipeline_id}")
        else:
            print(f"❌ Failed to get pipeline_id from response: {result}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return False
    
    # Step 2: Wait for pipeline to process some frames
    print("\n2. Waiting for pipeline to process frames...")
    time.sleep(5)  # Give pipeline time to start and process frames
    
    # Step 3: Test the latest_frame endpoint
    print(f"\n3. Testing /inference_pipelines/{pipeline_id}/latest_frame endpoint...")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/inference_pipelines/{pipeline_id}/latest_frame",
            timeout=10
        )
        response.raise_for_status()
        frame_data = response.json()
        
        # Verify response structure matches device-manager expectations
        print("\n📊 Response validation:")
        
        required_fields = ["success", "frame_id", "data", "camera_fps", "pipeline_fps"]
        missing_fields = []
        
        for field in required_fields:
            if field in frame_data:
                if field == "data" and frame_data[field]:
                    # Verify it's a valid base64 image
                    try:
                        # Extract base64 data
                        if frame_data[field].startswith("data:image"):
                            base64_str = frame_data[field].split(",")[1]
                        else:
                            base64_str = frame_data[field]
                        
                        # Try to decode
                        img_data = base64.b64decode(base64_str)
                        img = Image.open(BytesIO(img_data))
                        print(f"  ✅ {field}: Valid image ({img.size[0]}x{img.size[1]})")
                    except Exception as e:
                        print(f"  ❌ {field}: Invalid image data - {e}")
                elif field == "success":
                    print(f"  ✅ {field}: {frame_data[field]}")
                else:
                    value = frame_data[field]
                    if value is not None:
                        print(f"  ✅ {field}: {value}")
                    else:
                        print(f"  ⚠️  {field}: None (no data yet)")
            else:
                missing_fields.append(field)
                print(f"  ❌ {field}: Missing")
        
        if missing_fields:
            print(f"\n❌ Missing required fields: {missing_fields}")
            success = False
        else:
            print("\n✅ All required fields present")
            success = True
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get latest frame: {e}")
        success = False
    
    # Step 4: Clean up - terminate the pipeline
    print(f"\n4. Cleaning up - terminating pipeline {pipeline_id}...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/inference_pipelines/{pipeline_id}/terminate",
            timeout=10
        )
        response.raise_for_status()
        print("✅ Pipeline terminated")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Failed to terminate pipeline: {e}")
    
    return success

def main():
    """Main test function."""
    print("\n🚀 Starting Latest Frame Endpoint Test")
    print("Make sure the inference server is running with ENABLE_STREAM_API=True")
    print(f"Testing against: {API_BASE_URL}")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print("❌ Server health check failed")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to inference server")
        print(f"Make sure it's running at {API_BASE_URL}")
        return
    
    # Run the test
    if test_latest_frame_endpoint():
        print("\n✅ ✅ ✅ All tests passed! The latest_frame endpoint is working correctly.")
        print("\nThe endpoint is compatible with device-manager expectations.")
    else:
        print("\n❌ ❌ ❌ Tests failed. Please check the implementation.")
        print("\nTroubleshooting tips:")
        print("1. Make sure ENABLE_STREAM_API=True is set")
        print("2. Check that all code changes were applied correctly")
        print("3. Review server logs for any errors")

if __name__ == "__main__":
    main()