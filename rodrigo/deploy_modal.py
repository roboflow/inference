import time

import modal

from inference.core.env import (
    WEBRTC_MODAL_APP_NAME,
    WEBRTC_MODAL_IMAGE_TAG,
    WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
    WEBRTC_MODAL_PRELOAD_HF_IDS,
    WEBRTC_MODAL_PRELOAD_MODELS,
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
)
from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest, WebRTCOffer
from inference.core.interfaces.webrtc_worker.modal import app, spawn_rtc_peer_connection_modal
from inference.core.version import __version__
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WorkflowConfiguration,
)

docker_tag: str = WEBRTC_MODAL_IMAGE_TAG if WEBRTC_MODAL_IMAGE_TAG else __version__

with modal.enable_output():
    client = modal.Client.from_credentials(
        token_id=WEBRTC_MODAL_TOKEN_ID,
        token_secret=WEBRTC_MODAL_TOKEN_SECRET,
    )
    print(f"Deploying '{WEBRTC_MODAL_APP_NAME}' app with tag '{docker_tag}'")
    app.deploy(name=WEBRTC_MODAL_APP_NAME, client=client, tag=docker_tag)
    # print("Warming up functions")
    # for i in range(20):
    #     print("-------------------------")
    #     print(f"#{i+1}")
    #     print("-------------------------")
    #
    #     try:
    #         print("Spawning RTC peer connection modal CPU")
    #         answer = spawn_rtc_peer_connection_modal(
    #             webrtc_request=WebRTCWorkerRequest(
    #                 processing_timeout=0,
    #                 workflow_configuration=WorkflowConfiguration(
    #                     type="WorkflowConfiguration",
    #                     workflow_specification={
    #                         "version": "1.0",
    #                         "inputs": [{"type": "InferenceImage", "name": "image"}],
    #                         "steps": [],
    #                         "outputs": [{"type": "JsonField", "name": "image", "coordinates_system": "own", "selector": "$inputs.image"}],
    #                     }
    #                 ),
    #                 webrtc_offer=WebRTCOffer(
    #                     type="offer",
    #                     sdp="",
    #                 ),
    #                 api_key=WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
    #                 requested_plan="webrtc-cpu",
    #             )
    #         )
    #         print("RTC peer connection modal CPU spawned, answer:")
    #         print(answer)
    #     except Exception as e:
    #         print(e)
    #     try:
    #         print("Spawning RTC peer connection modal GPU (no preload)")
    #         answer = spawn_rtc_peer_connection_modal(
    #             webrtc_request=WebRTCWorkerRequest(
    #                 processing_timeout=0,
    #                 workflow_configuration=WorkflowConfiguration(
    #                     type="WorkflowConfiguration",
    #                     workflow_specification={
    #                         "version": "1.0",
    #                         "inputs": [{"type": "InferenceImage", "name": "image"}],
    #                         "steps": [],
    #                         "outputs": [{"type": "JsonField", "name": "image", "coordinates_system": "own", "selector": "$inputs.image"}],
    #                     }
    #                 ),
    #                 webrtc_offer=WebRTCOffer(
    #                     type="offer",
    #                     sdp="",
    #                 ),
    #                 requested_plan="webrtc-gpu-small",
    #                 api_key=WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
    #             )
    #         )
    #         print("RTC peer connection modal GPU spawned, answer:")
    #         print(answer)
    #     except Exception as e:
    #         print(e)
    #     if WEBRTC_MODAL_PRELOAD_HF_IDS:
    #         try:
    #             print("Spawning RTC peer connection modal GPU (preloaded HF IDs)")
    #             answer = spawn_rtc_peer_connection_modal(
    #                 webrtc_request=WebRTCWorkerRequest(
    #                     processing_timeout=0,
    #                     workflow_configuration=WorkflowConfiguration(
    #                         type="WorkflowConfiguration",
    #                         workflow_specification={
    #                             "version": "1.0",
    #                             "inputs": [{"type": "InferenceImage", "name": "image"}],
    #                             "steps": [
    #                                 {
    #                                    "type": "roboflow_core/roboflow_object_detection_model@v1",
    #                                     "name": "model",
    #                                     "images": "$inputs.image",
    #                                     "model_id": "instant/model",
    #                                 },
    #                             ],
    #                             "outputs": [{"type": "JsonField", "name": "image", "coordinates_system": "own", "selector": "$inputs.image"}],
    #                         }
    #                     ),
    #                     webrtc_offer=WebRTCOffer(
    #                         type="offer",
    #                         sdp="",
    #                     ),
    #                     requested_plan="webrtc-gpu-small",
    #                     api_key=WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
    #                 )
    #             )
    #             print("RTC peer connection modal GPU spawned, answer:")
    #             print(answer)
    #         except Exception as e:
    #             print(e)
    #     if WEBRTC_MODAL_PRELOAD_MODELS:
    #         print("Spawning RTC peer connection modal GPU (preloaded models)")
    #         preload_models = [m for m in WEBRTC_MODAL_PRELOAD_MODELS.split(",")]
    #         for preload_model in preload_models:
    #             print(f"Spawning RTC peer connection modal GPU (preloaded model: {preload_model})")
    #             try:
    #                 answer = spawn_rtc_peer_connection_modal(
    #                     webrtc_request=WebRTCWorkerRequest(
    #                         processing_timeout=0,
    #                         workflow_configuration=WorkflowConfiguration(
    #                             type="WorkflowConfiguration",
    #                             workflow_specification={
    #                                 "version": "1.0",
    #                                 "inputs": [{"type": "InferenceImage", "name": "image"}],
    #                                 "steps": [
    #                                     {
    #                                         "type": "roboflow_core/roboflow_object_detection_model@v1",
    #                                         "name": "model",
    #                                         "images": "$inputs.image",
    #                                         "model_id": preload_model,
    #                                     },
    #                                 ],
    #                                 "outputs": [{"type": "JsonField", "name": "image", "coordinates_system": "own", "selector": "$inputs.image"}],
    #                             }
    #                         ),
    #                         webrtc_offer=WebRTCOffer(
    #                             type="offer",
    #                             sdp="",
    #                         ),
    #                         requested_gpu="T4",
    #                         api_key=WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
    #                     )
    #                 )
    #                 print("RTC peer connection modal GPU spawned, answer:")
    #                 print(answer)
    #             except Exception as e:
    #                 print(e)
    #
    #     print(f"Sleeping scaledown timeout {WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW+5}s to ensure container is down")
    #     time.sleep(WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW+5)
