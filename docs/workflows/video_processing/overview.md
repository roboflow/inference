# Video Processing with Workflows

We've begun our journey into video processing using Workflows. Over time, we've expanded the number of 
video-specific blocks (e.g., the ByteTracker block) and continue to dedicate efforts toward improving 
their performance and robustness. The current state of this work is as follows:

* We've introduced the `WorkflowVideoMetadata` input to store metadata related to video frames, 
including declared FPS, measured FPS, timestamp, video source identifier, and file/stream flags. While this may not be the final approach 
for handling video metadata, it allows us to build stateful video-processing blocks at this stage. 
If your Workflow includes any blocks requiring input of kind `video_metadata`, you must define this input in 
your Workflow. The metadata functions as a batch-oriented parameter, treated by the Execution Engine in the same
way as `WorkflowImage`.

* The `InferencePipeline` supports 
[video processing with Workflows](/using_inference/inference_pipeline.md#inferencepipeline-and-roboflow-workflows) 
by automatically injecting `WorkflowVideoMetadata` into the `video_metadata` field. This allows you to seamlessly run
your Workflow using the `InferencePipeline` within the `inference` Python package.

* In the `0.21.0` release, we've initiated efforts to enable video processing management via the `inference` server API. 
This means that eventually, no custom scripts will be required to process video using Workflows and `InferencePipeline`.
You'll simply call an endpoint, specify the video source and the workflow, and the server will handle the rest—allowing
you to focus on consuming the results.


## Video management API - comments and status update

This is an experimental feature, and breaking changes may be introduced over time. There is a list of
[known issues](https://github.com/roboflow/inference/issues?q=is%3Aopen+is%3Aissue+label%3A%22Video+Management+API+issues%22).
Please visit the page to raise new issues or comment on existing ones.

### Release `0.21.0`

* Added basic endpoints to `list`, `start`, `pause`, `resume`, `terminate` and `consume` results of `InferencePipelines`
running under control of `inference` server. Endpoints are enabled in `inference` server docker images for CPU, GPU
and Jetson devices. Running inference server there would let you call 
[`http://127.0.0.1:9001/docs`](http://127.0.0.1:9001/docs) to retrieve OpenAPI schemas for endpoints.

* Added HTTP client for new endpoints into `InferenceHTTPClient` from `inference_sdk`. Here you may find examples on
how to use the client and API to start processing videos today:


!!! Note

    Package version `inference~=0.21.0` is used in this example. Because the feature is experimental, the code
    may evolve over time.

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://192.168.0.115:9001",
    api_key="<ROBOFLOW-API-KEY>"
)

# to list active pipelines
client.list_inference_pipelines()

# start processing - single stream
client.start_inference_pipeline_with_workflow(
    video_reference=["rtsp://192.168.0.73:8554/live0.stream"],
    workspace_name="<YOUR-WORKSPACE>",
    workflow_id="<YOUR-WORKFLOW-ID>",
    results_buffer_size=5,  # results are consumed from in-memory buffer - optionally you can control its size
)

# start processing - one RTSP stream and one camera
# USB camera cannot be passed easily to docker running on MacBook, but on Jetson devices it works :)

client.start_inference_pipeline_with_workflow(
    video_reference=["rtsp://192.168.0.73:8554/live0.stream", 0],
    workspace_name="<YOUR-WORKSPACE>",
    workflow_id="<YOUR-WORKFLOW-ID>",
    batch_collection_timeout=0.05,  # for consumption of multiple video sources it is ADVISED to 
    # set batch collection timeout (defined as fraction of seconds - 0.05 = 50ms)
)

# start_inference_pipeline_with_workflow(...) will provide you pipeline_id which may be used to:

# * get pipeline status
client.get_inference_pipeline_status(
    pipeline_id="182452f4-a2c1-4537-92e1-ec64d1e42de1",
)

# * pause pipeline
client.pause_inference_pipeline(
    pipeline_id="182452f4-a2c1-4537-92e1-ec64d1e42de1",
)

# * resume pipeline
client.resume_inference_pipeline(
    pipeline_id="182452f4-a2c1-4537-92e1-ec64d1e42de1",
)

# * terminate pipeline
client.terminate_inference_pipeline(
    pipeline_id="182452f4-a2c1-4537-92e1-ec64d1e42de1",
)

# * consume pipeline results
client.consume_inference_pipeline_result(
    pipeline_id="182452f4-a2c1-4537-92e1-ec64d1e42de1",
    excluded_fields=["workflow_output_field_to_exclude"]  # this is optional
    # if you wanted to get rid of some outputs to save bandwidth - feel free to discard them
)
```


The client presented above can be used to preview workflow outputs in a very **naive** way. Let's assume
that the workflow you defined runs an object detection model and renders its output using Workflows visualization
blocks that register the output image in the `preview` field. You can use the following script to poll and display processed video
frames:

```python
import cv2
from inference_sdk import InferenceHTTPClient
from inference.core.utils.image_utils import load_image


client = InferenceHTTPClient(
    api_url=f"http://127.0.0.1:9001",
    api_key="<YOUR-API-KEY>",
)

while True:
    result = client.consume_inference_pipeline_result(pipeline_id="<PIPELINE-ID>")
    if not result["outputs"] or not result["outputs"][0]:
        # "outputs" key contains list of workflow results - why list? InferencePipeline can 
        # run on multiple video sources at the same time - each "round" it attempts to 
        # grab single frame from all sources and run through Workflows Execution Engine
        # * when sources are inactive that may not be possible, hence empty list can be returned
        # * when one of the source do not provide frame (for instance due to batch collection timeout)
        # outputs list may contain `None` values!
        continue
    # let's assume single source
    source_result = result["outputs"][0]
    image, _ = load_image(source_result["preview"])  # "preview" is the name of workflow output with image
    cv2.imshow("frame", image)
    cv2.waitKey(1)
```



