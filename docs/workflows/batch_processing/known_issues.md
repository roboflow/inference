# Known issues with Roboflow Batch Processing

This page lists known issues, limitations, and workarounds for the Roboflow Batch Processing feature. 
Our team actively monitors and maintains Batch Processing to ensure performance, reliability, and scalability. However, 
as with any evolving system, certain edge cases or temporary bugs may affect functionality.

If you encounter a problem not listed here, please report it through our 
[support channels](https://github.com/roboflow/inference/issues), so we can investigate and improve your experience.


## Job timed out

### Issue
Batch jobs may terminate prematurely or behave unexpectedly if the **Processing Timeout Hours** value is 
set too low relative to the job’s size or complexity.

<div align="center">
    <img src="https://media.roboflow.com/inference/batch-processing/batch-processing-timeout.png" alt="Batch Processing Timeout settings"/>
</div>

### Details
The Processing Timeout Hours setting in the UI (or `--max-runtime-seconds` flag in CLI commands) defines the maximum 
cumulative machine runtime across all parallel workers. This affects both image and video jobs triggered via:

* Roboflow App UI

* `inference rf-cloud batch-processing process-images-with-workflow` command

* `inference rf-cloud batch-processing process-videos-with-workflow` command


### Important notes 

* **Total compute time:** The timeout represents total runtime across all machines. For example, if the limit is 
2 hours and the job spawns 2 machines, they can each run for a maximum of 1 hour before termination 
(2 machines × 1 hour = 2 hours total).

* **Divided per chunk:** Jobs are broken into processing chunks—such as image shards or individual video 
files—to enable parallelism (similar to a map-reduce pattern). The specified timeout is divided evenly across these 
chunks. If you set a short timeout and have many chunks, each may be given too little time to complete. 

* **Impact of machine type:** Running complex Workflows (e.g., those with multiple deep learning models) on CPU 
increases processing time. Use GPU where appropriate to avoid hitting the timeout.

### Recommendations

* For large datasets or multi-stage Workflows, start with a generous timeout (e.g., 4–6 hours).

* Monitor actual job runtimes to inform future timeout settings.

* Consider reducing the number of chunks or using video frame sub-sampling for faster processing.

* We are actively working on smarter timeout defaults, better runtime estimates, and improved chunk scheduling to 
make this more predictable.


## Workflow with SAHI runs too long

### Issue
Some jobs — particularly those using SAHI for image or video processing—may take significantly longer than expected 
or hit timeout limits, **especially when paired with large input resolutions and instance segmentation.**

### Root Causes and Recommendations

#### Excessive number of slices
SAHI splits large images or video frames into smaller slices to run detection more accurately. 
However, with default settings and high-resolution inputs, this can result in dozens or even hundreds of inferences 
per image/frame.

**Recommendation:** Check the configuration of the Image Slicer block. You can reduce the number of slices or downscale 
input images before slicing using a Resize Image block earlier in the Workflow.

#### Try larger model input size instead of SAHI
In many cases, training a model that supports larger input dimensions can eliminate the need for SAHI entirely. 
While such a model may be larger and require GPU usage, it can be more efficient than hundreds of smaller 
inferences per image.

**Recommendation:** This approach requires a solid dataset and should be tested on a small sample before fully 
committing. It **may not be practical** if you're relying only on pre-trained models.

#### Instance Segmentation bottleneck
When SAHI is used with instance segmentation, the Detections Stitch block (especially with 
Non-Maximum Suppression enabled) can become a major bottleneck.

**Recommendation:** In extreme cases, stitching results for a single frame can take tens of seconds. 
It is recommended for now to verify if SAHI is feasible for your instance segmentation use-case before fully committing 
to the solution


#### Video jobs with SAHI
Process fewer frames using FPS sub-sampling:
 
* It’s often unnecessary to run inference on every frame.

* In the UI, use the Video FPS sub-sampling dropdown to skip frames.

* In the CLI, set the --max-video-fps flag with `inference rf-cloud batch-processing process-videos-with-workflow` command

This can significantly reduce job time and cost, especially for long or high-FPS videos.

<div align="center">
    <img src="https://media.roboflow.com/inference/batch-processing/limiting-video-fps.png" alt="Batch Processing Timeout settings"/>
</div>


## Out of Memory (OOM) Errors

### Issue

Jobs may fail due to Out of Memory (OOM) errors when the Workflow consumes more RAM or VRAM than the allocated machine 
can provide. These errors most commonly occur due to excessive memory pressure on RAM, though GPU VRAM issues are 
also possible when using large models.

### Common Causes

* **SAHI + Instance Segmentation:** This combination is known to be extremely memory-intensive. SAHI multiplies 
inference calls per input, and instance segmentation generates large outputs (e.g., masks and scores). This often 
leads to unstable behavior or crashes, especially with Detections Stitch and NMS.

* **Too Many Workers Per Machine:** The system allows you to parallelize jobs by assigning multiple workers per machine. 
This can optimize cost and speed—if your Workflow is lightweight enough. However, **bulky Workflows** 
(e.g., those with multiple large models or complex post-processing) will exceed available memory when run with too 
many workers on a single node.

### Recommendations
Use fewer workers per machine (e.g., 1 or 2) when your Workflow includes:

* Large models

* SAHI (especially with instance segmentation)

* High-resolution input images

If encountering OOM errors:

* Lower the Workers Per Machine value under Advanced Options.

* Switch from CPU to GPU if your model or inference block requires higher memory throughput.

* Test your Workflow on a small dataset before running large batches.

* Reduce input resolution or simplify your Workflow by removing unneeded blocks.

<div align="center">
    <img src="https://media.roboflow.com/inference/batch-processing/workers-number-adjustment.png" alt="Batch Processing Timeout settings"/>
</div>

