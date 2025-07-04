import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from inference.core import logger
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference_cli.lib.benchmark.results_gathering import ResultsCollector


def run_pipeline_speed_benchmark(
    video_sources: List[str],
    results_collector: ResultsCollector,
    model_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    workspace_name: Optional[str] = None,
    workflow_specification: Optional[Dict[str, Any]] = None,
    workflow_parameters: Optional[Dict[str, Any]] = None,
    duration_seconds: int = 60,
    max_fps: Optional[float] = None,
    api_key: Optional[str] = None,
    model_configuration: Optional[str] = None,
) -> None:
    """
    Run benchmark for InferencePipeline with multiple concurrent pipelines.

    Args:
        video_sources: List of video sources (one per pipeline)
        results_collector: ResultsCollector instance to track metrics
        model_id: Model ID to use for inference (mutually exclusive with workflow params)
        workflow_id: Workflow ID to use (requires workspace_name)
        workspace_name: Workspace name for workflow
        workflow_specification: Local workflow specification dict
        workflow_parameters: Additional workflow parameters
        duration_seconds: Duration of benchmark in seconds
        max_fps: Maximum FPS limit for pipelines
        api_key: Roboflow API key
        model_configuration: Optional model configuration
    """
    num_pipelines = len(video_sources)

    # Validate inputs - either model_id or workflow params, not both
    if model_id and (workflow_id or workflow_specification):
        raise ValueError("Cannot specify both model_id and workflow parameters")
    if not model_id and not workflow_id and not workflow_specification:
        raise ValueError("Must specify either model_id or workflow parameters")
    if workflow_id and not workspace_name:
        raise ValueError("workspace_name is required when using workflow_id")

    if model_id:
        logger.info(
            f"Starting pipeline benchmark with {num_pipelines} pipelines for model {model_id} for {duration_seconds} seconds"
        )
    elif workflow_specification:
        logger.info(
            f"Starting pipeline benchmark with {num_pipelines} pipelines for local workflow for {duration_seconds} seconds"
        )
    else:
        logger.info(
            f"Starting pipeline benchmark with {num_pipelines} pipelines for workflow {workspace_name}/{workflow_id} for {duration_seconds} seconds"
        )

    # Start the benchmark timer
    results_collector.start_benchmark()

    # Create and start pipelines
    pipelines = []
    pipeline_stats = []

    for i, video_source in enumerate(video_sources):
        stats = {
            "pipeline_id": i,
            "frames_processed": 0,
            "errors": 0,
            "start_time": None,
            "last_frame_time": None,
        }
        pipeline_stats.append(stats)

        def create_callback(pipeline_idx):
            def on_prediction(predictions, video_frame):
                try:
                    current_time = time.time()
                    
                    # Track successful inference
                    pipeline_stats[pipeline_idx]["frames_processed"] += 1
                    if pipeline_stats[pipeline_idx]["start_time"] is None:
                        pipeline_stats[pipeline_idx]["start_time"] = current_time
                        pipeline_stats[pipeline_idx]["last_frame_time"] = current_time
                    
                    # Calculate actual duration since last frame
                    duration = current_time - pipeline_stats[pipeline_idx]["last_frame_time"]
                    pipeline_stats[pipeline_idx]["last_frame_time"] = current_time

                    # Report to results collector with actual duration
                    results_collector.register_inference_duration(
                        batch_size=1, duration=duration
                    )
                except Exception as e:
                    logger.error(f"Error in pipeline {pipeline_idx} callback: {e}")
                    pipeline_stats[pipeline_idx]["errors"] += 1
                    results_collector.register_error(
                        batch_size=1, status_code="pipeline_error"
                    )

            return on_prediction

        try:
            # Initialize pipeline - either with model or workflow
            if model_id:
                # Model-based pipeline
                pipeline_config = {
                    "model_id": model_id,
                    "video_reference": video_source,
                    "on_prediction": create_callback(i),
                    "api_key": api_key,
                }

                if max_fps is not None:
                    pipeline_config["max_fps"] = max_fps

                if model_configuration:
                    pipeline_config["model_configuration"] = model_configuration

                pipeline = InferencePipeline.init(**pipeline_config)
            else:
                # Workflow-based pipeline
                pipeline_config = {
                    "video_reference": video_source,
                    "on_prediction": create_callback(i),
                    "api_key": api_key,
                }

                if max_fps is not None:
                    pipeline_config["max_fps"] = max_fps

                if workflow_specification:
                    pipeline_config["workflow_specification"] = workflow_specification
                else:
                    pipeline_config["workspace_name"] = workspace_name
                    pipeline_config["workflow_id"] = workflow_id

                if workflow_parameters:
                    pipeline_config["workflows_parameters"] = workflow_parameters

                pipeline = InferencePipeline.init_with_workflow(**pipeline_config)

            pipelines.append(pipeline)

            # Start pipeline with dispatching in separate thread
            pipeline.start(use_main_thread=False)
            if model_id:
                logger.info(
                    f"Started pipeline {i} with model {model_id} on video source: {video_source}"
                )
            else:
                logger.info(
                    f"Started workflow pipeline {i} with video source: {video_source}"
                )

        except Exception as e:
            logger.error(f"Failed to start pipeline {i}: {e}")
            results_collector.register_error(
                batch_size=1, status_code="pipeline_init_error"
            )

    # Let pipelines run for the specified duration
    start_time = time.time()

    try:
        # Track when frames were last processed to detect completion
        last_frame_counts = [0] * len(pipelines)
        stall_check_interval = 3  # Check every 3 seconds
        stall_threshold = 5  # Consider stalled if no new frames for 5 seconds
        last_stall_check = time.time()
        
        while time.time() - start_time < duration_seconds:
            time.sleep(1)

            # Check if all pipelines have stopped
            all_stopped = True
            any_processing = False
            
            for i, pipeline in enumerate(pipelines):
                if hasattr(pipeline, "_stop") and pipeline._stop:
                    logger.debug(f"Pipeline {i} has stopped")
                else:
                    all_stopped = False
                
                # Check if pipeline is still processing frames
                current_frames = pipeline_stats[i]["frames_processed"]
                if current_frames > last_frame_counts[i]:
                    any_processing = True
                    last_frame_counts[i] = current_frames

            # If all pipelines have stopped, exit early
            if all_stopped:
                logger.info("All pipelines have completed processing")
                break
            
            # Check for stalled processing every few seconds
            current_time = time.time()
            if current_time - last_stall_check >= stall_check_interval:
                if not any_processing:
                    # Check if we've been stalled for too long
                    stall_duration = 0
                    for i, stats in enumerate(pipeline_stats):
                        if stats["last_frame_time"]:
                            frame_age = current_time - stats["last_frame_time"]
                            stall_duration = max(stall_duration, frame_age)
                    
                    if stall_duration > stall_threshold:
                        logger.info(f"No new frames processed for {stall_duration:.1f}s, assuming video processing complete")
                        break
                
                last_stall_check = current_time

            # Log periodic stats
            if int(time.time() - start_time) % 10 == 0:
                total_frames = sum(
                    stats["frames_processed"] for stats in pipeline_stats
                )
                total_errors = sum(stats["errors"] for stats in pipeline_stats)
                logger.info(
                    f"Progress: {int(time.time() - start_time)}s - Total frames: {total_frames}, Errors: {total_errors}"
                )

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")

    # Stop all pipelines
    logger.info("Stopping all pipelines...")
    for i, pipeline in enumerate(pipelines):
        try:
            pipeline.terminate()
            pipeline.join()
            logger.info(f"Pipeline {i} stopped")
        except Exception as e:
            logger.error(f"Error stopping pipeline {i}: {e}")

    # Calculate and log final statistics
    total_frames = sum(stats["frames_processed"] for stats in pipeline_stats)
    total_errors = sum(stats["errors"] for stats in pipeline_stats)
    actual_duration = time.time() - start_time
    
    # Calculate actual processing duration (from first frame to last frame)
    processing_start = None
    processing_end = None
    for stats in pipeline_stats:
        if stats["start_time"] is not None:
            if processing_start is None or stats["start_time"] < processing_start:
                processing_start = stats["start_time"]
            if stats["last_frame_time"] is not None:
                if processing_end is None or stats["last_frame_time"] > processing_end:
                    processing_end = stats["last_frame_time"]
    
    processing_duration = processing_end - processing_start if processing_start and processing_end else 0

    # Calculate error rate
    error_rate = (total_errors / (total_frames + total_errors) * 100) if (total_frames + total_errors) > 0 else 0
    
    benchmark_type = "Model" if model_id else "Workflow"
    print(f"\n=== {benchmark_type} Pipeline Benchmark Results ===")
    print(f"Total benchmark duration: {actual_duration:.2f} seconds")
    print(f"Actual processing duration: {processing_duration:.2f} seconds")
    print(f"Total pipelines: {num_pipelines}")
    print(f"Total frames processed: {total_frames}")
    print(f"Total errors: {total_errors}")
    print(f"Error rate: {error_rate:.2f}%")
    
    if num_pipelines > 0 and processing_duration > 0:
        print(
            f"Average FPS per pipeline: {total_frames / processing_duration / num_pipelines:.2f}"
        )
        print(f"Aggregate FPS: {total_frames / processing_duration:.2f}")
    elif num_pipelines > 0 and actual_duration > 0:
        # Fallback to total duration if processing duration is not available
        print(
            f"Average FPS per pipeline (using total duration): {total_frames / actual_duration / num_pipelines:.2f}"
        )
        print(f"Aggregate FPS (using total duration): {total_frames / actual_duration:.2f}")
    else:
        logger.info("No FPS data available")

    # Report per-pipeline statistics
    for stats in pipeline_stats:
        pipeline_id = stats["pipeline_id"]
        frames = stats["frames_processed"]
        errors = stats["errors"]
        if stats["start_time"]:
            pipeline_duration = time.time() - stats["start_time"]
            fps = frames / pipeline_duration if pipeline_duration > 0 else 0
            logger.info(
                f"Pipeline {pipeline_id}: {frames} frames, {errors} errors, {fps:.2f} FPS"
            )

    # Mark benchmark as finished
    results_collector.stop_benchmark()
