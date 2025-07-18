import time
from typing import Any, Dict, List, Optional

import supervision as sv

from inference.core import logger
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog
from inference_cli.lib.benchmark.results_gathering import ResultsCollector


class PipelineMetrics:
    """Track metrics for a single pipeline."""
    
    def __init__(self):
        self.fps_monitor = sv.FPSMonitor(sample_size=30)
        self.frame_count = 0
        self.error_count = 0
        self.start_time = None
        
    def start(self):
        """Start timing the pipeline."""
        self.start_time = time.time()
        
    def register_frame(self):
        """Register a successfully processed frame."""
        self.frame_count += 1
        self.fps_monitor.tick()
        
    def register_error(self):
        """Register an error during processing."""
        self.error_count += 1
        
    @property
    def fps(self) -> float:
        """Get current FPS."""
        return self.fps_monitor.fps
        
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


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
    Run benchmark for InferencePipeline with multiple concurrent pipelines using watchdog monitoring.

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

    # Log benchmark configuration
    _log_benchmark_start(
        num_pipelines, model_id, workflow_id, workspace_name, 
        workflow_specification, duration_seconds
    )

    # Initialize metrics tracking
    results_collector.start_benchmark()
    metrics = [PipelineMetrics() for _ in video_sources]
    watchdogs = [BasePipelineWatchDog() for _ in video_sources]
    pipelines = []

    # Create and start pipelines
    for i, video_source in enumerate(video_sources):
        pipeline = _create_pipeline(
            idx=i,
            video_source=video_source,
            metrics=metrics[i],
            watchdog=watchdogs[i],
            results_collector=results_collector,
            model_id=model_id,
            workflow_id=workflow_id,
            workspace_name=workspace_name,
            workflow_specification=workflow_specification,
            workflow_parameters=workflow_parameters,
            max_fps=max_fps,
            api_key=api_key,
            model_configuration=model_configuration,
        )
        
        if pipeline:
            pipelines.append(pipeline)
            metrics[i].start()

    # Run benchmark
    _run_benchmark_loop(
        pipelines=pipelines,
        metrics=metrics,
        watchdogs=watchdogs,
        duration_seconds=duration_seconds,
        results_collector=results_collector,
    )

    # Stop pipelines and display results
    _stop_pipelines(pipelines)
    _display_final_results(
        metrics=metrics,
        watchdogs=watchdogs,
        num_pipelines=num_pipelines,
        model_id=model_id,
    )
    
    results_collector.stop_benchmark()


def _log_benchmark_start(
    num_pipelines: int,
    model_id: Optional[str],
    workflow_id: Optional[str],
    workspace_name: Optional[str],
    workflow_specification: Optional[Dict[str, Any]],
    duration_seconds: int,
) -> None:
    """Log benchmark start information."""
    if model_id:
        logger.info(
            f"Starting pipeline benchmark with {num_pipelines} pipelines "
            f"for model {model_id} for {duration_seconds} seconds"
        )
    elif workflow_specification:
        logger.info(
            f"Starting pipeline benchmark with {num_pipelines} pipelines "
            f"for local workflow for {duration_seconds} seconds"
        )
    else:
        logger.info(
            f"Starting pipeline benchmark with {num_pipelines} pipelines "
            f"for workflow {workspace_name}/{workflow_id} for {duration_seconds} seconds"
        )


def _create_pipeline(
    idx: int,
    video_source: str,
    metrics: PipelineMetrics,
    watchdog: BasePipelineWatchDog,
    results_collector: ResultsCollector,
    model_id: Optional[str],
    workflow_id: Optional[str],
    workspace_name: Optional[str],
    workflow_specification: Optional[Dict[str, Any]],
    workflow_parameters: Optional[Dict[str, Any]],
    max_fps: Optional[float],
    api_key: Optional[str],
    model_configuration: Optional[str],
) -> Optional[InferencePipeline]:
    """Create and start a single pipeline."""
    
    def on_prediction(predictions, video_frame):
        """Callback for processing predictions."""
        try:
            metrics.register_frame()
            # Register with results collector for benchmark metrics
            # Use actual FPS to estimate processing time
            fps = metrics.fps
            duration = 1.0 / fps if fps > 0 else 0.033
            results_collector.register_inference_duration(
                batch_size=1, duration=duration
            )
        except Exception as e:
            logger.error(f"Error in pipeline {idx} callback: {e}")
            metrics.register_error()
            results_collector.register_error(
                batch_size=1, status_code="pipeline_error"
            )

    try:
        # Debug logging for video source
        logger.info(f"Pipeline {idx} - Raw video_source value: {repr(video_source)}")
        logger.info(f"Pipeline {idx} - Contains exclamation mark: {'!' in str(video_source)}")
        
        # Base pipeline configuration
        pipeline_config = {
            "video_reference": video_source,
            "on_prediction": on_prediction,
            "api_key": api_key,
            "watchdog": watchdog,
        }

        if max_fps is not None:
            pipeline_config["max_fps"] = max_fps

        # Create pipeline based on type
        if model_id:
            pipeline_config["model_id"] = model_id
            if model_configuration:
                pipeline_config["model_configuration"] = model_configuration
            pipeline = InferencePipeline.init(**pipeline_config)
            logger.info(
                f"Started pipeline {idx} with model {model_id} on video source: {video_source}"
            )
        else:
            if workflow_specification:
                pipeline_config["workflow_specification"] = workflow_specification
            else:
                pipeline_config["workspace_name"] = workspace_name
                pipeline_config["workflow_id"] = workflow_id

            if workflow_parameters:
                pipeline_config["workflows_parameters"] = workflow_parameters

            pipeline = InferencePipeline.init_with_workflow(**pipeline_config)
            logger.info(
                f"Started workflow pipeline {idx} with video source: {video_source}"
            )

        # Start pipeline
        pipeline.start(use_main_thread=False)
        return pipeline

    except Exception as e:
        logger.error(f"Failed to start pipeline {idx}: {e}")
        results_collector.register_error(
            batch_size=1, status_code="pipeline_init_error"
        )
        return None


def _run_benchmark_loop(
    pipelines: List[InferencePipeline],
    metrics: List[PipelineMetrics],
    watchdogs: List[BasePipelineWatchDog],
    duration_seconds: int,
    results_collector: ResultsCollector,
) -> None:
    """Run the main benchmark loop with periodic reporting."""
    start_time = time.time()
    last_report_time = start_time
    report_interval = 10  # Print report every 10 seconds

    try:
        while time.time() - start_time < duration_seconds:
            time.sleep(1)
            elapsed = time.time() - start_time

            # Print periodic reports
            if time.time() - last_report_time >= report_interval:
                _print_progress_report(elapsed, metrics, watchdogs)
                last_report_time = time.time()

            # Check if all pipelines have stopped
            all_stopped = all(
                hasattr(pipeline, "_stop") and pipeline._stop
                for pipeline in pipelines
            )
            if all_stopped:
                logger.info("All pipelines have completed processing")
                break

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")


def _print_progress_report(
    elapsed: float,
    metrics: List[PipelineMetrics],
    watchdogs: List[BasePipelineWatchDog],
) -> None:
    """Print periodic progress report."""
    logger.info(f"\n=== Progress Report at {int(elapsed)}s ===")
    
    total_frames = 0
    total_errors = 0
    
    for i, (metric, watchdog) in enumerate(zip(metrics, watchdogs)):
        fps = metric.fps
        frames = metric.frame_count
        errors = metric.error_count
        total_frames += frames
        total_errors += errors
        
        logger.info(
            f"Pipeline {i}: FPS={fps:.2f}, "
            f"Frames={frames}, Errors={errors}"
        )
        
        # Add watchdog latency information if available
        report = watchdog.get_report()
        if report and report.latency_reports:
            for latency_report in report.latency_reports:
                if latency_report.e2e_latency:
                    logger.info(
                        f"  - E2E Latency: {latency_report.e2e_latency:.3f}s"
                    )
                if latency_report.inference_latency:
                    logger.info(
                        f"  - Inference Latency: {latency_report.inference_latency:.3f}s"
                    )
    
    logger.info(f"Total: {total_frames} frames, {total_errors} errors\n")


def _stop_pipelines(pipelines: List[InferencePipeline]) -> None:
    """Stop all pipelines gracefully."""
    logger.info("Stopping all pipelines...")
    for i, pipeline in enumerate(pipelines):
        try:
            pipeline.terminate()
            pipeline.join()
            logger.info(f"Pipeline {i} stopped")
        except Exception as e:
            logger.error(f"Error stopping pipeline {i}: {e}")


def _display_final_results(
    metrics: List[PipelineMetrics],
    watchdogs: List[BasePipelineWatchDog],
    num_pipelines: int,
    model_id: Optional[str],
) -> None:
    """Display final benchmark results."""
    # Calculate totals
    total_frames = sum(m.frame_count for m in metrics)
    total_errors = sum(m.error_count for m in metrics)
    
    # Header
    benchmark_type = "Model" if model_id else "Workflow"
    print(f"\n=== {benchmark_type} Pipeline Benchmark Results ===")
    
    # Summary statistics
    if metrics and metrics[0].start_time:
        actual_duration = max(m.elapsed_time for m in metrics)
        print(f"Total benchmark duration: {actual_duration:.2f} seconds")
    
    print(f"Total pipelines: {num_pipelines}")
    print(f"Total frames processed: {total_frames}")
    print(f"Total errors: {total_errors}")
    
    if total_frames + total_errors > 0:
        error_rate = (total_errors / (total_frames + total_errors)) * 100
        print(f"Error rate: {error_rate:.2f}%")

    # Per-pipeline statistics
    print("\n=== Per-Pipeline Statistics ===")
    aggregate_fps = 0
    
    for i, (metric, watchdog) in enumerate(zip(metrics, watchdogs)):
        fps = metric.fps
        aggregate_fps += fps
        
        print(f"\nPipeline {i}:")
        print(f"  Frames processed: {metric.frame_count}")
        print(f"  Errors: {metric.error_count}")
        print(f"  Average FPS: {fps:.2f}")
        
        # Watchdog statistics
        report = watchdog.get_report()
        if report:
            if report.inference_throughput:
                print(f"  Watchdog FPS: {report.inference_throughput:.2f}")
            
            if report.latency_reports:
                for latency_report in report.latency_reports:
                    if latency_report.e2e_latency:
                        print(f"  Average E2E latency: {latency_report.e2e_latency:.3f}s")
                    if latency_report.inference_latency:
                        print(f"  Average inference latency: {latency_report.inference_latency:.3f}s")
                    if latency_report.frame_decoding_latency:
                        print(f"  Average frame decoding latency: {latency_report.frame_decoding_latency:.3f}s")

    # Aggregate statistics
    if num_pipelines > 0:
        print(f"\nAggregate FPS (all pipelines): {aggregate_fps:.2f}")
        print(f"Average FPS per pipeline: {aggregate_fps / num_pipelines:.2f}")