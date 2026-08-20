from development.video_poc.experiments.process_isolation.mps_probe import MPSController


def test_mps_client_environment_is_scoped(tmp_path):
    controller = MPSController(
        "/usr/bin/nvidia-cuda-mps-control",
        base_directory=str(tmp_path),
        active_thread_percentage=25,
        pinned_device_memory_limit="0=8G",
    )
    env = controller.client_environment({"CUDA_VISIBLE_DEVICES": "GPU-test"})
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == "25"
    assert env["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"] == "0=8G"
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == str(tmp_path / "pipe")
    assert env["CUDA_MPS_LOG_DIRECTORY"] == str(tmp_path / "log")


def test_mps_active_thread_percentage_is_validated():
    try:
        MPSController("control", active_thread_percentage=0)
    except ValueError as error:
        assert "[1, 100]" in str(error)
    else:
        raise AssertionError("expected ValueError")
