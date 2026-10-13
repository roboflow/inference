"""Real V2 compiler/retained session with a fake camera and actual host owner."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from roboflow_workflows.v2.blocks.industrial.onvif import (
    OnvifController,
    OnvifTrackingBlock,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from tests.v2.unit.blocks.industrial.test_onvif import (
    TARGET,
    Clock,
    ManualSleep,
    RecordingCamera,
    TravellingCamera,
    predictions,
)


def definition(**options):
    return {
        "version": "2.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            }
        ],
        "steps": [
            {
                "type": OnvifTrackingBlock.type,
                "name": "camera",
                "predictions": "$inputs.predictions",
                "camera_ip": "camera.invalid",
                "camera_port": 80,
                "camera_username": "viewer",
                "camera_password": "sentinel-password",
                "dead_zone": 5,
                "pid_kp": 1.0,
                "pid_kd": 0.0,
                "camera_update_rate_limit": 0,
                **options,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "selected",
                "selector": "$steps.camera.predictions",
                "coordinates_system": "own",
            },
            {
                "type": "JsonField",
                "name": "seeking",
                "selector": "$steps.camera.seeking",
            },
        ],
    }


def plan(**options):
    return compile_workflow(
        definition(**options), catalogue=Catalogue([OnvifTrackingBlock])
    )


def run(session, value=None):
    return session.run(
        {"predictions": [predictions() if value is None else value]}
    ).rows()[0]


def test_retained_session_reuses_owner_and_session_close_does_not_close_it():
    camera = RecordingCamera()
    with OnvifController(TARGET, port_factory=lambda t: camera) as owner:
        compiled = plan()
        session = compiled.create_session(resources={"onvif_controller": owner})
        assert run(session)["seeking"]
        assert run(session)["seeking"]
        session.close()
        assert camera.open_count == 1 and camera.close_count == 0
        other = compiled.create_session(resources={"onvif_controller": owner})
        assert run(other)["seeking"]
        other.close()
    assert camera.close_count == 1


def test_two_sessions_with_distinct_owners_isolate_tracker_state_and_close():
    a, b = RecordingCamera(), RecordingCamera()
    with OnvifController(TARGET, port_factory=lambda t: a) as one, OnvifController(
        TARGET, port_factory=lambda t: b
    ) as two:
        compiled = plan()
        first = compiled.create_session(resources={"onvif_controller": one})
        second = compiled.create_session(resources={"onvif_controller": two})
        value = predictions(
            ((70, 40, 90, 60), (10, 40, 30, 60)), confidence=[0.9, 0.1], trackers=[0, 1]
        )
        run(first, value)
        value = predictions(
            ((70, 40, 90, 60), (10, 40, 30, 60)), confidence=[0.1, 0.9], trackers=[0, 1]
        )
        run(second, value)
        assert one._tracked == 0 and two._tracked == 1
        first.close()
        one.close()
        assert run(second, value)["seeking"] and not two.closed
        second.close()
    assert a.close_count == b.close_count == 1


def test_intentionally_shared_owner_serializes_commands_across_sessions():
    camera = RecordingCamera()
    with OnvifController(TARGET, port_factory=lambda t: camera) as owner:
        compiled = plan()
        sessions = [
            compiled.create_session(resources={"onvif_controller": owner})
            for _ in range(2)
        ]
        with ThreadPoolExecutor(2) as pool:
            assert all(result["seeking"] for result in pool.map(run, sessions))
        assert camera.open_count == 1 and len(camera.commands) == 2
        for session in sessions:
            session.close()
        assert camera.close_count == 0


def test_host_quiesce_before_processing_reset_retires_delayed_work():
    camera, timer = RecordingCamera(), ManualSleep()
    with OnvifController(TARGET, port_factory=lambda t: camera, sleep=timer) as owner:
        compiled = plan(
            default_position_preset="home", move_to_position_after_idle_seconds=3
        )
        session = compiled.create_session(resources={"onvif_controller": owner})
        run(session, predictions(()))
        assert timer.started.wait(2)
        old = timer.waiters[-1][1]
        # No unfinished foreground runs here: host has stopped admissions/drained.
        owner.quiesce()
        receipt = session.apply_update(session.prepare_update(compiled, reset=True))
        assert old.cancelled() and receipt.graph_version == 1
        owner.resume()
        assert run(session)["seeking"]
        assert not camera.preset_called.is_set()
        session.close()


def test_engine_reset_alone_does_not_claim_host_timer_cancellation():
    camera, timer = RecordingCamera(), ManualSleep()
    with OnvifController(TARGET, port_factory=lambda t: camera, sleep=timer) as owner:
        compiled = plan(
            default_position_preset="home", move_to_position_after_idle_seconds=3
        )
        session = compiled.create_session(resources={"onvif_controller": owner})
        run(session, predictions(()))
        assert timer.started.wait(2)
        timer_future = timer.waiters[-1][1]
        session.apply_update(session.prepare_update(compiled, reset=True))
        assert not timer_future.done()
        owner.quiesce()
        assert timer_future.cancelled()
        session.close()


def test_retained_runs_keep_idle_preset_travelling_until_new_target():
    camera, timer = TravellingCamera(), ManualSleep()
    with OnvifController(TARGET, port_factory=lambda t: camera, sleep=timer) as owner:
        compiled = plan(
            default_position_preset="home", move_to_position_after_idle_seconds=3
        )
        session = compiled.create_session(resources={"onvif_controller": owner})
        run(session, predictions(()))
        assert timer.started.wait(2)
        timer.fire(owner)
        assert camera.preset_called.wait(2)
        assert run(session, predictions(()))["seeking"] is False
        assert run(session, predictions(((45, 45, 55, 55),)))["seeking"] is False
        assert camera.motion == "preset:home"
        assert camera.commands == [("stop",), ("preset", "home")]
        assert run(session)["seeking"] and camera.motion == "continuous"
        run(session, predictions(()))
        assert camera.motion is None and camera.commands[-1] == ("stop",)
        session.close()
    assert camera.commands[-2:] == [("stop",), ("close",)]


def test_retained_direct_preset_runs_do_not_abort_travel():
    camera, clock = TravellingCamera(), Clock()
    with OnvifController(TARGET, port_factory=lambda t: camera, clock=clock) as owner:
        compiled = plan(
            movement_type="Go To Preset",
            default_position_preset="home",
            camera_update_rate_limit=250,
        )
        session = compiled.create_session(resources={"onvif_controller": owner})
        for _ in range(3):
            assert run(session)["seeking"] is False
            clock.advance(0.033)
        assert camera.motion == "preset:home"
        assert camera.commands == [("stop",), ("preset", "home")]
        session.close()
    assert camera.motion is None and camera.commands[-2:] == [("stop",), ("close",)]


def test_compiled_error_and_disabled_missing_owner():
    session = plan().create_session()
    with pytest.raises(StepExecutionError, match="host-owned"):
        run(session)
    session.close()
    session = plan(disable_sink=True).create_session()
    assert run(session)["seeking"] is False
    session.close()


def test_retained_session_recovers_after_a_reported_command_error():
    camera = RecordingCamera()
    with OnvifController(TARGET, port_factory=lambda t: camera) as owner:
        session = plan().create_session(resources={"onvif_controller": owner})
        camera.fail.add("move")
        with pytest.raises(StepExecutionError) as caught:
            run(session)
        assert "sentinel-password" not in str(caught.value)
        camera.fail.clear()
        assert run(session)["seeking"]
        session.close()
    assert camera.open_count == camera.close_count == 1
