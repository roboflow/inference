import importlib
import json
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference_cli.lib import container_adapter, podman_adapter
from inference_cli.lib.exceptions import DockerConnectionErrorException
from inference_cli.lib.podman_adapter import (
    PodmanGPUNotConfiguredError,
    build_podman_launch_command,
    find_cdi_spec,
    find_running_podman_inference_containers,
    pull_image_with_podman,
)


def _ps_row(
    image: str,
    state: str = "running",
    cid: str = "abc123",
    name: str = "inference-server",
) -> str:
    return json.dumps(
        {
            "Id": cid,
            "Names": [name],
            "Image": image,
            "State": state,
            "Created": "2026-09-15T15:29:09",
            "Ports": [
                {"host_ip": "127.0.0.1", "container_port": 9001, "host_port": 9001}
            ],
        }
    )


class TestDetectContainerRuntime:
    def setup_method(self) -> None:
        self._env = mock.patch.dict(
            container_adapter.os.environ,
            {container_adapter.CONTAINER_RUNTIME_ENV_VAR: ""},
        )
        self._env.start()

    def teardown_method(self) -> None:
        self._env.stop()

    def test_env_override_selects_podman(self) -> None:
        with mock.patch.dict(
            container_adapter.os.environ,
            {container_adapter.CONTAINER_RUNTIME_ENV_VAR: "podman"},
        ), mock.patch.object(podman_adapter, "podman_is_installed", return_value=True):
            assert (
                container_adapter.detect_container_runtime()
                == container_adapter.CONTAINER_RUNTIME_PODMAN
            )

    def test_env_override_rejects_unknown_runtime(self) -> None:
        with mock.patch.dict(
            container_adapter.os.environ,
            {container_adapter.CONTAINER_RUNTIME_ENV_VAR: "nerdctl"},
        ):
            with pytest.raises(DockerConnectionErrorException):
                container_adapter.detect_container_runtime()

    def test_podman_override_requires_binary(self) -> None:
        with mock.patch.dict(
            container_adapter.os.environ,
            {container_adapter.CONTAINER_RUNTIME_ENV_VAR: "podman"},
        ), mock.patch.object(podman_adapter, "podman_is_installed", return_value=False):
            with pytest.raises(DockerConnectionErrorException):
                container_adapter.detect_container_runtime()

    def test_docker_override_requires_daemon(self) -> None:
        with mock.patch.dict(
            container_adapter.os.environ,
            {container_adapter.CONTAINER_RUNTIME_ENV_VAR: "docker"},
        ), mock.patch("docker.from_env") as from_env_mock:
            import docker as real_docker

            from_env_mock.side_effect = real_docker.errors.DockerException()
            with pytest.raises(DockerConnectionErrorException):
                container_adapter.detect_container_runtime()

    @mock.patch.object(container_adapter, "docker")
    def test_plain_docker_endpoint_returns_docker(self, docker_mock: MagicMock) -> None:
        docker_mock.from_env.return_value.version.return_value = {
            "Components": [{"Name": "Engine"}]
        }
        assert (
            container_adapter.detect_container_runtime()
            == container_adapter.CONTAINER_RUNTIME_DOCKER
        )

    @mock.patch.object(container_adapter, "docker")
    def test_podman_compat_socket_detected_as_podman(
        self, docker_mock: MagicMock
    ) -> None:
        docker_mock.from_env.return_value.version.return_value = {
            "Components": [{"Name": "Podman Engine"}]
        }
        # The compat-socket path requires a local podman binary (the CLI shells
        # out to it), so stub the check here.
        with mock.patch.object(
            podman_adapter, "podman_is_installed", return_value=True
        ):
            assert (
                container_adapter.detect_container_runtime()
                == container_adapter.CONTAINER_RUNTIME_PODMAN
            )

    @mock.patch.object(container_adapter, "docker")
    def test_podman_compat_socket_without_local_binary_fails(
        self, docker_mock: MagicMock
    ) -> None:
        docker_mock.from_env.return_value.version.return_value = {
            "Components": [{"Name": "Podman Engine"}]
        }
        with mock.patch.object(
            podman_adapter, "podman_is_installed", return_value=False
        ):
            with pytest.raises(DockerConnectionErrorException):
                container_adapter.detect_container_runtime()

    @mock.patch.object(podman_adapter, "podman_is_installed", return_value=True)
    @mock.patch("docker.from_env")
    def test_no_docker_endpoint_with_podman_binary(
        self, from_env_mock: MagicMock, _installed_mock: MagicMock
    ) -> None:
        docker_mock = importlib.import_module("docker")
        from_env_mock.side_effect = docker_mock.errors.DockerException()
        assert (
            container_adapter.detect_container_runtime()
            == container_adapter.CONTAINER_RUNTIME_PODMAN
        )

    @mock.patch.object(podman_adapter, "podman_is_installed", return_value=False)
    @mock.patch("docker.from_env")
    def test_no_runtime_raises_connection_error(
        self, from_env_mock: MagicMock, _installed_mock: MagicMock
    ) -> None:
        import docker as real_docker

        from_env_mock.side_effect = real_docker.errors.DockerException()
        with pytest.raises(DockerConnectionErrorException):
            container_adapter.detect_container_runtime()


class TestGpuLaunchExtras:
    def test_gpu_command_adds_sys_admin_and_keep_groups(self) -> None:
        with mock.patch.object(
            podman_adapter,
            "find_cdi_spec",
            return_value=(mock.MagicMock(), "nvidia.com/gpu=all"),
        ), mock.patch.object(
            podman_adapter, "_warn_if_selinux_devices_denied", return_value=None
        ):
            command, _ = build_podman_launch_command(
                image="roboflow/roboflow-inference-server-gpu:latest",
                development=False,
                environment=["PORT=9001"],
                extra_environment=[],
                bind_address="127.0.0.1",
                port=9001,
                volumes={},
                device_requests=["nvidia.com/gpu=all"],
            )
        assert "--cap-add" in command and "SYS_ADMIN" in command
        assert "keep-groups" in command


class TestPodmanContainerLogs:
    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_logs_returns_combined_streams(self, run_mock: MagicMock) -> None:
        run_mock.return_value = MagicMock(stdout=b"ok", stderr=b"")
        container = podman_adapter.PodmanContainer(id="c1", attrs={}, image_tags=[])
        assert container.logs(tail=10) == b"ok"
        run_mock.assert_called_once_with(
            ["podman", "logs", "--tail", "10", "c1"], check=True, capture_output=True
        )


class TestFindRunningPodmanInferenceContainers:
    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_running_inference_server_is_matched(self, run_mock: MagicMock) -> None:
        run_mock.return_value = MagicMock(
            stdout=_ps_row("docker.io/roboflow/roboflow-inference-server-gpu:latest")
        )
        containers = find_running_podman_inference_containers()
        assert len(containers) == 1
        assert containers[0].id == "abc123"
        assert containers[0].attrs["Name"] == "inference-server"
        assert "9001/tcp" in containers[0].attrs["Config"]["ExposedPorts"]

    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_null_ports_row_does_not_crash(self, run_mock: MagicMock) -> None:
        # `podman ps --format {{json .}}` emits "Ports": null for containers
        # without published ports (e.g. started with --network host).
        row = json.loads(
            _ps_row("docker.io/roboflow/roboflow-inference-server-cpu:latest")
        )
        row["Ports"] = None
        run_mock.return_value = MagicMock(stdout=json.dumps(row))
        containers = find_running_podman_inference_containers()
        assert len(containers) == 1
        assert containers[0].attrs["Config"]["Env"] == []
        assert containers[0].attrs["Config"]["ExposedPorts"] == {}

    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_env_port_comes_from_published_host_port(self, run_mock: MagicMock) -> None:
        # When `podman inspect` yields no usable Config.Env, the port shown by
        # the shared "already running" prompt falls back to Ports[].host_port.
        row = json.loads(
            _ps_row("docker.io/roboflow/roboflow-inference-server-cpu:latest")
        )
        row["Ports"] = [
            {"host_ip": "127.0.0.1", "container_port": 9001, "host_port": 9123}
        ]
        # ps succeeds; inspect returns a document without Config.Env
        run_mock.side_effect = [
            MagicMock(stdout=json.dumps(row) + "\n"),
            MagicMock(stdout=json.dumps({"Id": "abc123"})),
        ]
        containers = find_running_podman_inference_containers()
        assert containers[0].attrs["Config"]["Env"] == ["PORT=9123"]

    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_env_port_prefers_inspect_config(self, run_mock: MagicMock) -> None:
        # `podman inspect` is authoritative for Config.Env; the Ports-based
        # guess must not shadow it (e.g. dev mode publishing 9001 and 9002).
        ps_row = json.loads(
            _ps_row("docker.io/roboflow/roboflow-inference-server-cpu:latest")
        )
        ps_row["Ports"] = [
            {"host_ip": "127.0.0.1", "container_port": 9002, "host_port": 9002},
            {"host_ip": "127.0.0.1", "container_port": 9001, "host_port": 9123},
        ]
        inspect_doc = {"Id": "abc123", "Config": {"Env": ["PORT=9123", "FOO=bar"]}}
        run_mock.side_effect = [
            MagicMock(stdout=json.dumps(ps_row) + "\n"),
            MagicMock(stdout=json.dumps(inspect_doc)),
        ]
        containers = find_running_podman_inference_containers()
        assert containers[0].attrs["Config"]["Env"] == ["PORT=9123", "FOO=bar"]

    @mock.patch.object(
        container_adapter, "detect_container_runtime", return_value="podman"
    )
    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_status_survives_container_without_published_ports(
        self, run_mock: MagicMock, _runtime_mock: MagicMock, capsys
    ) -> None:
        # --network host containers report "Ports": null; status must print a
        # placeholder instead of indexing an empty exposed-port mapping.
        row = json.loads(
            _ps_row("docker.io/roboflow/roboflow-inference-server-cpu:latest")
        )
        row["Ports"] = None
        run_mock.return_value = MagicMock(stdout=json.dumps(row))
        container_adapter.check_inference_server_status()
        assert "not published" in capsys.readouterr().out

    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_short_image_name_without_registry_is_matched(
        self, run_mock: MagicMock
    ) -> None:
        run_mock.return_value = MagicMock(
            stdout=_ps_row("roboflow/roboflow-inference-server-cpu:latest")
        )
        assert len(find_running_podman_inference_containers()) == 1

    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_exited_and_foreign_containers_are_skipped(
        self, run_mock: MagicMock
    ) -> None:
        run_mock.return_value = MagicMock(
            stdout="\n".join(
                [
                    _ps_row(
                        "roboflow/roboflow-inference-server-cpu:latest", state="exited"
                    ),
                    _ps_row("docker.io/library/nginx:latest", cid="other"),
                ]
            )
        )
        assert find_running_podman_inference_containers() == []

    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_kill_invokes_podman(self, run_mock: MagicMock) -> None:
        run_mock.return_value = MagicMock(
            stdout=_ps_row("docker.io/roboflow/roboflow-inference-server-gpu:latest")
        )
        containers = find_running_podman_inference_containers()
        run_mock.reset_mock()
        containers[0].kill()
        run_mock.assert_called_once_with(
            ["podman", "kill", "abc123"], check=True, capture_output=True, text=True
        )


class TestFindCdiSpec:
    def test_default_search_directories_include_etc_cdi(self) -> None:
        directories = [str(d) for d in podman_adapter._cdi_search_directories()]
        assert "/etc/cdi" in directories

    def test_per_user_directory_is_not_scanned(self) -> None:
        # Podman only scans its configured cdi_spec_dirs (default /etc/cdi and
        # /var/run/cdi); a spec under $XDG_DATA_HOME would never be honoured.
        directories = [str(d) for d in podman_adapter._cdi_search_directories()]
        assert all(".local" not in d and "XDG" not in d for d in directories)

    def test_nvidia_kind_spec_yields_all_devices(self, tmp_path) -> None:
        spec = tmp_path / "nvidia.yaml"
        spec.write_text('---\ncdiVersion: "0.7.0"\nkind: nvidia.com/gpu\ndevices: []\n')
        with mock.patch.object(
            podman_adapter,
            "_cdi_search_directories",
            return_value=[tmp_path],
        ):
            found = find_cdi_spec()
        assert found is not None
        _, device = found
        assert device == "nvidia.com/gpu=all"

    def test_non_nvidia_spec_is_ignored(self, tmp_path) -> None:
        spec = tmp_path / "amd.yaml"
        spec.write_text("kind: amd.com/gpu\ndevices: []\n")
        with mock.patch.object(
            podman_adapter, "_cdi_search_directories", return_value=[tmp_path]
        ):
            assert find_cdi_spec() is None

    def test_nvidia_named_spec_of_other_kind_is_ignored(self, tmp_path) -> None:
        # CDI device references are always vendor/class=device; only the
        # nvidia.com/gpu kind can serve the GPU requirement.
        spec = tmp_path / "nvidia-i915.yaml"
        spec.write_text('{"kind": "intel.com/gpu"}\n')
        with mock.patch.object(
            podman_adapter, "_cdi_search_directories", return_value=[tmp_path]
        ):
            assert find_cdi_spec() is None


class TestBuildPodmanLaunchCommand:
    def test_docker_hub_short_names_are_qualified_for_pull(self) -> None:
        # Fedora enforces podman short-name resolution: a bare
        # `roboflow/...` pull fails without a TTY, so the CLI must qualify it.
        assert (
            podman_adapter._resolve_pull_ref(
                "roboflow/roboflow-inference-server-cpu:latest"
            )
            == "docker.io/roboflow/roboflow-inference-server-cpu:latest"
        )
        assert (
            podman_adapter._resolve_pull_ref("quay.io/ns/img:1") == "quay.io/ns/img:1"
        )
        assert podman_adapter._resolve_pull_ref("localhost/img:1") == "localhost/img:1"
        assert (
            podman_adapter._resolve_pull_ref("registry.local:5000/img:1")
            == "registry.local:5000/img:1"
        )

    @mock.patch.object(podman_adapter.subprocess, "run")
    def test_pull_qualifies_short_names(self, run_mock: MagicMock) -> None:
        run_mock.return_value = MagicMock(returncode=1)  # image not present
        pull_image_with_podman("roboflow/roboflow-inference-server-cpu:latest")
        assert run_mock.call_args_list[0].args[0] == [
            "podman",
            "image",
            "exists",
            "docker.io/roboflow/roboflow-inference-server-cpu:latest",
        ]

    def _args(self, device_requests=None, development=False):
        return dict(
            image="roboflow/roboflow-inference-server-gpu:latest",
            development=development,
            environment=["PORT=9001"],
            extra_environment=["HOME=/tmp/home"],
            bind_address="127.0.0.1",
            port=9001,
            volumes={"/tmp": {"bind": "/tmp", "mode": "rw"}},
            device_requests=device_requests,
        )

    def test_gpu_requires_cdi_spec(self) -> None:
        with mock.patch.object(podman_adapter, "find_cdi_spec", return_value=None):
            with pytest.raises(PodmanGPUNotConfiguredError):
                build_podman_launch_command(**self._args(device_requests=["x"]))

    def test_gpu_uses_cdi_device_flag(self) -> None:
        with mock.patch.object(
            podman_adapter,
            "find_cdi_spec",
            return_value=(mock.MagicMock(), "nvidia.com/gpu=all"),
        ):
            command, mode = build_podman_launch_command(
                **self._args(device_requests=["nvidia.com/gpu=all"])
            )
        assert mode == "cdi"
        assert "--device" in command
        assert command[command.index("--device") + 1] == "nvidia.com/gpu=all"

    def test_security_hardening_translated(self) -> None:
        command, mode = build_podman_launch_command(**self._args())
        assert mode == "none"
        assert command[0] == "podman" and command[1] == "run"
        assert command[-1] == self._args()["image"]
        assert "--read-only" in command
        idx = command.index("--security-opt")
        assert "label=disable" in command[idx + 1 :]
        assert "--security-opt" in command
        assert "no-new-privileges" in command
        assert "--cap-drop" in command and "ALL" in command
        assert "--cap-add" in command and "NET_BIND_SERVICE" in command

    def test_loopback_publishing_and_development_ports(self) -> None:
        command, _ = build_podman_launch_command(**self._args(development=True))
        assert "127.0.0.1:9001:9001" in command
        assert "127.0.0.1:9002:9002" in command

    def test_environment_and_volumes_forwarded(self) -> None:
        command, _ = build_podman_launch_command(**self._args())
        assert "PORT=9001" in command
        assert "HOME=/tmp/home" in command
        assert "-v" in command and "/tmp:/tmp:rw" in command


class TestPodmanDispatch:
    @mock.patch.object(
        container_adapter, "detect_container_runtime", return_value="podman"
    )
    @mock.patch.object(podman_adapter, "launch_inference_container_with_podman")
    @mock.patch.object(container_adapter, "pull_image")
    @mock.patch.object(
        container_adapter, "find_running_inference_containers", return_value=[]
    )
    @mock.patch.object(container_adapter, "docker")
    def test_start_uses_podman_and_never_docker_sdk(
        self,
        docker_mock: MagicMock,
        _find_mock: MagicMock,
        _pull_mock: MagicMock,
        launch_mock: MagicMock,
        _runtime_mock: MagicMock,
    ) -> None:
        container_adapter.start_inference_container(
            image="roboflow/roboflow-inference-server-gpu:latest",
            labels=["managed-by=inference-cli"],
        )
        launch_mock.assert_called_once()
        assert launch_mock.call_args.kwargs["require_gpu"] is True
        # list-form labels must not be silently dropped on the podman path
        assert launch_mock.call_args.kwargs["labels"] == ["managed-by=inference-cli"]
        docker_mock.from_env.assert_not_called()

    @mock.patch.object(
        container_adapter,
        "detect_container_runtime",
        return_value=container_adapter.CONTAINER_RUNTIME_PODMAN,
    )
    @mock.patch.object(podman_adapter, "launch_inference_container_with_podman")
    @mock.patch.object(container_adapter, "pull_image")
    @mock.patch.object(
        container_adapter, "find_running_inference_containers", return_value=[]
    )
    def test_jetson_image_fails_loudly_on_podman(
        self,
        _find_mock: MagicMock,
        _pull_mock: MagicMock,
        _launch_mock: MagicMock,
        _runtime_mock: MagicMock,
    ) -> None:
        with pytest.raises(podman_adapter.PodmanJetsonUnsupportedError):
            container_adapter.start_inference_container(
                image="roboflow/roboflow-inference-server-jetson-6.2.0:latest"
            )

    @mock.patch.object(
        container_adapter, "detect_container_runtime", return_value="podman"
    )
    @mock.patch.object(podman_adapter, "pull_image_with_podman")
    @mock.patch.object(container_adapter, "docker")
    def test_pull_uses_podman(
        self, docker_mock: MagicMock, pull_mock: MagicMock, _runtime_mock: MagicMock
    ) -> None:
        container_adapter.pull_image("roboflow/roboflow-inference-server-cpu:latest")
        pull_mock.assert_called_once()
        docker_mock.from_env.assert_not_called()

    @mock.patch.object(
        container_adapter, "detect_container_runtime", return_value="podman"
    )
    @mock.patch.object(podman_adapter, "find_running_podman_inference_containers")
    def test_find_delegates_to_podman(
        self, find_mock: MagicMock, _runtime_mock: MagicMock
    ) -> None:
        find_mock.return_value = []
        assert container_adapter.find_running_inference_containers() == []
        find_mock.assert_called_once()
