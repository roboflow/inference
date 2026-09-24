"""Import-light entry points of the stream manager's child processes.

Under the `spawn` start method a child imports the module its pickled target
or process class lives in before running any of it, so a target defined next
to the manager runtime would freeze the stream configuration at that import -
before anything the parent passed could be installed. The two entry points
here - `run_stream_manager` for the manager process a server launches, and
`PipelineManagerProcess` for each pipeline process the manager starts -
therefore live in a module that imports only the configuration and the host
contract. Each installs the configuration and host descriptor it was given
first, and imports the manager runtime only afterwards.
"""

import signal
from multiprocessing import Process, Queue

from inference.core.interfaces.stream.configuration import (
    StreamsConfiguration,
    configure_process,
)
from inference.core.interfaces.stream_manager.manager_app.host import (
    PipelineHostDescriptor,
    import_attribute,
    install_default_host_descriptor,
)


def install_process_settings(
    configuration: StreamsConfiguration,
    host_descriptor: PipelineHostDescriptor,
) -> None:
    """Install this process's stream configuration and default host descriptor.

    Args:
        configuration: Stream configuration of the process.
        host_descriptor: Descriptor of the pipeline host.

    Raises:
        StreamsConfigurationError: A different configuration is already in use.
    """
    configure_process(configuration)
    install_default_host_descriptor(host_descriptor)


def run_stream_manager(
    *,
    configuration: StreamsConfiguration,
    host_descriptor: PipelineHostDescriptor,
    expected_warmed_up_pipelines: int = 0,
) -> None:
    """Configure this process, then run the stream manager in it.

    Use as the `Process` target that launches the manager.

    Args:
        configuration: Stream configuration of the manager and its pipelines.
        host_descriptor: Descriptor of the pipeline host.
        expected_warmed_up_pipelines: Number of idle pipeline processes kept ready.
    """
    install_process_settings(
        configuration=configuration,
        host_descriptor=host_descriptor,
    )

    from inference.core.interfaces.stream_manager.manager_app.app import start

    start(
        expected_warmed_up_pipelines=expected_warmed_up_pipelines,
        host_descriptor=host_descriptor,
    )


class PipelineManagerProcess(Process):
    """A pipeline process: configures itself, then runs the pipeline manager.

    Attributes are plain values and queues only, so the object pickles under
    any start method. `manager_class` names the pipeline manager class by
    import path for the same reason.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        command_queue: Queue,
        responses_queue: Queue,
        configuration: StreamsConfiguration,
        host_descriptor: PipelineHostDescriptor,
        manager_class: str,
    ):
        super().__init__()
        self._pipeline_id = pipeline_id
        self._command_queue = command_queue
        self._responses_queue = responses_queue
        self._configuration = configuration
        self._host_descriptor = host_descriptor
        self._manager_class = manager_class

    def run(self) -> None:
        # The pipeline manager ignores SIGINT (the stream manager terminates it
        # with SIGTERM); do so already while the runtime is being imported.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        install_process_settings(
            configuration=self._configuration,
            host_descriptor=self._host_descriptor,
        )

        manager_class = import_attribute(self._manager_class)
        # Imports the host's modules while the process is still idle, as a
        # forked process inherits them, so the first request does not.
        import_attribute(self._host_descriptor.factory)

        manager = manager_class.init(
            pipeline_id=self._pipeline_id,
            command_queue=self._command_queue,
            responses_queue=self._responses_queue,
            host_descriptor=self._host_descriptor,
        )
        manager.run()
