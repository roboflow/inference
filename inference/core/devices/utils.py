import logging
import subprocess

logging.basicConfig(level=logging.WARNING)


class CommandExecutionError(Exception):
    """Raised when there's a failure in command execution."""

    pass


class SerialNumberNotFoundError(Exception):
    """Raised when the Serial Number is not found."""

    pass


def get_trt_device_id():
    """Return the TRT device ID for the current host device by trying several methods.

    Tries different methods to get the TRT device ID.

    Returns:
        str: The TRT device ID.
    """
    try:
        return get_tegra_serial()
    except SerialNumberNotFoundError as e:
        logging.warning(f"Failed to retrieve Tegra serial number: {e}")

    try:
        return get_gpu_serial_num()
    except SerialNumberNotFoundError as e:
        logging.warning(f"Failed to retrieve GPU serial number: {e}")

    return get_gpu_serial_num_uid()


def run_command(cmd):
    """Executes a shell command and returns the output.

    Args:
        cmd (list): A command to execute.

    Returns:
        str: Output of the command execution.

    Raises:
        CommandExecutionError: If the command execution fails.
    """
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
        return result.stdout.decode("utf-8")
    except Exception as e:
        raise CommandExecutionError(f"Failed to execute command {cmd}: {e}")


def parse_output(stdout, keyword):
    """Parses the output for a specific keyword.

    Args:
        stdout (str): The output to parse.
        keyword (str): The keyword to search for.

    Returns:
        str: The value following the keyword.

    Raises:
        SerialNumberNotFoundError: If the keyword is not found in the output.
    """
    for line in stdout.split("\n"):
        if keyword in line:
            return line.split(": ")[1]
    raise SerialNumberNotFoundError(f"'{keyword}' not found in command output.")


def get_gpu_serial_num():
    """Gets the GPU serial number for the first GPU of the current host.

    Returns:
        str: The GPU serial number.
    """
    return parse_output(run_command(["nvidia-smi", "-q"]), "Serial Number")


def get_gpu_serial_num_uid():
    """Returns the GPU UUID for the first GPU of the current host.

    Returns:
        str: The GPU UUID.
    """
    return parse_output(run_command(["nvidia-smi", "-q"]), "GPU UUID")


def get_tegra_serial():
    """Returns the Tegra serial number for the first GPU of the current host.

    Returns:
        str: The Tegra serial number.
    """
    return parse_output(run_command(["lshw"]), "serial:")
