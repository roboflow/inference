import os
import sys
import contextlib
import functools


def touch(path):
    try:
        # update the access and modification times
        os.utime(path, None)
    except FileNotFoundError:
        # create the file if it doesn't exist
        with open(path, "w"):
            pass


@contextlib.contextmanager
def stream_redirected(stream_name, to=os.devnull, file_descriptor_mode=False):
    if file_descriptor_mode:
        stream = getattr(sys, stream_name)
        fd = stream.fileno()

        with open(to, "w") as file:
            saved_fd = os.dup(fd)
            os.dup2(file.fileno(), fd)
            try:
                yield
            finally:
                os.dup2(saved_fd, fd)
                os.close(saved_fd)
    else:
        with open(to, "w") as out:
            with getattr(contextlib, f"redirect_{stream_name}")(out):
                yield  # run code within context


stdout_redirected = functools.partial(stream_redirected, "stdout")
stderr_redirected = functools.partial(stream_redirected, "stderr")
