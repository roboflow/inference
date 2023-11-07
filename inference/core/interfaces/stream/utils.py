from typing import Union, Optional


def translate_stream_reference(
    stream_reference: Optional[Union[str, int]]
) -> Union[str, int]:
    if stream_reference is None:
        raise ValueError(
            "`stream_reference` is not defined. If your intention was to use `STREAM_ID` env "
            "variable - check if it is exported properly."
        )
    if stream_reference == "webcam":
        return 0
    return stream_reference
