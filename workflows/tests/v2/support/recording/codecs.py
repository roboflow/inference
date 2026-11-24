"""An in-memory blob store and a point payload codec for recording tests."""

import zlib
from typing import Any, List

from roboflow_workflows.v2.engine.recording import PayloadCodec, type_name_of


class MemoryBlobs:
    """In-memory blob sink and source with the store's reference shape."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.puts: List[bytes] = []

    def put(self, data) -> list:
        raw = bytes(data)
        self.puts.append(raw)
        ref = [len(self.data), len(raw), zlib.crc32(raw)]
        self.data += raw
        return ref

    def get(self, ref) -> bytearray:
        offset, length, _ = ref
        return bytearray(self.data[offset : offset + length])


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y = x, y


def encode_point(point: Point, sink, encoder) -> Any:
    return {"xy": encoder.encode((point.x, point.y), path=".xy")}


def decode_point(node: Any, source, decoder) -> Point:
    x, y = decoder.decode(node["xy"], path=".xy")
    return Point(x, y)


POINT_CODEC = PayloadCodec(
    name="test/point@1",
    type_name=type_name_of(Point),
    encode=encode_point,
    decode=decode_point,
)
