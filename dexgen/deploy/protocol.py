"""Length-prefixed JSON protocol between the CUDA controller and SDK container."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


MAX_MESSAGE_BYTES = 256 * 1024


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        data.extend(chunk)
    return bytes(data)


def send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not encoded or len(encoded) > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message size: {len(encoded)}")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)


def recv_message(sock: socket.socket) -> dict[str, Any]:
    (size,) = struct.unpack("!I", recv_exact(sock, 4))
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message size: {size}")
    payload = json.loads(recv_exact(sock, size).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol message must be a JSON object")
    return payload
