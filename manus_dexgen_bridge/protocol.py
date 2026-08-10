"""Small, versioned TCP protocol for XHand retargeting targets.

Each message is ``uint32_be(header_bytes) + JSON header + 21 float32_le``.
The fixed binary joint payload keeps the realtime path compact while JSON makes
the representation contract explicit and easy to inspect in a packet capture.
"""

from __future__ import annotations

import json
import socket
import struct
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np


SCHEMA = "geort.xhand_retarget.v1"
JOINT_COUNT = 21
PAYLOAD_BYTES = JOINT_COUNT * np.dtype("<f4").itemsize
MAX_HEADER_BYTES = 16 * 1024


def parse_tcp_address(address: str) -> tuple[str, int]:
    if not address.startswith("tcp://"):
        raise ValueError(f"address must start with tcp://, got {address!r}")
    host, separator, port = address[6:].rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid TCP address: {address!r}")
    port_int = int(port)
    if not 0 < port_int < 65536:
        raise ValueError(f"invalid TCP port: {port_int}")
    return host, port_int


def recv_exact(connection: socket.socket, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = connection.recv(length - len(chunks))
        if not chunk:
            raise ConnectionError("bridge peer closed the TCP connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _validate_joint_names(joint_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in joint_names)
    if len(names) != JOINT_COUNT or len(set(names)) != JOINT_COUNT:
        raise ValueError("joint_names must contain exactly 21 unique names")
    return names


@dataclass(frozen=True)
class RetargetFrame:
    sequence: int
    timestamp_ns: int
    valid: bool
    joint_names: tuple[str, ...]
    q_target: np.ndarray

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.timestamp_ns < 0:
            raise ValueError("sequence and timestamp_ns must be non-negative")
        _validate_joint_names(self.joint_names)
        q_target = np.asarray(self.q_target, dtype=np.float32)
        if q_target.shape != (JOINT_COUNT,) or not np.isfinite(q_target).all():
            raise ValueError("q_target must be a finite float vector shaped [21]")
        object.__setattr__(self, "q_target", q_target.copy())

    @classmethod
    def create(
        cls, sequence: int, joint_names: Sequence[str], q_target: np.ndarray,
        *, valid: bool = True, timestamp_ns: int | None = None,
    ) -> "RetargetFrame":
        return cls(
            sequence=int(sequence),
            timestamp_ns=time.time_ns() if timestamp_ns is None else int(timestamp_ns),
            valid=bool(valid),
            joint_names=_validate_joint_names(joint_names),
            q_target=q_target,
        )


def encode_frame(frame: RetargetFrame) -> bytes:
    header = {
        "schema": SCHEMA,
        "sequence": frame.sequence,
        "timestamp_ns": frame.timestamp_ns,
        "valid": frame.valid,
        "joint_names": list(frame.joint_names),
        "shape": [JOINT_COUNT],
        "dtype": "<f4",
    }
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded_header) > MAX_HEADER_BYTES:
        raise ValueError("frame header exceeds maximum size")
    return struct.pack("!I", len(encoded_header)) + encoded_header + frame.q_target.astype("<f4", copy=False).tobytes()


def decode_frame(connection: socket.socket) -> RetargetFrame:
    (header_size,) = struct.unpack("!I", recv_exact(connection, 4))
    if header_size == 0 or header_size > MAX_HEADER_BYTES:
        raise ValueError(f"invalid bridge header size: {header_size}")
    header = json.loads(recv_exact(connection, header_size).decode("utf-8"))
    if header.get("schema") != SCHEMA:
        raise ValueError(f"unexpected bridge schema: {header.get('schema')!r}")
    if header.get("shape") != [JOINT_COUNT] or header.get("dtype") != "<f4":
        raise ValueError("unexpected q_target array contract")
    q_target = np.frombuffer(recv_exact(connection, PAYLOAD_BYTES), dtype="<f4").copy()
    return RetargetFrame.create(
        header["sequence"], header["joint_names"], q_target,
        valid=header["valid"], timestamp_ns=header["timestamp_ns"],
    )
