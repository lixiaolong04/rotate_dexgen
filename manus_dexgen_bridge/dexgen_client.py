"""Non-blocking latest-frame client for the GeoRT retargeting publisher."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

from manus_dexgen_bridge.protocol import RetargetFrame, decode_frame, parse_tcp_address


@dataclass(frozen=True)
class CommandSnapshot:
    frame: RetargetFrame | None
    status: str
    age_s: float | None
    source_age_s: float | None
    error: str | None
    received_frames: int
    sequence_gaps: int
    out_of_order_frames: int


class LatestCommandClient:
    """Reconnect in a daemon thread; the control thread reads only snapshots."""

    def __init__(
        self, address: str = "tcp://127.0.0.1:8020", *, stale_timeout_s: float = 0.2,
        reconnect_delay_s: float = 0.2,
    ) -> None:
        if stale_timeout_s <= 0.0 or reconnect_delay_s <= 0.0:
            raise ValueError("timeouts must be positive")
        self.host, self.port = parse_tcp_address(address)
        self.address = address
        self.stale_timeout_s = float(stale_timeout_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._lock = threading.Lock()
        self._latest: RetargetFrame | None = None
        self._received_at: float | None = None
        self._error: str | None = None
        self._received_frames = 0
        self._sequence_gaps = 0
        self._out_of_order_frames = 0
        self._running = True
        self._connection: socket.socket | None = None
        self._thread = threading.Thread(target=self._recv_loop, name="geort-command-client", daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        while self._running:
            connection: socket.socket | None = None
            try:
                connection = socket.create_connection((self.host, self.port), timeout=2.0)
                connection.settimeout(1.0)
                with self._lock:
                    self._connection = connection
                    self._error = None
                last_sequence: int | None = None
                while self._running:
                    frame = decode_frame(connection)
                    with self._lock:
                        if last_sequence is not None:
                            if frame.sequence > last_sequence + 1:
                                self._sequence_gaps += frame.sequence - last_sequence - 1
                            elif frame.sequence <= last_sequence:
                                self._out_of_order_frames += 1
                        last_sequence = frame.sequence
                        self._latest = frame
                        self._received_at = time.monotonic()
                        self._received_frames += 1
            except (ConnectionError, OSError, ValueError, KeyError, UnicodeDecodeError) as exc:
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                if self._running:
                    time.sleep(self.reconnect_delay_s)
            finally:
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass
                with self._lock:
                    if self._connection is connection:
                        self._connection = None

    def snapshot(self) -> CommandSnapshot:
        with self._lock:
            frame = self._latest
            received_at = self._received_at
            error = self._error
            count = self._received_frames
            gaps = self._sequence_gaps
            out_of_order = self._out_of_order_frames
        age_s = None if received_at is None else time.monotonic() - received_at
        source_age_s = (
            None if frame is None else max(0.0, (time.time_ns() - frame.timestamp_ns) / 1.0e9)
        )
        if error is not None and frame is None:
            status = "error"
        elif (
            frame is None
            or age_s is None
            or age_s > self.stale_timeout_s
            or source_age_s is None
            or source_age_s > self.stale_timeout_s
            or not frame.valid
        ):
            status = "stale"
        else:
            status = "recording"
        return CommandSnapshot(
            frame, status, age_s, source_age_s, error, count, gaps, out_of_order
        )

    def close(self) -> None:
        self._running = False
        with self._lock:
            connection = self._connection
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._thread.join(timeout=2.0)
