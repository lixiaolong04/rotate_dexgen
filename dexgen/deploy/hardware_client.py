"""Synchronous localhost client for the XHand SDK hardware service."""

from __future__ import annotations

import socket
from typing import Sequence

import numpy as np

from dexgen.deploy.protocol import recv_message, send_message


class XHandHardwareClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8031,
        *,
        timeout_s: float = 2.0,
    ) -> None:
        self.sock = None
        try:
            self.sock = socket.create_connection(
                (host, int(port)), timeout=timeout_s
            )
            self.sock.settimeout(timeout_s)
            hello = self.request("ping")
            if hello.get("status") != "ok":
                raise RuntimeError(f"hardware service rejected ping: {hello}")
        except Exception:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            raise

    def request(
        self,
        operation: str,
        *,
        target: Sequence[float] | np.ndarray | None = None,
    ) -> dict:
        if self.sock is None:
            raise RuntimeError("hardware client is closed")
        request = {"op": str(operation)}
        if target is not None:
            vector = np.asarray(target, dtype=np.float32).reshape(-1)
            if vector.shape != (21,) or not np.isfinite(vector).all():
                raise ValueError("hardware target must be a finite 21-vector")
            request["target"] = vector.tolist()
        send_message(self.sock, request)
        response = recv_message(self.sock)
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", f"hardware request failed: {response}"))
        return response

    def state(self) -> dict:
        return self.request("state")

    def target(self, target: Sequence[float] | np.ndarray) -> dict:
        return self.request("target", target=target)

    def hold(self) -> dict:
        return self.request("hold")

    def zero_tactile(self) -> dict:
        return self.request("zero_tactile")

    def relax(self) -> None:
        try:
            self.request("relax")
        except (ConnectionError, OSError):
            pass

    def close(self, *, relax: bool = True) -> None:
        if self.sock is None:
            return
        try:
            if relax:
                self.relax()
        finally:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> "XHandHardwareClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(relax=True)
