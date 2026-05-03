"""Minimal WebSocket client for the OpenClaw gateway protocol.

Replaces the prior ``openclaw agent --local`` subprocess shell-out with
direct WS RPC calls to the gateway running on ``127.0.0.1:18789``. This
is what enables real mid-turn push parity via ``sessions.steer`` —
``--local`` was documented as a one-shot embedded agent that bypasses
the gateway entirely (per docs/cli/agent.md), so it had no in-flight
session to interrupt.

Scope: only the methods the executor needs — ``connect``, ``sessions.send``,
``sessions.steer``, ``agent.wait``, ``chat.history``. Not a full SDK port.

Per docs/gateway/operator-scopes.md, the gateway's ``operator`` role
covers control-plane callers (CLI, automation, trusted helpers). It's
distinct from the ``node`` role that requires UI-paired devices — we
don't need pairing here. The container-side gateway is started with
``openclaw gateway --dev --bind loopback`` so connect on ``127.0.0.1``
satisfies trusted-loopback auth without a token.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class GatewayError(Exception):
    """Raised when a gateway RPC returns ``ok: false``."""

    def __init__(self, method: str, code: str, message: str) -> None:
        super().__init__(f"{method}: {code} — {message}")
        self.method = method
        self.code = code
        self.message = message


class GatewayClient:
    """One persistent WebSocket connection to the OpenClaw gateway.

    Caller pattern::

        gw = GatewayClient(url="ws://127.0.0.1:18789/")
        await gw.connect(scopes=["operator.write"])
        runId = (await gw.request("sessions.send", {...}))["runId"]
        await gw.close()
    """

    # Protocol version — the gateway accepts a min/max range during
    # connect (see docs/gateway/protocol.md handshake). v3 is current.
    PROTOCOL = 3

    def __init__(
        self,
        url: str,
        *,
        client_id: str = "molecule-runtime-openclaw",
        client_version: str = "0.1.0",
    ) -> None:
        self._url = url
        self._client_id = client_id
        self._client_version = client_version
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._pending: Dict[str, asyncio.Future] = {}
        self._connected_event = asyncio.Event()
        self._closing = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        *,
        scopes: list[str] | None = None,
        role: str = "operator",
        timeout: float = 30.0,
    ) -> None:
        """Open the WS + complete the connect handshake.

        Idempotent — calling twice on the same client is a no-op after
        the first ``hello-ok`` lands.
        """
        if self._connected_event.is_set():
            return
        if self._session is None:
            self._session = aiohttp.ClientSession()
        # Open WS — the gateway sends ``connect.challenge`` first, but
        # we ignore the nonce (no token-binding required for the
        # loopback-trusted operator role we use).
        self._ws = await self._session.ws_connect(self._url, heartbeat=15.0)
        self._reader_task = asyncio.create_task(
            self._reader_loop(), name="openclaw-gateway-reader"
        )
        await self.request(
            "connect",
            {
                "minProtocol": self.PROTOCOL,
                "maxProtocol": self.PROTOCOL,
                "client": {
                    "id": self._client_id,
                    "version": self._client_version,
                    "platform": "linux",
                    "mode": "operator",
                },
                "role": role,
                "scopes": scopes or ["operator.write"],
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {},
                "userAgent": f"{self._client_id}/{self._client_version}",
            },
            timeout=timeout,
        )
        self._connected_event.set()

    async def close(self) -> None:
        """Tear down the WS + reader task. Idempotent."""
        self._closing = True
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader_task
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._session is not None:
            await self._session.close()
        # Fail any in-flight requests so callers don't hang.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("gateway closed"))
        self._pending.clear()
        self._connected_event.clear()

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    async def request(
        self,
        method: str,
        params: Dict[str, Any] | None = None,
        *,
        timeout: float | None = 30.0,
    ) -> Dict[str, Any]:
        """Send a JSON-RPC-style ``req`` frame and await the matching ``res``.

        Raises ``GatewayError`` on ``ok: false``, ``ConnectionError`` if
        the WS is gone, ``asyncio.TimeoutError`` if the response doesn't
        land within ``timeout``.
        """
        if self._ws is None or self._ws.closed:
            raise ConnectionError("gateway WS not connected")
        req_id = uuid.uuid4().hex
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        frame = {
            "type": "req",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        await self._ws.send_str(json.dumps(frame))
        try:
            if timeout is None:
                return await fut
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    # ------------------------------------------------------------------
    # Reader loop
    # ------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    frame = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("gateway: dropped malformed frame")
                    continue
                ftype = frame.get("type")
                if ftype == "res":
                    self._dispatch_res(frame)
                elif ftype == "event":
                    # Events are surface noise for the executor's
                    # purposes — agent.wait + chat.history give us the
                    # response shape we need without subscribing to
                    # session.message broadcasts. Future work could
                    # surface streaming deltas via a callback.
                    pass
                elif ftype == "ping":
                    # Keep the keepalive cycle visible in the protocol
                    # log without spamming.
                    logger.debug("gateway ping")
                # Ignore other frame types (invoke / pong / pre-connect challenge).
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, ConnectionResetError) as exc:
            logger.warning("gateway WS reader: %s", exc)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("gateway WS closed"))
            self._pending.clear()

    def _dispatch_res(self, frame: Dict[str, Any]) -> None:
        req_id = frame.get("id")
        if not isinstance(req_id, str):
            return
        fut = self._pending.get(req_id)
        if fut is None or fut.done():
            return
        if frame.get("ok") is False:
            err = frame.get("error") or {}
            fut.set_exception(
                GatewayError(
                    method=str(err.get("method", "?")),
                    code=str(err.get("code", "ERR")),
                    message=str(err.get("message", "")),
                )
            )
        else:
            fut.set_result(frame.get("payload") or {})
