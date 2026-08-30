import contextlib
import inspect
import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.exceptions
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

# Connection-level failures, which a reconnect can recover. An error carried inside a server
# response is not one of these and is never retried.
_TRANSIENT_ERRORS = (websockets.exceptions.WebSocketException, OSError, TimeoutError)

# websockets added keepalive to the threading client in 14.0, which needs Python 3.9. The LIBERO
# client venv is Python 3.8 and resolves 13.1, where connect() rejects the argument.
_CONNECT_TAKES_PING_TIMEOUT = "ping_timeout" in inspect.signature(websockets.sync.client.connect).parameters


class PolicyServerUnreachable(RuntimeError):
    """The server did not answer a request and reconnecting did not recover it."""


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.

    Args:
        ping_timeout: seconds this side's keepalive waits for a pong before dropping the
            connection, where the installed websockets supports it. None disables the deadline,
            so a server that stalls mid-request is waited out rather than hung up on.
        connect_timeout_s: seconds to keep retrying a connection, both at construction and when
            reconnecting mid-run. None retries forever.
        max_retries: how many times `infer` reconnects and resends a request before raising
            PolicyServerUnreachable.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        ping_timeout: Optional[float] = None,
        connect_timeout_s: Optional[float] = 300.0,
        max_retries: int = 3,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ping_timeout = ping_timeout
        self._connect_timeout_s = connect_timeout_s
        self._max_retries = max_retries
        self._reconnects = 0
        self._ws = None
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        deadline = None if self._connect_timeout_s is None else time.time() + self._connect_timeout_s
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                options = {"compression": None, "max_size": None, "additional_headers": headers}
                if _CONNECT_TAKES_PING_TIMEOUT:
                    options["ping_timeout"] = self._ping_timeout
                conn = websockets.sync.client.connect(self._uri, **options)
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except _TRANSIENT_ERRORS as e:
                if deadline is not None and time.time() >= deadline:
                    raise PolicyServerUnreachable(
                        f"No policy server at {self._uri} after {self._connect_timeout_s:.0f}s"
                    ) from e
                logging.info("Still waiting for server...")
                time.sleep(5)

    def _drop_connection(self) -> None:
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.close()
            self._ws = None
            self._reconnects += 1

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        attempts = 1 + self._max_retries
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                if self._ws is None:
                    # Raises PolicyServerUnreachable past connect_timeout_s, ending the loop.
                    self._ws, self._server_metadata = self._wait_for_server()
                self._ws.send(data)
                response = self._ws.recv()
            except _TRANSIENT_ERRORS as e:
                last_error = e
                logging.error(f"Policy request failed on attempt {attempt}/{attempts} ({type(e).__name__}: {e})")
                self._drop_connection()
                continue
            if isinstance(response, str):
                # we're expecting bytes; if the server sends a string, it's an error.
                raise RuntimeError(f"Error in inference server:\n{response}")
            if attempt > 1:
                logging.info(f"Policy request recovered on attempt {attempt}")
            return msgpack_numpy.unpackb(response)
        message = f"Policy server at {self._uri} unreachable after {attempts} attempts"
        raise PolicyServerUnreachable(message) from last_error

    @override
    def reset(self) -> None:
        pass
