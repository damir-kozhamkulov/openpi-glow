"""Transport checks for the LIBERO eval path: does a stalled server survive, and does a closed
connection recover. No GPU and no model - the policy is a stub that blocks the event loop.

The two sides run in the two venvs they run in production, because their websockets versions
differ and that difference is what broke the 2026-08-26 eval:

    GLOW_TEST_SERVER_PYTHON=/venv-server/bin/python \
      PYTHONPATH=/app:/app/packages/openpi-client/src \
      /venv-client/bin/python \
      /app/packages/openpi-client/src/openpi_client/websocket_client_policy_test.py

Without GLOW_TEST_SERVER_PYTHON the server runs under the same interpreter as the client, and
falls back to a stand-in server if openpi is not importable there - the output says which.
Takes about a minute: one case stalls a server past the keepalive on purpose.
"""

import os
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import websockets

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy as _wcp

_STALL_S = 25.0  # longer than the 20 s keepalive default the server used on 2026-08-26
_ACTIONS = np.zeros((10, 7), dtype=np.float32)
_SERVER_PYTHON = os.environ.get("GLOW_TEST_SERVER_PYTHON", sys.executable)


# --------------------------------------------------------------------------------------------
# Server side. Runs in its own process, under the server venv when one is named.
# --------------------------------------------------------------------------------------------
def _serve(mode: str, port: int) -> None:
    if mode == "stall":
        _serve_stalling_policy(port)
    else:
        _serve_raw(mode, port)


def _serve_stalling_policy(port: int) -> None:
    """openpi's own server, serving a policy whose first inference blocks the event loop."""
    try:
        from openpi.serving import websocket_policy_server
    except ImportError:
        print("IMPL raw (openpi not importable, real server code NOT exercised)", flush=True)
        _serve_raw("stall-raw", port)
        return

    from openpi_client import base_policy

    class _StallingPolicy(base_policy.BasePolicy):
        def __init__(self):
            self.stalled = False

        def infer(self, obs):
            if not self.stalled:
                self.stalled = True
                time.sleep(_STALL_S)  # blocks the loop exactly as a torch.compile autotune does
            return {"actions": _ACTIONS}

        def reset(self):
            pass

    print("IMPL openpi", flush=True)
    websocket_policy_server.WebsocketPolicyServer(
        policy=_StallingPolicy(), host="127.0.0.1", port=port, metadata={"server": "test"}
    ).serve_forever()


def _serve_raw(mode: str, port: int) -> None:
    """Stand-in server. `close` hangs up on the first request; `stall-raw` blocks the loop."""
    import asyncio

    import websockets.asyncio.server as _server

    packer = msgpack_numpy.Packer()
    state = {"connections": 0, "closed": False, "stalled": False}

    async def handler(ws):
        state["connections"] += 1
        await ws.send(packer.pack({"server": "test"}))
        while True:
            try:
                await ws.recv()
            except Exception:
                return
            if mode == "close" and not state["closed"]:
                state["closed"] = True
                await ws.close()
                return
            if mode == "stall-raw" and not state["stalled"]:
                state["stalled"] = True
                time.sleep(_STALL_S)
            await ws.send(packer.pack({"actions": _ACTIONS, "connections": state["connections"]}))

    async def run():
        async with _server.serve(
            handler, "127.0.0.1", port, compression=None, max_size=None, ping_timeout=None
        ) as server:
            await server.serve_forever()

    asyncio.run(run())


# --------------------------------------------------------------------------------------------
# Test harness
# --------------------------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(mode):
    port = _free_port()
    env = dict(os.environ)
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join([package_root, env["PYTHONPATH"]]) if "PYTHONPATH" in env else package_root
    log = tempfile.NamedTemporaryFile(prefix="glow-ws-test-", suffix=".log", delete=False)
    proc = subprocess.Popen(
        [_SERVER_PYTHON, os.path.abspath(__file__), "--serve", mode, str(port)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    proc.glow_log = log.name  # type: ignore[attr-defined]
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"server exited with {proc.returncode}; log:\n{_tail(log.name)}")
        try:
            with socket.create_connection(("127.0.0.1", port), 1):
                return proc, port
        except OSError:
            time.sleep(0.2)
    _stop(proc)
    raise AssertionError(f"server never bound port {port}; log:\n{_tail(log.name)}")


def _tail(path, lines=15):
    with open(path) as f:
        return "".join(f.readlines()[-lines:])


def _impl(proc) -> str:
    with open(proc.glow_log) as f:
        for line in f:
            if line.startswith("IMPL "):
                return line.strip()[5:]
    return "unknown"


def _stop(proc) -> None:
    proc.kill()
    proc.wait()


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------
def test_client_connects_under_the_installed_websockets():
    """The eval's two venvs resolve different websockets; connect() must accept what we pass."""
    version = getattr(websockets, "__version__", "unknown")
    supported = _wcp._CONNECT_TAKES_PING_TIMEOUT  # noqa: SLF001
    proc, port = _start_server("close")
    try:
        client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
        assert client.get_server_metadata()["server"] == "test"
        print(f"ok  client websockets {version}, ping_timeout supported: {supported}")
    finally:
        _stop(proc)


def test_server_survives_a_stall_longer_than_the_keepalive():
    """The 2026-08-26 failure: the server closed the connection during a 526 s autotune."""
    proc, port = _start_server("stall")
    try:
        client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
        started = time.time()
        result = client.infer({"x": 1})
        elapsed = time.time() - started
        assert elapsed >= _STALL_S, f"the server did not actually stall ({elapsed:.1f}s)"
        assert client._reconnects == 0, "the connection dropped during the stall"  # noqa: SLF001
        np.testing.assert_array_equal(result["actions"], _ACTIONS)
        print(f"ok  {elapsed:.0f}s stall survived on the original connection, server={_impl(proc)}")
    finally:
        _stop(proc)


def test_closed_connection_is_reconnected_and_the_request_resent():
    """The amplifier: one close used to fail every remaining episode of the run."""
    proc, port = _start_server("close")
    try:
        client = _wcp.WebsocketClientPolicy("127.0.0.1", port)
        result = client.infer({"x": 1})
        assert client._reconnects == 1, f"expected one reconnect, got {client._reconnects}"  # noqa: SLF001
        assert result["connections"] == 2, f"answered by connection {result['connections']}, not the new one"
        np.testing.assert_array_equal(result["actions"], _ACTIONS)
        assert client.infer({"x": 2})["connections"] == 2, "the reconnected connection was not reused"
        print("ok  closed connection reconnected, request resent, answer intact")
    finally:
        _stop(proc)


def test_dead_server_raises_instead_of_hanging():
    proc, port = _start_server("close")
    client = _wcp.WebsocketClientPolicy("127.0.0.1", port, connect_timeout_s=3.0)
    _stop(proc)
    started = time.time()
    try:
        client.infer({"x": 1})
    except _wcp.PolicyServerUnreachable:
        print(f"ok  dead server raised PolicyServerUnreachable after {time.time() - started:.0f}s")
        return
    raise AssertionError("expected PolicyServerUnreachable once the server was gone")


def main() -> int:
    tests = [
        test_client_connects_under_the_installed_websockets,
        test_server_survives_a_stall_longer_than_the_keepalive,
        test_closed_connection_is_reconnected_and_the_request_resent,
        test_dead_server_raises_instead_of_hanging,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as e:  # noqa: BLE001 - a report, not a handler
            failures += 1
            print(f"FAIL  {test.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--serve":
        _serve(sys.argv[2], int(sys.argv[3]))
    else:
        sys.exit(main())
