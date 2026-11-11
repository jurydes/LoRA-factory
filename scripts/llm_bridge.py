"""Forward the container's OpenAI API requests via a Windows network adapter.

Run this on the Docker host when a VPN prevents containers from reaching the LLM
server directly. Configuration comes from the host-only src/dev.env file.
"""
from __future__ import annotations

import os
import secrets
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from settings import load_global_config  # noqa: E402


def interface_address(name: str) -> str:
    try:
        socket.inet_aton(name)
        return name
    except OSError:
        addresses = psutil.net_if_addrs().get(name, [])
        for address in addresses:
            if address.family == socket.AF_INET:
                return address.address
    raise ValueError(f"No IPv4 address for network interface {name!r}")


settings = load_global_config().openai_server
if not settings.bind_interface:
    raise SystemExit("Set OPENAI_SERVER__BIND_INTERFACE in src/dev.env")
if not settings.key or settings.key == "EMPTY":
    raise SystemExit("Set a non-empty OPENAI_SERVER__KEY in src/dev.env")

upstream_base = str(settings.url).rstrip("/")
client = httpx.Client(
    transport=httpx.HTTPTransport(local_address=interface_address(settings.bind_interface)),
    timeout=httpx.Timeout(300, connect=10),
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # EOF terminates streamed responses, including SSE.

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        path = urlsplit(self.path)
        if not path.path.startswith("/v1/"):
            self.send_error(404)
            return
        expected = f"Bearer {settings.key}"
        if not secrets.compare_digest(self.headers.get("Authorization", ""), expected):
            self.send_error(401)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if not 0 <= length <= 32 * 1024 * 1024:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else None
        suffix = path.path[3:] if upstream_base.endswith("/v1") else path.path
        target = upstream_base + suffix
        if path.query:
            target += "?" + path.query
        headers = {"Authorization": expected}
        for key in ("Content-Type", "Accept"):
            if value := self.headers.get(key):
                headers[key] = value
        sent = False
        try:
            with client.stream(self.command, target, content=body, headers=headers) as upstream:
                self.send_response(upstream.status_code)
                for key in ("Content-Type", "Content-Encoding", "Cache-Control"):
                    if value := upstream.headers.get(key):
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                sent = True
                for chunk in upstream.iter_raw():
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except httpx.HTTPError as exc:
            if not sent:
                self.send_error(502, type(exc).__name__)
            else:
                self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            pass

if __name__ == "__main__":
    port = int(os.environ.get("FACTORY_LLM_BRIDGE_PORT", "8001"))
    print(f"LLM bridge listening on 127.0.0.1:{port}; upstream via {settings.bind_interface}", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    finally:
        client.close()
