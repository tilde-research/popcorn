"""Read-only loopback stream for projecting a live benchmark session into the site."""

from __future__ import annotations

import json
import queue
import re
import threading
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from types import TracebackType
from typing import Any
from urllib.parse import parse_qs, urlsplit

from popcorn.bench.model import Record

DEFAULT_PORT = 8765
PROTOCOL = 1
SERVICE = "popcorn.bench.live"
REPLAY = 256
_PRODUCTION_ORIGIN = "https://tilde-research.github.io"
_LOCAL_ORIGIN = re.compile(r"http://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?\Z")


def _package_version() -> str:
    try:
        return version("popcorn")
    except PackageNotFoundError:
        return "unknown"


class _Subscriber:
    def __init__(self, op: str | None) -> None:
        self.op = op
        self.events: queue.Queue[tuple[int, str, dict[str, Any]] | None] = queue.Queue(maxsize=REPLAY)

    def offer(self, event: tuple[int, str, dict[str, Any]] | None) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass


class _HTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = False
    block_on_close = True

    def __init__(self, address: tuple[str, int], session: LiveServer) -> None:
        self.session = session
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _HTTPServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5)

    def _origin(self) -> str | None:
        return self.headers.get("Origin")

    def _allowed(self) -> bool:
        return self.server.session.allows_request(self.headers.get("Host", ""), self._origin())

    def _cors(self) -> None:
        origin = self._origin()
        if origin is not None:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        if not self._allowed():
            self._json(HTTPStatus.FORBIDDEN, {"error": "origin or host is not allowed"})
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Last-Event-ID")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        if not self._allowed():
            self._json(HTTPStatus.FORBIDDEN, {"error": "origin or host is not allowed"})
            return
        target = urlsplit(self.path)
        if target.path == "/v1/status":
            self._json(HTTPStatus.OK, self.server.session.status())
            return
        if target.path == "/v1/events":
            values = parse_qs(target.query)
            op = values.get("op", [None])[0]
            self._events(op)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _event(self, sequence: int | None, name: str, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        identity = f"id: {sequence}\n" if sequence is not None else ""
        self.wfile.write(f"{identity}event: {name}\ndata: {data}\n\n".encode())
        self.wfile.flush()

    def _events(self, op: str | None) -> None:
        try:
            last = int(self.headers.get("Last-Event-ID", "0"))
        except ValueError:
            last = 0
        subscriber, snapshot, replay = self.server.session.subscribe(op)
        self.send_response(HTTPStatus.OK)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self._event(None, "session", snapshot)
            for sequence, payload in replay:
                if sequence > last:
                    self._event(sequence, "record", payload)
            if snapshot["state"] != "running":
                self._event(None, "complete", snapshot)
                return
            while True:
                try:
                    item = subscriber.events.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    return
                sequence, name, payload = item
                if sequence > last:
                    self._event(sequence, name, payload)
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            self.server.session.unsubscribe(subscriber)


class LiveServer:
    """A benchmark-scoped HTTP/SSE server bound to the IPv4 loopback interface."""

    def __init__(
        self,
        port: int,
        *,
        device: str,
        ops: list[str],
        total: int,
        cached: int = 0,
        skipped: int = 0,
        command: str = "fill",
        replay: int = REPLAY,
    ) -> None:
        self.host = "127.0.0.1"
        self.port = port
        self.device = device
        self.ops = sorted(set(ops))
        self.total = total
        self.cached = cached
        self.skipped = skipped
        self.command = command
        self.session_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.version = _package_version()
        self.completed = 0
        self.measured = 0
        self.pruned_count = 0
        self.statuses: Counter[str] = Counter()
        self.op_statuses: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.state = "running"
        self.error = ""
        self._sequence = 0
        self._replay_size = replay
        self._records: defaultdict[str, deque[tuple[int, dict[str, Any]]]] = defaultdict(
            lambda: deque(maxlen=self._replay_size)
        )
        self._subscribers: list[_Subscriber] = []
        self._lock = threading.RLock()
        self._server: _HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> LiveServer:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(exc)

    def start(self) -> LiveServer:
        if self._server is not None:
            return self
        server = _HTTPServer((self.host, self.port), self)
        self._server = server
        self.port = server.server_port
        self._thread = threading.Thread(target=server.serve_forever, name="popcorn-bench-live", daemon=True)
        self._thread.start()
        return self

    def allows_request(self, host: str, origin: str | None) -> bool:
        allowed_hosts = {f"{self.host}:{self.port}", f"localhost:{self.port}"}
        if host not in allowed_hosts:
            return False
        return origin is None or origin == _PRODUCTION_ORIGIN or _LOCAL_ORIGIN.fullmatch(origin) is not None

    def _status_unlocked(self) -> dict[str, Any]:
        return {
            "service": SERVICE,
            "protocol": PROTOCOL,
            "version": self.version,
            "session": self.session_id,
            "state": self.state,
            "command": self.command,
            "host": self.host,
            "port": self.port,
            "device": self.device,
            "ops": self.ops,
            "total": self.total,
            "completed": self.completed,
            "measured": self.measured,
            "pruned": self.pruned_count,
            "cached": self.cached,
            "skipped": self.skipped,
            "statuses": dict(sorted(self.statuses.items())),
            "op_statuses": {op: dict(sorted(statuses.items())) for op, statuses in sorted(self.op_statuses.items())},
            "started_at": self.started_at,
            "error": self.error,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def subscribe(self, op: str | None) -> tuple[_Subscriber, dict[str, Any], list[tuple[int, dict[str, Any]]]]:
        with self._lock:
            subscriber = _Subscriber(op)
            self._subscribers.append(subscriber)
            if op is None:
                replay = [event for records in self._records.values() for event in records]
            else:
                selected = self._records.get(op)
                replay = list(selected) if selected is not None else []
            replay.sort(key=lambda event: event[0])
            return subscriber, self._status_unlocked(), replay

    def unsubscribe(self, subscriber: _Subscriber) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _emit_unlocked(self, name: str, payload: dict[str, Any], op: str | None = None) -> int:
        self._sequence += 1
        event = (self._sequence, name, payload)
        for subscriber in self._subscribers:
            if name != "record" or subscriber.op is None or subscriber.op == op:
                subscriber.offer(event)
        return self._sequence

    def _progress_unlocked(self) -> None:
        self._emit_unlocked("progress", self._status_unlocked())

    def record(self, record: Record) -> None:
        with self._lock:
            self.completed += 1
            self.measured += 1
            outcome = "bench_error" if record.result.bench_error else record.result.status
            self.statuses[outcome] += 1
            self.op_statuses[record.op][outcome] += 1
            payload = {"record": record.to_dict()}
            sequence = self._emit_unlocked("record", payload, record.op)
            self._records[record.op].append((sequence, payload))
            self._progress_unlocked()

    def pruned(self) -> None:
        with self._lock:
            self.completed += 1
            self.pruned_count += 1
            self._progress_unlocked()

    def close(self, error: BaseException | str | None = None) -> None:
        with self._lock:
            if self.state != "running":
                return
            if error is not None:
                self.state = "error"
                detail = str(error)
                self.error = detail if isinstance(error, str) else f"{type(error).__name__}: {detail}".rstrip(": ")
            else:
                self.state = "complete"
            self._emit_unlocked("complete", self._status_unlocked())
            subscribers = list(self._subscribers)
            for subscriber in subscribers:
                subscriber.offer(None)
            server, thread = self._server, self._thread
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        self._server = None
        self._thread = None
