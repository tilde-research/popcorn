import json
import socket
import sys
from argparse import Namespace
from http.client import HTTPConnection, HTTPResponse

import pytest
import torch

from popcorn import KERNELS
from popcorn.bench.__main__ import cmd_fill, main
from popcorn.bench.grid import cases
from popcorn.bench.live import PROTOCOL, SERVICE, LiveServer
from popcorn.bench.model import Environment, Record, Result

ORIGIN = "https://tilde-research.github.io"


def _record(op="alpha", case_id="case", status="pass"):
    config = {"dims": {"D": 4}, "batch": [], "dtype": "float32", "args": {}, "present": []}
    result = Result(
        status=status,
        grad=False,
        benchmarked=status == "pass",
        bench={"fwd_ms": 1.0, "ref_fwd_ms": 2.0} if status == "pass" else {},
    )
    return Record(
        op,
        "fast",
        "D=4,float32",
        case_id,
        config,
        Environment("Test GPU", torch.__version__, "1.0", "2026-08-04T00:00:00+00:00"),
        result,
    )


def _request(live, method, path, *, origin=ORIGIN):
    connection = HTTPConnection(live.host, live.port, timeout=2)
    connection.request(method, path, headers={"Origin": origin})
    return connection, connection.getresponse()


def _event(response: HTTPResponse):
    fields = {}
    while True:
        line = response.readline().decode().rstrip("\r\n")
        if not line:
            if fields:
                fields["data"] = json.loads(fields.get("data", "{}"))
                return fields
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        fields[name] = value.lstrip()


def test_live_status_and_private_network_cors():
    with LiveServer(0, device="Test GPU", ops=["alpha"], total=3, cached=2, skipped=1) as live:
        connection, response = _request(live, "GET", "/v1/status")
        payload = json.loads(response.read())
        connection.close()

        assert response.status == 200
        assert response.getheader("Access-Control-Allow-Origin") == ORIGIN
        assert response.getheader("Access-Control-Allow-Private-Network") == "true"
        assert payload == {
            **payload,
            "service": SERVICE,
            "protocol": PROTOCOL,
            "state": "running",
            "host": "127.0.0.1",
            "port": live.port,
            "device": "Test GPU",
            "ops": ["alpha"],
            "total": 3,
            "completed": 0,
            "measured": 0,
            "pruned": 0,
            "cached": 2,
            "skipped": 1,
            "op_statuses": {},
        }

        connection, response = _request(live, "OPTIONS", "/v1/events")
        response.read()
        connection.close()
        assert response.status == 204
        assert response.getheader("Access-Control-Allow-Methods") == "GET, OPTIONS"
        assert response.getheader("Access-Control-Allow-Private-Network") == "true"

        connection, response = _request(live, "GET", "/v1/status", origin="http://localhost:3000")
        response.read()
        connection.close()
        assert response.status == 200
        assert response.getheader("Access-Control-Allow-Origin") == "http://localhost:3000"

        connection, response = _request(live, "GET", "/v1/status", origin="https://example.com")
        response.read()
        connection.close()
        assert response.status == 403

        connection = HTTPConnection(live.host, live.port, timeout=2)
        connection.putrequest("GET", "/v1/status", skip_host=True)
        connection.putheader("Host", f"example.com:{live.port}")
        connection.putheader("Origin", ORIGIN)
        connection.endheaders()
        response = connection.getresponse()
        response.read()
        connection.close()
        assert response.status == 403


def test_live_status_counts_benchmark_errors_per_op():
    live = LiveServer(0, device="Test GPU", ops=["alpha"], total=1)
    record = _record()
    record.result.bench_error = "timer failed"
    live.record(record)
    live.close()

    assert live.status()["statuses"] == {"bench_error": 1}
    assert live.status()["op_statuses"] == {"alpha": {"bench_error": 1}}


def test_live_sse_replays_and_filters_records_then_completes():
    live = LiveServer(0, device="Test GPU", ops=["alpha", "beta"], total=3).start()
    live.record(_record("alpha", "first"))
    live.record(_record("beta", "other"))

    connection, response = _request(live, "GET", "/v1/events?op=alpha")
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    session = _event(response)
    replay = _event(response)
    assert session["event"] == "session"
    assert session["data"]["completed"] == 2
    assert session["data"]["op_statuses"] == {"alpha": {"pass": 1}, "beta": {"pass": 1}}
    assert replay["event"] == "record"
    assert replay["data"]["record"]["op"] == "alpha"
    assert replay["data"]["record"]["case_id"] == "first"

    live.record(_record("beta", "ignored"))
    live.record(_record("alpha", "second"))
    streamed = None
    for _ in range(4):
        candidate = _event(response)
        if candidate["event"] == "record":
            streamed = candidate
            break
    assert streamed is not None
    assert streamed["data"]["record"]["op"] == "alpha"
    assert streamed["data"]["record"]["case_id"] == "second"

    live.close()
    complete = None
    for _ in range(4):
        candidate = _event(response)
        if candidate["event"] == "complete":
            complete = candidate
            break
    connection.close()
    assert complete is not None
    assert complete["data"]["state"] == "complete"
    assert complete["data"]["completed"] == 4


def test_live_server_reports_an_occupied_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(OSError):
            LiveServer(port, device="cpu", ops=["alpha"], total=1).start()


@pytest.mark.parametrize(
    ("arguments", "port"),
    [(["fill", "rms_norm", "--live"], 8765), (["fill", "rms_norm", "--live", "9000"], 9000)],
)
def test_fill_live_cli_parses_default_and_custom_ports(monkeypatch, arguments, port):
    parsed = []
    monkeypatch.setitem(main.__globals__, "cmd_fill", parsed.append)
    monkeypatch.setattr(sys, "argv", ["python -m popcorn.bench", *arguments])

    main()

    assert parsed[0].ops == ["rms_norm"]
    assert parsed[0].live == port


def test_fill_live_cli_parses_a_website_slice_after_the_live_flag(monkeypatch):
    parsed = []
    monkeypatch.setitem(main.__globals__, "cmd_fill", parsed.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m popcorn.bench",
            "fill",
            "rms_norm",
            "--live",
            "--slice",
            "normalized_shape",
            "dtype=float16",
            "...=[2,3]",
            "eps=0.000001",
        ],
    )

    main()

    assert parsed[0].live == 8765
    assert parsed[0].slice == [
        "normalized_shape",
        "dtype=float16",
        "...=[2,3]",
        "eps=0.000001",
    ]


def test_live_replay_is_bounded_and_error_state_is_retained():
    live = LiveServer(0, device="Test GPU", ops=["alpha"], total=3, replay=2)
    for index in range(3):
        live.record(_record("alpha", f"case-{index}"))
    subscriber, _, replay = live.subscribe("alpha")
    live.unsubscribe(subscriber)
    live.close(RuntimeError("benchmark stopped"))

    assert [event[1]["record"]["case_id"] for event in replay] == ["case-1", "case-2"]
    assert live.status()["state"] == "error"
    assert live.status()["error"] == "RuntimeError: benchmark stopped"


def test_fill_emits_each_record_to_the_live_session(monkeypatch, tmp_path):
    op = KERNELS["rms_norm"]
    backend = next(name for name in op.available_backends() if name != "torch")
    case = cases(op, limit=1)[0]
    emitted = []
    closed = []
    started = {}
    written = []

    class FakeStore:
        user = tmp_path

        def write_user(self, records):
            written.extend(records)

    class FakeLive:
        host = "127.0.0.1"

        def __init__(self, port, **metadata):
            self.port = port
            started.update(port=port, **metadata)

        def start(self):
            started["running"] = True
            return self

        def record(self, record):
            emitted.append(record)

        def pruned(self):
            raise AssertionError("the only case should not be pruned")

        def close(self, error=None):
            closed.append(error)

    record = _record(op.name)
    record.impl = backend
    record.case = str(case)
    record.case_id = case.case_id
    record.config = case.config()
    monkeypatch.setitem(cmd_fill.__globals__, "Store", FakeStore)
    monkeypatch.setitem(cmd_fill.__globals__, "LiveServer", FakeLive)
    monkeypatch.setitem(cmd_fill.__globals__, "_work", lambda *args: [(op, backend, case)])
    monkeypatch.setitem(cmd_fill.__globals__, "_usable_records", lambda *args: [])
    monkeypatch.setitem(cmd_fill.__globals__, "available", lambda impl: True)
    monkeypatch.setitem(cmd_fill.__globals__, "infeasible", lambda *args: False)
    monkeypatch.setattr(op.bench, "run_case", lambda *args, **kwargs: record)

    cmd_fill(
        Namespace(
            ops=[op.name],
            backend=backend,
            limit=1,
            device="cpu",
            hardware=None,
            force=True,
            in_process=True,
            timeout=30,
            reps=1,
            flush=10,
            live=8765,
        )
    )

    assert started["running"] is True
    assert started["port"] == 8765
    assert started["ops"] == [op.name]
    assert started["total"] == 1
    assert emitted == written == [record]
    assert closed == [None]
