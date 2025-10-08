import asyncio
import json
import socket
import sys
from contextlib import suppress
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.net import lan_discovery_query, lan_discovery_server
from server_browser_utils import format_uptime, load_known_servers, save_known_servers


def _get_free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_load_known_servers_filters_invalid_entries(tmp_path):
    payload = {
        "servers": [
            {"host": "example.com", "port": "5000", "label": " Example "},
            {"host": "   ", "port": 4000},
            {"host": "bad", "port": "not-a-number"},
            "ignored",
        ]
    }
    path = tmp_path / "known.json"
    path.write_text(json.dumps(payload))

    servers = load_known_servers(str(path))

    assert len(servers) == 2
    assert servers[0]["host"] == "example.com"
    assert servers[0]["port"] == 5000
    assert servers[0]["label"] == "Example"
    assert servers[1]["host"] == "bad"
    assert servers[1]["port"] == 0


def test_save_known_servers_round_trip(tmp_path):
    servers = [
        {"host": "alpha", "port": 1234, "discovery_port": 6000, "label": "Alpha"},
        {"host": "beta", "port": 0},
    ]
    path = tmp_path / "known.json"

    save_known_servers(str(path), servers)

    written = json.loads(path.read_text())
    assert written == {"servers": servers}


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (None, "?"),
        (45, "45s"),
        (75, "1m 15s"),
        (3661, "1h 1m"),
        ("90", "1m 30s"),
        ("invalid", "?"),
    ],
)
def test_format_uptime_variants(seconds, expected):
    assert format_uptime(seconds) == expected


def test_lan_discovery_query_receives_stats():
    discovery_port = _get_free_udp_port()
    info_payload = {"uptime": 12, "players": {"total": 3}}
    info_calls = 0

    def info_cb():
        nonlocal info_calls
        info_calls += 1
        return info_payload

    async def _run():
        server_task = asyncio.create_task(
            lan_discovery_server("TestServer", 4242, discovery_port, info_cb),
            name="test_discovery_server",
        )
        try:
            await asyncio.sleep(0.05)
            response = await lan_discovery_query("127.0.0.1", discovery_port, timeout=0.2)
        finally:
            server_task.cancel()
            with suppress(asyncio.CancelledError):
                await server_task

        assert response["name"] == "TestServer"
        assert response["tcp_port"] == 4242
        assert response["discovery_port"] == discovery_port
        assert response["stats"] == info_payload
        assert info_calls > 0

    asyncio.run(_run())


def test_lan_discovery_query_handles_timeout():
    async def _run():
        unused_port = _get_free_udp_port()
        response = await lan_discovery_query("127.0.0.1", unused_port, timeout=0.05)
        assert response == {}

    asyncio.run(_run())
