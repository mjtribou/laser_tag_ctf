# common/net.py
import asyncio
import json
import struct
from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, Optional

# Simple newline-delimited JSON protocol helpers

async def send_json(writer: asyncio.StreamWriter, obj: Dict[str, Any]):
    data = (json.dumps(obj) + "\n").encode("utf-8")
    writer.write(data)
    await writer.drain()

async def read_json(reader: asyncio.StreamReader) -> Dict[str, Any]:
    line = await reader.readline()
    if not line:
        return {}
    return json.loads(line.decode("utf-8"))

DISCOVERY_MAGIC = "LASERTAG_DISCOVER_V1"

async def lan_discovery_broadcast(port: int, timeout: float = 1.0):
    """Broadcasts a UDP discovery packet and listens briefly for replies."""
    import socket, asyncio
    loop = asyncio.get_event_loop()
    # Create listening socket first to catch responses
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_sock.bind(("", 0))  # random port
    recv_port = recv_sock.getsockname()[1]
    recv_sock.setblocking(False)

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    send_sock.setblocking(False)

    msg = json.dumps({"magic": DISCOVERY_MAGIC, "reply_port": recv_port}).encode("utf-8")
    send_sock.sendto(msg, ("255.255.255.255", port))

    servers = []

    async def receive_replies():
        end = loop.time() + timeout
        while loop.time() < end:
            try:
                data, addr = await loop.run_in_executor(None, recv_sock.recvfrom, 2048)
                try:
                    res = json.loads(data.decode("utf-8"))
                    if res.get("magic") == DISCOVERY_MAGIC and "name" in res:
                        res.setdefault("addr", addr[0])
                        servers.append(res)
                except Exception:
                    pass
            except Exception:
                await asyncio.sleep(0.01)
        recv_sock.close()
        send_sock.close()

    await receive_replies()
    return servers

async def lan_discovery_query(host: str, port: int, timeout: float = 1.0) -> Dict[str, Any]:
    """Send a unicast discovery query directly to a host."""
    import socket

    loop = asyncio.get_event_loop()

    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    recv_sock.bind(("", 0))
    recv_port = recv_sock.getsockname()[1]
    recv_sock.setblocking(False)

    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_sock.setblocking(False)

    msg = json.dumps({"magic": DISCOVERY_MAGIC, "reply_port": recv_port}).encode("utf-8")

    try:
        send_sock.sendto(msg, (host, port))
    except Exception:
        recv_sock.close()
        send_sock.close()
        return {}

    async def receive_one() -> Dict[str, Any]:
        end = loop.time() + timeout
        while loop.time() < end:
            try:
                data, addr = await loop.run_in_executor(None, recv_sock.recvfrom, 2048)
                try:
                    res = json.loads(data.decode("utf-8"))
                    if res.get("magic") == DISCOVERY_MAGIC and "name" in res:
                        res.setdefault("addr", addr[0])
                        return res
                except Exception:
                    pass
            except Exception:
                await asyncio.sleep(0.01)
        return {}

    try:
        return await receive_one()
    finally:
        recv_sock.close()
        send_sock.close()

async def lan_discovery_server(
    name: str,
    port: int,
    discovery_port: int,
    info_cb: Optional[Callable[[], Dict[str, Any]]] = None,
):
    """UDP task that replies to LAN discovery pings using asyncio DatagramProtocol."""
    import asyncio, json
    from asyncio import DatagramProtocol

    class DiscoveryProtocol(DatagramProtocol):
        def __init__(self, name: str, tcp_port: int, info_cb: Optional[Callable[[], Dict[str, Any]]]):
            self.name = name
            self.tcp_port = tcp_port
            self.transport = None
            self.info_cb = info_cb

        def connection_made(self, transport):
            self.transport = transport
            sock = transport.get_extra_info("socket")
            try:
                # allow broadcast, just in case
                import socket
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except Exception:
                pass
            print(f"[discovery] listening on UDP :{discovery_port}")

        def datagram_received(self, data, addr):
            try:
                msg = json.loads(data.decode("utf-8"))
                if msg.get("magic") != DISCOVERY_MAGIC:
                    return
                reply_port = int(msg.get("reply_port", addr[1]))
                payload: Dict[str, Any] = {
                    "magic": DISCOVERY_MAGIC,
                    "name": self.name,
                    "tcp_port": self.tcp_port,
                    "discovery_port": discovery_port,
                }
                if self.info_cb is not None:
                    try:
                        info = self.info_cb()
                        if info:
                            payload["stats"] = info
                    except Exception:
                        pass
                reply = json.dumps(payload).encode("utf-8")
                # Unicast reply back to the requester’s indicated port
                self.transport.sendto(reply, (addr[0], reply_port))
            except Exception:
                # ignore malformed packets
                pass

        def error_received(self, exc):
            # Keep running even if occasional network errors happen
            print(f"[discovery] error_received: {exc}")

        def connection_lost(self, exc):
            # Socket closed; nothing else to do.
            pass

    loop = asyncio.get_running_loop()
    # Bind to 0.0.0.0 on discovery_port
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: DiscoveryProtocol(name, port, info_cb),
        local_addr=("0.0.0.0", discovery_port),
        allow_broadcast=True,
    )
    try:
        # Run forever until cancelled by the server shutdown
        await asyncio.Event().wait()
    finally:
        transport.close()
