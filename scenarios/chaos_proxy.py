"""A TCP proxy that loses responses: the request reaches the server, the answer never comes back.

It watches HTTP traffic for ``tools/call`` requests to chosen tools and, with
probability ``--drop``, lets the server process the request and then cuts the
client's connection as the response starts to arrive. That is the classic
"did my payment go through?" failure.

    python scenarios/chaos_proxy.py --listen 8812 --target 8811 --tools charge,send_email --drop 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import sys


class Chaos:
    def __init__(self, target: int, tools: set[str], drop: float, seed: int) -> None:
        self.target = target
        self.pattern = re.compile(rb'"method"\s*:\s*"tools/call".*?"name"\s*:\s*"(' + b"|".join(re.escape(t.encode()) for t in tools) + rb')"', re.S)
        self.drop = drop
        self.rng = random.Random(seed)
        self.dropped = 0

    async def handle(self, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        server_r, server_w = await asyncio.open_connection("127.0.0.1", self.target)
        doomed = asyncio.Event()

        async def upstream() -> None:
            buf = b""
            while data := await client_r.read(65536):
                buf = (buf + data)[-65536:]
                if self.pattern.search(buf):
                    buf = b""
                    if self.rng.random() < self.drop:
                        doomed.set()
                server_w.write(data)
                await server_w.drain()
            server_w.close()

        async def downstream() -> None:
            while data := await server_r.read(65536):
                if doomed.is_set():
                    # The server has done the work; the client never hears about it.
                    self.dropped += 1
                    print(f"dropped a response ({self.dropped})", file=sys.stderr, flush=True)
                    client_w.transport.abort()
                    server_w.close()
                    return
                client_w.write(data)
                await client_w.drain()
            client_w.close()

        tasks = [asyncio.create_task(upstream()), asyncio.create_task(downstream())]
        try:
            await asyncio.gather(*tasks)
        except (ConnectionError, OSError):
            pass
        finally:
            for t in tasks:
                t.cancel()
            client_w.transport.abort()
            server_w.close()


async def serve(listen: int, chaos: Chaos) -> None:
    server = await asyncio.start_server(chaos.handle, "127.0.0.1", listen)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--drop", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(serve(args.listen, Chaos(args.target, set(args.tools.split(",")), args.drop, args.seed)))


if __name__ == "__main__":
    main()
