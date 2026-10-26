import asyncio
import time

import pytest

from toolgraph.sim.vloop import SimulationDeadlock, run_virtual


def test_hour_long_sleeps_finish_instantly():
    async def main():
        loop = asyncio.get_running_loop()

        async def wait(d):
            await asyncio.sleep(d)
            return loop.time()

        return await asyncio.gather(wait(3600), wait(5), wait(1800))

    t0 = time.perf_counter()
    assert run_virtual(main) == [3600, 5, 1800]
    assert time.perf_counter() - t0 < 1.0


def test_time_does_not_advance_while_work_is_ready():
    async def main():
        loop = asyncio.get_running_loop()
        start = loop.time()
        for _ in range(1000):
            await asyncio.sleep(0)
        return loop.time() - start

    assert run_virtual(main) == 0


def test_timeouts_use_virtual_time():
    async def main():
        loop = asyncio.get_running_loop()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(10):
                await asyncio.sleep(100)
        return loop.time()

    assert run_virtual(main) == 10


def test_start_time():
    async def main():
        return asyncio.get_running_loop().time()

    assert run_virtual(main, start=500.0) == 500.0


def test_deadlock_is_detected():
    async def main():
        await asyncio.Event().wait()

    with pytest.raises(SimulationDeadlock):
        run_virtual(main)
