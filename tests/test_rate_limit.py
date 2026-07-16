import asyncio

import pytest
from aiolimiter import AsyncLimiter

from pillscan_server.errors import RateLimitExceeded
from pillscan_server.service import AnalysisGate


@pytest.mark.asyncio
async def test_analysis_gate_rejects_when_rate_capacity_does_not_recover_in_time() -> None:
    gate = AnalysisGate(
        limiter=AsyncLimiter(1, time_period=60),
        concurrency=asyncio.Semaphore(1),
        wait_seconds=0.01,
    )

    async with gate.acquire():
        pass

    with pytest.raises(RateLimitExceeded):
        async with gate.acquire():
            pass


@pytest.mark.asyncio
async def test_analysis_gate_releases_concurrency_slot() -> None:
    gate = AnalysisGate(
        limiter=AsyncLimiter(2, time_period=1),
        concurrency=asyncio.Semaphore(1),
        wait_seconds=0.1,
    )

    async with gate.acquire():
        pass
    async with gate.acquire():
        pass
