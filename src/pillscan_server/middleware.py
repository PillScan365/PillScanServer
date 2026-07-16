import re
from time import perf_counter
from uuid import uuid4

from loguru import logger
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from pillscan_server.models import PipelineTimings

SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestIDMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get("x-request-id", "")
        request_id = supplied if SAFE_REQUEST_ID.fullmatch(supplied) else uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope.get("method", "")
        path = scope.get("path", "")
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
                timings = scope.get("state", {}).get("pipeline_timings")
                if timings is not None:
                    headers.append(
                        "Server-Timing",
                        _server_timing_header(timings, perf_counter() - started_at),
                    )
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            timings = scope.get("state", {}).get("pipeline_timings")
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            pipeline_timings = timings.model_dump() if timings is not None else None
            logger.bind(
                request_id=request_id,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
                pipeline_timings=pipeline_timings,
            ).info(
                "request_completed duration_ms={} pipeline_timings={}",
                duration_ms,
                pipeline_timings,
            )


def _server_timing_header(timings: PipelineTimings, elapsed_seconds: float) -> str:
    values = timings.model_dump()
    stage_names = {
        "upload_read_ms": "upload",
        "image_normalization_ms": "image",
        "rate_limit_wait_ms": "rate_limit",
        "concurrency_wait_ms": "queue",
        "vision_analysis_ms": "vision",
        "catalog_resolution_ms": "catalog",
    }
    stages = [f"{stage_names[key]};dur={values[key]:.2f}" for key in stage_names]
    stages.append(f"app;dur={elapsed_seconds * 1000:.2f}")
    return ", ".join(stages)
