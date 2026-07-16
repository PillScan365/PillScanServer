import uvicorn

from pillscan_server.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "pillscan_server.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
        access_log=False,
    )
