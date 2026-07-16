import inspect
import logging
import sys

from loguru import logger

from pillscan_server.config import Settings


class InterceptHandler(logging.Handler):
    """Route standard-library and Uvicorn records through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = inspect.currentframe()
        depth = 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def configure_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level.upper(),
        serialize=settings.log_format == "json",
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )

    handler = InterceptHandler()
    logging.basicConfig(handlers=[handler], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error"):
        standard_logger = logging.getLogger(name)
        standard_logger.handlers = [handler]
        standard_logger.propagate = False
    logging.getLogger("uvicorn.access").disabled = True
