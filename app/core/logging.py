"""Structured JSON logging with a per-request ID.

Every log line written while handling a request carries that request's ID,
so all lines for one request can be found together.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

logger = logging.getLogger("app")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        # Extra fields passed as logger.info("...", extra={"fields": {...}})
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Our middleware logs each request, so uvicorn's own access log would be a duplicate
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, logs one line per request and returns the ID as X-Request-ID."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error", extra={"fields": {"method": request.method, "path": request.url.path}})
            raise
        else:
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "request",
                extra={
                    "fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    }
                },
            )
            return response
        finally:
            request_id_var.reset(token)
