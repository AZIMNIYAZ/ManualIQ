"""Structured JSON logging with a per-request ID.

Every log line written while handling a request carries that request's ID, so all
lines for one request can be found together. Every record passes through redact():
the API key never reaches a log.

Only the API process writes the log file (LOG_FILE): on Windows, rotation fails when
two processes have the same file open, so scripts log to the console only.
"""

import json
import logging
import logging.handlers
import re
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.errors import error_response, redact

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

logger = logging.getLogger("app")

# A client may send its own X-Request-ID; it is copied into every log line, so only safe values are kept
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024
LOG_FILE_BACKUPS = 3
RESERVED_KEYS = {"time", "level", "logger", "request_id", "message", "exception"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        # Extra fields passed as logger.info("...", extra={"fields": {...}}); they can never
        # overwrite the reserved keys above (a field called "message" would replace the event name)
        for key, value in getattr(record, "fields", {}).items():
            entry[f"field_{key}" if key in RESERVED_KEYS else key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return redact(json.dumps(entry, default=str))


def log_file_enabled(path: str | None) -> bool:
    return bool(path) and path.lower() != "none"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Replaces the root handlers (calling it again never duplicates records)."""
    formatter = JsonFormatter()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file_enabled(log_file):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_FILE_MAX_BYTES, backupCount=LOG_FILE_BACKUPS, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
    root = logging.getLogger()
    for old in root.handlers:
        old.close()
    root.handlers = handlers
    root.setLevel(level)
    # qdrant-client logs every HTTP call through httpx at INFO: noise, keep warnings only
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Our middleware logs each request, so uvicorn's own access log would be a duplicate
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, logs one line per request and returns the ID as X-Request-ID.

    It also turns any unexpected exception into a generic 500 internal_error here, so the
    header, the body's request_id and the logged traceback always match. (FastAPI's own
    last-resort handler sits outside this middleware and would lose the header.)
    """

    def __init__(self, app, slow_request_ms: float = 15000):
        super().__init__(app)
        self.slow_request_ms = slow_request_ms

    async def dispatch(self, request: Request, call_next):
        sent = request.headers.get("X-Request-ID", "")
        request_id = sent if REQUEST_ID_RE.match(sent) else uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                # Full traceback in the log; nothing internal in the response
                logger.exception("unhandled_error", extra={"fields": {
                    "code": "internal_error", "method": request.method, "path": request.url.path}})
                response = error_response(500, "internal_error", "An unexpected error occurred", request_id)
            response.headers["X-Request-ID"] = request_id
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            fields = {"method": request.method, "path": request.url.path, "status": response.status_code,
                      "duration_ms": duration_ms}
            if duration_ms > self.slow_request_ms:
                logger.warning("slow_request", extra={"fields": {**fields, "threshold_ms": self.slow_request_ms}})
            logger.info("request", extra={"fields": fields})
            return response
        finally:
            request_id_var.reset(token)
