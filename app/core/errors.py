"""One error contract for the whole API:

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": null}}

- `code` is stable and machine-readable; it is defined once, on each exception class
  (AppError subclasses), never hard-coded in endpoints.
- `request_id` equals the X-Request-ID header and the request's log lines.
- `details` is null except for validation errors (a list of {field, problem}).
- Status codes, Allow (405) and Retry-After (429/503) are preserved.
- Messages pass through redact(): the API key never reaches a client or a log.

Unexpected exceptions are handled in the request-ID middleware (app.core.logging), which
returns a generic 500 internal_error without any internal detail.
"""

import logging
import re

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.errors")

_SECRETS: set[str] = set()
_KEY_PATTERN = re.compile(r"xai-[A-Za-z0-9_-]{16,}")


def register_secret(value: str | None) -> None:
    """Values that must never appear in an error message or a log record (e.g. the API key)."""
    if value and len(value) >= 8:
        _SECRETS.add(value)


def redact(text: str) -> str:
    for secret in _SECRETS:
        text = text.replace(secret, "[REDACTED]")
    return _KEY_PATTERN.sub("[REDACTED]", text)


class AppError(Exception):
    """Base for every error the API reports deliberately."""

    status_code = 500
    code = "internal_error"
    retry_after: int | None = None  # seconds; sent as Retry-After when set

    def __init__(self, message: str = "", retry_after: int | None = None, details=None):
        super().__init__(message)
        if retry_after is not None:
            self.retry_after = retry_after
        self.details = details


def error_body(code: str, message: str, request_id: str | None, details=None) -> dict:
    return {"error": {"code": code, "message": redact(message), "request_id": request_id, "details": details}}


def error_response(status: int, code: str, message: str, request_id: str | None, details=None,
                   headers: dict | None = None) -> JSONResponse:
    headers = dict(headers or {})
    if request_id:
        headers["X-Request-ID"] = request_id
    return JSONResponse(status_code=status, content=error_body(code, message, request_id, details), headers=headers)


def _request_id() -> str | None:
    from app.core.logging import request_id_var  # avoid an import cycle
    return request_id_var.get()


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    logger.warning("request_failed", extra={"fields": {
        "code": exc.code, "status": exc.status_code, "error": redact(str(exc))}})
    return error_response(exc.status_code, exc.code, str(exc), _request_id(), exc.details, headers)


def _field(loc) -> str:
    return ".".join(str(part) for part in loc)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for err in exc.errors():
        problem = err.get("msg", "invalid value").removeprefix("Value error, ")
        if err.get("type") == "json_invalid":
            problem = "request body is not valid JSON"
        details.append({"field": _field(err.get("loc", ())), "problem": problem})
    logger.warning("request_failed", extra={"fields": {"code": "validation_error", "status": 422, "details": details}})
    return error_response(422, "validation_error", "Request validation failed", _request_id(), details)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code == 404:
        code, message = "route_not_found", f"No endpoint at {request.url.path}"
    elif exc.status_code == 405:
        code, message = "method_not_allowed", f"{request.method} is not allowed on {request.url.path}"
    else:
        code, message = "http_error", str(exc.detail)
    logger.warning("request_failed", extra={"fields": {"code": code, "status": exc.status_code}})
    return error_response(exc.status_code, code, message, _request_id(), headers=exc.headers)


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
