"""Answer generation with Grok through xAI's OpenAI-compatible API.

The interface lets another provider replace Grok without touching the rest of the
code. The API key is only needed here: /search works without it.
"""

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import openai
from fastapi import Request

from app.core.config import Settings

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    status_code = 502


class GenerationNotConfigured(GenerationError):
    status_code = 503


class LLMAuthenticationError(GenerationError):
    status_code = 502


class LLMRateLimited(GenerationError):
    status_code = 429


class LLMTimeout(GenerationError):
    status_code = 504


class LLMUnavailable(GenerationError):
    status_code = 502


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int  # billed as output by reasoning models; 0 if not reported
    latency_ms: float
    finish_reason: str | None


class GenerationService(Protocol):
    model: str

    def generate(self, system: str, user: str) -> GenerationResult: ...


class GrokGenerationService:
    def __init__(self, settings: Settings, model: str | None = None, temperature: float | None = None,
                 max_output_tokens: int | None = None):
        """Defaults to the answer model and settings; the query rewriter passes its own."""
        if not settings.xai_api_key or not (model or settings.xai_chat_model):
            raise GenerationNotConfigured("Generation not configured: set XAI_API_KEY and XAI_CHAT_MODEL in .env")
        self.model = model or settings.xai_chat_model
        self.temperature = settings.llm_temperature if temperature is None else temperature
        self.max_output_tokens = max_output_tokens or settings.llm_max_output_tokens
        self.client = openai.OpenAI(
            api_key=settings.xai_api_key,
            base_url=settings.xai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,  # the SDK retries rate limits and network errors
        )

    def generate(self, system: str, user: str) -> GenerationResult:
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
        except openai.AuthenticationError as e:
            raise LLMAuthenticationError("LLM authentication failed: check XAI_API_KEY") from e
        except openai.RateLimitError as e:
            raise LLMRateLimited("LLM rate limit reached: try again shortly") from e
        except openai.APITimeoutError as e:
            raise LLMTimeout("The LLM did not respond in time") from e
        except openai.APIConnectionError as e:
            raise LLMUnavailable(f"Could not reach the LLM API: {e}") from e
        except openai.APIStatusError as e:  # bad request, unknown model, server errors
            raise LLMUnavailable(f"LLM API error {e.status_code}: {e.message}") from e

        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        if not answer:
            raise LLMUnavailable(f"The LLM returned an empty answer (finish_reason={choice.finish_reason})")
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        return GenerationResult(
            answer=answer,
            model=response.model or self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            reasoning_tokens=(getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            finish_reason=choice.finish_reason,
        )


def create_generator(settings: Settings) -> GenerationService | None:
    """None when no key/model is configured: the app still starts and /search works."""
    if not settings.xai_api_key or not settings.xai_chat_model:
        logger.warning("Generation not configured (XAI_API_KEY / XAI_CHAT_MODEL missing): /chat will return 503")
        return None
    return GrokGenerationService(settings)


def create_rewriter(settings: Settings) -> GenerationService | None:
    """The model used to rewrite follow-up questions (fast, deterministic, short output)."""
    if not settings.xai_api_key or not (settings.xai_rewrite_model or settings.xai_chat_model):
        return None
    return GrokGenerationService(settings, model=settings.xai_rewrite_model or settings.xai_chat_model,
                                 temperature=0.0, max_output_tokens=settings.rewrite_max_output_tokens)


def get_rewriter(request: Request) -> GenerationService | None:
    return getattr(request.app.state, "rewriter", None)


def get_generator(request: Request) -> GenerationService:
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        raise GenerationNotConfigured("Generation not configured: set XAI_API_KEY and XAI_CHAT_MODEL in .env")
    return generator
