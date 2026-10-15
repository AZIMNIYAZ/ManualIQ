"""One-shot RAG answer: search -> context -> LLM -> answer checks.

No conversation memory yet (Step 9 adds Redis history and query rewriting).
"""

import logging
import time

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.models.chat import ChatRequest, ChatResponse, ChatSource, LlmInfo
from app.services import retrieval_service
from app.services.answer_checks import check_answer
from app.services.brand_detection import AliasTable
from app.services.context_builder import ContextSource, build_context
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.prompts import NOT_FOUND_ANSWER, PROMPT_VERSION, SYSTEM_PROMPT, user_message

logger = logging.getLogger(__name__)


def _source(s: ContextSource) -> ChatSource:
    r = s.result
    return ChatSource(n=s.n, citation=r.citation, brand=r.brand, model=r.model, section=r.section,
                      chunk_id=r.chunk_id, score=r.score)


def answer(
    client: QdrantClient,
    embedder: EmbeddingService,
    generator: GenerationService,
    settings: Settings,
    request: ChatRequest,
    aliases: AliasTable,
) -> ChatResponse:
    started = time.perf_counter()
    found = retrieval_service.search(client, embedder, settings, request.to_search(), aliases)
    search_ms = (time.perf_counter() - started) * 1000

    if not found.results:  # nothing to answer from: don't call (or pay for) the LLM
        return ChatResponse(
            answer=NOT_FOUND_ANSWER, found_in_documents=False, sources=[], filters=found.filters,
            warnings=[{"type": "no_results", "detail": "Search returned no chunks; the LLM was not called"}],
            llm=LlmInfo(model=None, prompt_version=PROMPT_VERSION),
            timings_ms={"search": round(search_ms, 1), "generate": 0.0,
                        "total": round((time.perf_counter() - started) * 1000, 1)},
        )

    context = build_context(found.results, settings.context_max_tokens)
    generated = generator.generate(SYSTEM_PROMPT, user_message(context.text, request.message))
    check = check_answer(generated.answer, context)

    by_n = {s.n: s for s in context.sources}
    response = ChatResponse(
        answer=generated.answer,
        found_in_documents=not check.not_found,
        sources=[_source(by_n[n]) for n in check.cited],
        filters=found.filters,
        warnings=check.warnings,
        llm=LlmInfo(
            model=generated.model, prompt_version=PROMPT_VERSION,
            input_tokens=generated.input_tokens, output_tokens=generated.output_tokens,
            reasoning_tokens=generated.reasoning_tokens,
            context_tokens=context.tokens, sources_dropped=context.dropped,
        ),
        timings_ms={"search": round(search_ms, 1), "generate": generated.latency_ms,
                    "total": round((time.perf_counter() - started) * 1000, 1)},
        context=context.text if request.debug else None,
        retrieved=[_source(s) for s in context.sources] if request.debug else None,
    )
    logger.info("chat", extra={"fields": {
        "message": request.message, "brands": found.filters.brands, "found": response.found_in_documents,
        "cited": [s.chunk_id for s in response.sources], "warnings": [w["type"] for w in check.warnings],
        "llm": response.llm.model_dump(), "timings_ms": response.timings_ms}})
    return response
