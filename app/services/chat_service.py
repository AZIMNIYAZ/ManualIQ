"""RAG answer with conversation memory:

  history (Redis) -> rewrite follow-up into a standalone query -> search -> context
  -> LLM -> answer checks -> save the turn (user + assistant, one transaction)

Previous answers are never sent to the answer model: their [n] citations referred to
other sources, and the answer must be grounded in this turn's sources only. History is
used only by the rewriter.

Brand precedence: the request's brand field (per request, never stored) > brands from
query rewriting > brands detected in the (rewritten) question.
"""

import logging
import time
from datetime import datetime, timezone

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.models.chat import ChatRequest, ChatResponse, ChatSource, LlmInfo, RewriteInfo
from app.models.session import StoredMessage
from app.services import retrieval_service
from app.services.answer_checks import check_answer
from app.services.brand_detection import AliasTable, detect_brands
from app.services.context_builder import ContextSource, build_context
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.prompts import NOT_FOUND_ANSWER, PROMPT_VERSION, SYSTEM_PROMPT, user_message
from app.services.query_rewriter import REWRITE_PROMPT_VERSION, RewriteResult, rewrite
from app.services.session_service import SessionStore, SessionUnavailable

logger = logging.getLogger(__name__)


def _source(s: ContextSource) -> ChatSource:
    r = s.result
    return ChatSource(n=s.n, citation=r.citation, brand=r.brand, model=r.model, section=r.section,
                      chunk_id=r.chunk_id, score=r.score)


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


def answer(
    client: QdrantClient,
    embedder: EmbeddingService,
    generator: GenerationService,
    settings: Settings,
    request: ChatRequest,
    aliases: AliasTable,
    sessions: SessionStore | None = None,
    rewriter: GenerationService | None = None,
) -> ChatResponse:
    started = time.perf_counter()
    warnings: list[dict] = []
    session_id = request.session_id or SessionStore.new_session_id()

    # ---- 1. History (memory is optional: without Redis the chat still answers, and says so)
    history: list[StoredMessage] = []
    memory_ok = sessions is not None
    if sessions is not None:
        try:
            history = sessions.history(session_id, settings.history_messages_in_prompt)
        except SessionUnavailable as e:
            memory_ok = False
            warnings.append({"type": "history_unavailable", "detail": str(e)})

    # ---- 2. Rewrite follow-ups into a standalone question (no history: no extra call)
    t = time.perf_counter()
    rewritten: RewriteResult | None = None
    if history and rewriter is not None:
        try:
            valid = list(retrieval_service.collection_brands(client, settings.collection_name))
        except Exception as e:
            raise retrieval_service.VectorDatabaseUnavailable(f"Vector database unavailable: {e}") from e
        rewritten = rewrite(request.message, history, rewriter, valid, aliases, settings.rewrite_answer_chars)
        warnings += rewritten.warnings
    rewrite_ms = _ms(t) if rewritten else 0.0
    query = rewritten.query if rewritten else request.message

    # ---- 3. Brand precedence: request field > rewrite > detection in the question
    if request.brand:
        brand, brand_source = request.brand, "request"
        from_message = rewritten.brands if rewritten and rewritten.brands else detect_brands(request.message, aliases)
        if from_message and sorted(from_message) != sorted(request.brand):
            warnings.append({"type": "brand_conflict",
                             "detail": f"request brand {request.brand} overrides {from_message} from the message"})
    elif rewritten and rewritten.rewritten and rewritten.brands:
        brand, brand_source = rewritten.brands, "rewrite"
    else:
        brand, brand_source = None, "request"  # None: search() detects brands in the query

    t = time.perf_counter()
    found = retrieval_service.search(client, embedder, settings, request.to_search(query, brand), aliases,
                                     brand_source=brand_source)
    search_ms = _ms(t)
    rewritten_query = query if rewritten and rewritten.rewritten else None
    rewrite_info = (RewriteInfo(model=rewritten.model, prompt_version=REWRITE_PROMPT_VERSION,
                                input_tokens=rewritten.input_tokens, output_tokens=rewritten.output_tokens)
                    if rewritten else None)

    # ---- 4. Generate (skipped when nothing was retrieved)
    if not found.results:
        warnings.append({"type": "no_results", "detail": "Search returned no chunks; the LLM was not called"})
        response = ChatResponse(
            session_id=session_id, answer=NOT_FOUND_ANSWER, found_in_documents=False, sources=[],
            filters=found.filters, warnings=warnings, llm=LlmInfo(model=None, prompt_version=PROMPT_VERSION),
            rewritten_query=rewritten_query, history_used=len(history), rewrite=rewrite_info,
            timings_ms={"rewrite": rewrite_ms, "search": search_ms, "generate": 0.0},
        )
    else:
        context = build_context(found.results, settings.context_max_tokens)
        generated = generator.generate(SYSTEM_PROMPT, user_message(context.text, query, original=request.message))
        check = check_answer(generated.answer, context)
        by_n = {s.n: s for s in context.sources}
        response = ChatResponse(
            session_id=session_id,
            answer=generated.answer,
            found_in_documents=not check.not_found,
            sources=[_source(by_n[n]) for n in check.cited],
            filters=found.filters,
            warnings=warnings + check.warnings,
            llm=LlmInfo(
                model=generated.model, prompt_version=PROMPT_VERSION,
                input_tokens=generated.input_tokens, output_tokens=generated.output_tokens,
                reasoning_tokens=generated.reasoning_tokens,
                context_tokens=context.tokens, sources_dropped=context.dropped,
            ),
            rewritten_query=rewritten_query, history_used=len(history), rewrite=rewrite_info,
            timings_ms={"rewrite": rewrite_ms, "search": search_ms, "generate": generated.latency_ms},
            context=context.text if request.debug else None,
            retrieved=[_source(s) for s in context.sources] if request.debug else None,
        )

    # ---- 5. Save the turn: user + assistant together, only after a successful answer
    if memory_ok:
        now = datetime.now(timezone.utc)
        try:
            sessions.append_turn(
                session_id,
                StoredMessage(role="user", content=request.message, timestamp=now),
                StoredMessage(role="assistant", content=response.answer, timestamp=now,
                              rewritten_query=rewritten_query, brands=response.filters.brands,
                              sources=[s.citation for s in response.sources],
                              found_in_documents=response.found_in_documents),
            )
        except SessionUnavailable as e:
            response.warnings.append({"type": "history_not_saved", "detail": str(e)})

    response.timings_ms["total"] = _ms(started)
    logger.info("chat", extra={"fields": {
        "session_id": session_id, "message": request.message, "rewritten_query": rewritten_query,
        "history_used": len(history), "brands": response.filters.brands,
        "brand_source": response.filters.brand_source, "found": response.found_in_documents,
        "cited": [s.chunk_id for s in response.sources], "warnings": [w["type"] for w in response.warnings],
        "llm": response.llm.model_dump(), "timings_ms": response.timings_ms}})
    return response
