"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter.

Also provides :func:`structured_llm` — the single place where a Pydantic schema
is bound to the model — with a validated fallback for models/providers that do
not support the default structured-output mechanism.

The whole lab reads its model from ``.env`` (``LLM_MODEL`` / ``LLM_BASE_URL``);
nothing hardcodes a model name.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from common.schemas import PRAnalysis

M = TypeVar("M", bound=BaseModel)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

#: ``with_structured_output`` strategies, tried in order.
STRUCTURED_METHODS: tuple[str | None, ...] = (None, "json_schema", "function_calling")

#: Provider status codes that must surface to the caller instead of falling back.
_TRANSIENT_STATUSES = frozenset({401, 402, 403, 408, 409, 429, 500, 502, 503, 504})

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# Remembers, per model+messages-signature, whether native structured output works
# so we don't pay for a failed attempt on every LLM call in a session.
_UNSUPPORTED: set[str] = set()
_UNSUPPORTED_LOCK = threading.Lock()


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Build the chat model from environment variables.

    ``max_retries`` is bounded so a transient provider error surfaces as an
    exception the nodes can turn into a friendly message instead of hanging.
    """
    import os

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        temperature=temperature,
        max_retries=2,
    )


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply (```json fences etc.)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("model reply contained no JSON object")
    return match.group(0)


def is_transient_llm_error(exc: BaseException) -> bool:
    """True for provider failures that must NOT be masked by the JSON fallback.

    Auth (401), missing credits (402), forbidden (403), rate limit (429) and
    server/timeout errors are operational problems: retrying them as prose would
    hide the real cause from the operator.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int):
        return status in _TRANSIENT_STATUSES
    name = type(exc).__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutException",
        "RateLimitError",
        "AuthenticationError",
        "PermissionDeniedError",
    }


class StructuredOutputLLM:
    """Binds ``PRAnalysis`` to a chat model, with a validated fallback.

    Strategy:

    1. Try ``with_structured_output(PRAnalysis)`` using each method in
       :data:`STRUCTURED_METHODS` (native ``json_schema`` / ``function_calling``).
    2. If the provider rejects *all* of them because it does not support the
       mechanism, fall back to prompting for JSON in plain text and **validate**
       the reply with Pydantic — never trust the raw text.

    Transient provider failures (401/402/429/5xx/timeouts — auth, credits, rate
    limit) are re-raised untouched: they are not a structured-output
    incompatibility, and silently retrying them as prose would hide a real
    error from the operator.

    Both ``invoke`` and ``ainvoke`` are exposed so sync CLI graphs and async
    graphs share one implementation, and so tests can inject their own object
    with the same two methods.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        schema: type[M] = PRAnalysis,
        *,
        methods: tuple[str | None, ...] = STRUCTURED_METHODS,
    ) -> None:
        self._llm = llm
        self._schema = schema
        self._methods = methods
        model = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
        # Keyed on the instance too: two clients for the same model can be
        # configured differently, and a test's fake must not inherit another
        # fake's "unsupported" verdict.
        self._cache_key = f"{model}:{schema.__name__}:{id(llm)}"

    # ── native binding ────────────────────────────────────────────────────
    def _native(self, method: str | None):
        return self._llm.with_structured_output(self._schema, method=method)

    def _native_usable(self) -> bool:
        with _UNSUPPORTED_LOCK:
            return self._cache_key not in _UNSUPPORTED

    def _mark_unsupported(self) -> None:
        with _UNSUPPORTED_LOCK:
            if len(_UNSUPPORTED) > 64:  # bound the cache (keys hold object ids)
                _UNSUPPORTED.clear()
            _UNSUPPORTED.add(self._cache_key)

    def _repair_prompt(self, messages: list[Any]) -> list[Any]:
        """Append a JSON-schema instruction for the text fallback."""
        instruction = (
            "Respond with ONLY a single JSON object (no prose, no code fences) that "
            "validates against this JSON schema:\n"
            + json.dumps(self._schema.model_json_schema(), ensure_ascii=False)
        )
        return [*messages, {"role": "user", "content": instruction}]

    def _parse(self, content: Any) -> M:
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return self._schema.model_validate_json(_extract_json(str(content)))

    def _bind_or_raise(self, exc: Exception, method: str | None) -> None:
        """Record «unsupported» for mechanism failures; re-raise real errors."""
        if is_transient_llm_error(exc):
            raise exc
        logger.debug("structured output method=%r unusable: %s", method, exc)

    # ── public API ────────────────────────────────────────────────────────
    def invoke(self, messages: list[Any], **kwargs: Any) -> M:
        if self._native_usable():
            for method in self._methods:
                try:
                    return self._native(method).invoke(messages, **kwargs)
                except Exception as exc:
                    self._bind_or_raise(exc, method)
            self._mark_unsupported()
        reply = self._llm.invoke(self._repair_prompt(messages), **kwargs)
        return self._parse(reply.content)

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> M:
        if self._native_usable():
            for method in self._methods:
                try:
                    return await self._native(method).ainvoke(messages, **kwargs)
                except Exception as exc:
                    self._bind_or_raise(exc, method)
            self._mark_unsupported()
        reply = await self._llm.ainvoke(self._repair_prompt(messages), **kwargs)
        return self._parse(reply.content)


def structured_llm(
    schema: type[M] = PRAnalysis,
    *,
    llm: BaseChatModel | None = None,
    temperature: float = 0.2,
    methods: tuple[str | None, ...] = STRUCTURED_METHODS,
) -> StructuredOutputLLM:
    """Return a :class:`StructuredOutputLLM` bound to ``schema``.

    Pass ``llm`` to inject an alternative chat model (used by the test suite and
    by anyone wanting a different provider) — everything else keeps working.
    """
    return StructuredOutputLLM(
        llm if llm is not None else get_llm(temperature), schema, methods=methods
    )


def reset_structured_output_cache() -> None:
    """Forget which models were found to lack native structured output.

    Only useful for tests (and for a long-lived process that switched models);
    the cache is otherwise transparent.
    """
    with _UNSUPPORTED_LOCK:
        _UNSUPPORTED.clear()


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "STRUCTURED_METHODS",
    "StructuredOutputLLM",
    "get_llm",
    "is_transient_llm_error",
    "reset_structured_output_cache",
    "structured_llm",
    "ValidationError",
]
