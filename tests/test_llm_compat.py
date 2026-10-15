"""Structured-output compatibility (lab §9.4).

The lab must work with OpenRouter models whose structured-output support varies,
so ``common.llm.structured_llm`` tries the native bindings and falls back to
"ask for JSON, then validate with Pydantic". These tests pin that behaviour,
including the distinction between "this provider cannot do it" (fall back) and
"this request failed for an operational reason" (surface the error).
"""

from __future__ import annotations

from typing import Any

import pytest

from common import llm as llm_module
from common.llm import StructuredOutputLLM, is_transient_llm_error, structured_llm
from common.schemas import PRAnalysis

VALID_JSON = (
    '{"summary": "ok", "risk_factors": [], "comments": [], "confidence": 0.5,'
    ' "confidence_reasoning": "fine", "escalation_questions": []}'
)


@pytest.fixture(autouse=True)
def clean_cache():
    """The «unsupported» cache is process-global; keep tests independent."""
    llm_module.reset_structured_output_cache()
    yield
    llm_module.reset_structured_output_cache()


class FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    """Minimal ``BaseChatModel`` stand-in for the fallback path."""

    def __init__(
        self,
        *,
        content: str = VALID_JSON,
        bind_error: Exception | None = None,
        invoke_error: Exception | None = None,
        model_name: str = "fake/model",
    ) -> None:
        self.model_name = model_name
        self.content = content
        self.bind_error = bind_error
        self.invoke_error = invoke_error
        self.bind_calls: list[Any] = []
        self.prompts: list[list[Any]] = []

    # — the two hooks StructuredOutputLLM uses —
    def with_structured_output(self, schema: Any, *, method: Any = None):
        self.bind_calls.append(method)
        if self.bind_error is not None:
            raise self.bind_error

        class _Native:
            def invoke(self, messages, **kwargs):  # pragma: no cover - not used
                raise AssertionError("native path should not be used in this test")

            async def ainvoke(self, messages, **kwargs):  # pragma: no cover
                raise AssertionError("native path should not be used in this test")

        return _Native()

    def invoke(self, messages: list[Any], **kwargs: Any) -> FakeReply:
        self.prompts.append(messages)
        if self.invoke_error is not None:
            raise self.invoke_error
        return FakeReply(self.content)

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> FakeReply:
        return self.invoke(messages, **kwargs)


class HTTPStatusError(Exception):
    """Provider error carrying a status code, like ``openai.APIStatusError``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Error code: {status_code}")
        self.status_code = status_code


# ─── transient vs. incompatibility ─────────────────────────────────────────


@pytest.mark.parametrize("status", [401, 402, 403, 429, 500, 503])
def test_status_codes_that_are_operational_errors(status: int) -> None:
    assert is_transient_llm_error(HTTPStatusError(status)) is True


@pytest.mark.parametrize("status", [400, 404, 422])
def test_status_codes_that_mean_the_mechanism_is_unsupported(status: int) -> None:
    assert is_transient_llm_error(HTTPStatusError(status)) is False


def test_timeout_and_connection_errors_are_transient() -> None:
    assert is_transient_llm_error(TimeoutError("slow")) is True
    assert is_transient_llm_error(ConnectionError("down")) is True


# ─── fallback behaviour ────────────────────────────────────────────────────


def test_falls_back_to_validated_json_when_binding_fails() -> None:
    model = FakeChatModel(bind_error=ValueError("this model does not support tools"))
    bound = structured_llm(PRAnalysis, llm=model)
    result = bound.invoke([{"role": "user", "content": "review this"}])

    assert isinstance(result, PRAnalysis)
    assert result.confidence == 0.5
    # All native methods were attempted, then the model was asked for JSON.
    assert set(model.bind_calls) == set(llm_module.STRUCTURED_METHODS)
    assert "JSON schema" in model.prompts[0][-1]["content"]


def test_clickhouse_style_failure_is_remembered_so_later_calls_skip_the_probes() -> None:
    model = FakeChatModel(bind_error=RuntimeError("unsupported"))
    bound = structured_llm(PRAnalysis, llm=model, methods=(None,))
    bound.invoke([{"role": "user", "content": "1"}])
    bound.invoke([{"role": "user", "content": "2"}])
    assert len(model.bind_calls) == 1  # marked unsupported after the first failure


def test_json_in_a_code_fence_is_extracted() -> None:
    model = FakeChatModel(content=f"Sure!\n```json\n{VALID_JSON}\n```\n")
    bound = StructuredOutputLLM(model, PRAnalysis, methods=(None,))
    model.bind_error = RuntimeError("nope")
    assert bound.invoke([{"role": "user", "content": "x"}]).confidence == 0.5


def test_prose_wrapped_json_is_extracted() -> None:
    model = FakeChatModel(content=f"Here is my review: {VALID_JSON} Hope that helps.")
    model.bind_error = RuntimeError("nope")
    bound = StructuredOutputLLM(model, PRAnalysis, methods=(None,))
    assert bound.invoke([{"role": "user", "content": "x"}]).summary == "ok"


def test_invalid_json_raises_a_validation_error() -> None:
    model = FakeChatModel(content='{"summary": "ok", "confidence": 4.2}')
    model.bind_error = RuntimeError("nope")
    bound = StructuredOutputLLM(model, PRAnalysis, methods=(None,))
    with pytest.raises(Exception):
        bound.invoke([{"role": "user", "content": "x"}])


def test_non_json_reply_raises_instead_of_returning_junk() -> None:
    model = FakeChatModel(content="I cannot help with that.")
    model.bind_error = RuntimeError("nope")
    bound = StructuredOutputLLM(model, PRAnalysis, methods=(None,))
    with pytest.raises(ValueError, match="no JSON object"):
        bound.invoke([{"role": "user", "content": "x"}])


def test_credits_error_is_surfaced_not_masked_by_the_fallback() -> None:
    """A 402 "insufficient credits" is an account problem, not a JSON problem."""
    model = FakeChatModel(bind_error=HTTPStatusError(402))
    bound = structured_llm(PRAnalysis, llm=model)
    with pytest.raises(HTTPStatusError):
        bound.invoke([{"role": "user", "content": "x"}])
    assert model.prompts == []  # the JSON fallback was never attempted


def test_rate_limit_is_surfaced() -> None:
    model = FakeChatModel(bind_error=HTTPStatusError(429))
    bound = structured_llm(PRAnalysis, llm=model)
    with pytest.raises(HTTPStatusError):
        bound.invoke([{"role": "user", "content": "x"}])


async def test_async_fallback_matches_the_sync_one() -> None:
    model = FakeChatModel(bind_error=RuntimeError("unsupported"))
    bound = structured_llm(PRAnalysis, llm=model)
    result = await bound.ainvoke([{"role": "user", "content": "x"}])
    assert isinstance(result, PRAnalysis) and result.summary == "ok"


def test_the_model_name_is_read_from_env_not_hardcoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "some-provider/some-model")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    chat = llm_module.get_llm()
    assert chat.model_name == "some-provider/some-model"
    assert str(chat.openai_api_base) == "https://example.invalid/v1"


def test_get_llm_without_a_key_fails_with_a_helpful_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        llm_module.get_llm()
