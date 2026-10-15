"""Exercise 3 — escalation Q&A and synthesis.

Verifies that escalation is triggered below the threshold, that the reviewer
receives *specific* questions, that the answers reach the synthesis prompt, and
that the refined review cannot quietly drop an unresolved security risk or
inflate its confidence without an answer.
"""

from __future__ import annotations

import pytest
from common.db import memory_checkpointer
from langgraph.types import Command

from common.schemas import ReviewComment
from exercises.agent import Mode, build_graph, config_for
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis

ESCALATION_QUESTIONS = [
    "app/auth.py:12 — why MD5 instead of a password KDF like bcrypt/argon2?",
    "sync/store.py:40 — is SYNC_URL meant to be HTTPS in production, and who holds the token?",
    "sync/store.py — Storage.add builds SQL by string concatenation; is that input "
    "ever attacker-controlled?",
]


@pytest.fixture
def escalation_setup():
    initial = make_analysis(
        0.31,
        summary="Adds user login and cloud sync.",
        risk_factors=[
            "MD5 password hashing is cryptographically broken",
            "SYNC_URL is plain HTTP; the sync token is stored in plaintext",
            "SQL injection in Storage.add via string-concatenated query",
        ],
        comments=[
            ReviewComment(file="app/auth.py", line=12, severity="blocker", body="MD5 password hashing"),
            ReviewComment(file="sync/store.py", line=40, severity="blocker", body="plaintext token + HTTP upload"),
        ],
        questions=ESCALATION_QUESTIONS,
        reasoning="Multiple security red flags; intent of the sync path is unclear.",
    )
    refined = make_analysis(
        0.6,
        summary="Adds user login and cloud sync; sync URL and token handling still need work.",
        risk_factors=["SYNC_URL is plain HTTP; the sync token is stored in plaintext"],
        comments=[
            ReviewComment(file="app/auth.py", line=12, severity="blocker", body="MD5 password hashing"),
            ReviewComment(file="sync/store.py", line=40, severity="suggestion", body="prefer HTTPS"),
        ],
        reasoning="The reviewer confirmed the SQL path is parameterised but not the TLS migration.",
    )
    factory = FakeLLMFactory(initial, refined)
    gh = FakeGitHub()
    app = build_graph(
        mode=Mode.EXERCISE_3, checkpointer=memory_checkpointer(), llm_factory=factory, github=gh
    )
    return app, factory, gh, initial, refined


async def test_low_confidence_triggers_escalation_with_the_specific_questions(escalation_setup) -> None:
    app, _, gh, _, _ = escalation_setup
    result = await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "escalation"
    assert payload["confidence"] == pytest.approx(0.31)
    assert payload["questions"] == ESCALATION_QUESTIONS
    assert len(payload["questions"]) >= 2
    assert payload["risk_factors"]
    # Each question names the code it is about — not a generic prompt.
    assert all(".py" in q for q in payload["questions"])
    assert gh.posts == []  # nothing posted while we wait


async def test_answers_are_stored_in_state(escalation_setup) -> None:
    app, _, _, _, _ = escalation_setup
    await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))
    answers = {q: f"answer for {i}" for i, q in enumerate(ESCALATION_QUESTIONS)}
    final = await app.ainvoke(Command(resume=answers), config_for("esc"))
    assert final["escalation_answers"] == answers


async def test_synthesis_receives_diff_original_findings_and_answers(escalation_setup) -> None:
    app, factory, _, _, _ = escalation_setup
    await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))
    answers = {ESCALATION_QUESTIONS[0]: "MD5 is legacy; migrating next sprint."}
    await app.ainvoke(Command(resume=answers), config_for("esc"))

    prompt = factory.last_prompt()
    assert "app/auth.py" in prompt                       # the diff
    assert "MD5 password hashing" in prompt              # the original findings
    assert "MD5 is legacy; migrating next sprint." in prompt  # the reviewer's answer
    assert "<<<UNTRUSTED_PR_CONTENT>>>" in prompt        # still injection-fenced
    assert factory.call_count == 2                       # analyze + synthesize, once each


async def test_unresolved_blocker_survives_synthesis(escalation_setup) -> None:
    """The refined analysis must not drop a blocker the reviewer did not resolve."""
    app, _, gh, _, _ = escalation_setup
    await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))
    final = await app.ainvoke(
        Command(resume={ESCALATION_QUESTIONS[0]: "it's legacy code"}), config_for("esc")
    )
    refined = final["analysis"]
    bodies = [c.body for c in refined.comments]
    assert "MD5 password hashing" in bodies
    assert "plaintext token + HTTP upload" in bodies  # re-instated: the model dropped it
    assert any("HTTP" in rf for rf in refined.risk_factors)


async def test_confidence_is_not_inflated_without_answers() -> None:
    """Never raise confidence when the reviewer gave nothing to justify it."""
    initial = make_analysis(0.3, questions=["q1"])
    refined = make_analysis(0.95, summary="Looks fine now.", risk_factors=[], comments=[])
    app = build_graph(
        mode=Mode.EXERCISE_3,
        checkpointer=memory_checkpointer(),
        llm_factory=FakeLLMFactory(initial, refined),
        github=FakeGitHub(),
    )
    await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))
    final = await app.ainvoke(Command(resume={}), config_for("esc"))
    assert final["analysis"].confidence <= initial.confidence


async def test_refined_review_is_posted_after_escalation(escalation_setup) -> None:
    app, _, gh, _, _ = escalation_setup
    await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))
    final = await app.ainvoke(
        Command(resume={q: "answered" for q in ESCALATION_QUESTIONS}), config_for("esc")
    )
    assert final["final_action"] == "committed"
    assert len(gh.posts) == 1
    body = gh.posts[0]["body"]
    assert "Reviewer answers to escalation questions" in body
    assert ESCALATION_QUESTIONS[0] in body


async def test_synthesis_failure_keeps_the_session_alive_and_posts_nothing() -> None:
    initial = make_analysis(0.3, questions=["q1", "q2"])
    factory = FakeLLMFactory(initial, RuntimeError("provider down"))
    gh = FakeGitHub()
    app = build_graph(
        mode=Mode.EXERCISE_3, checkpointer=memory_checkpointer(), llm_factory=factory, github=gh
    )
    await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))
    final = await app.ainvoke(Command(resume={"q1": "a", "q2": "b"}), config_for("esc"))
    assert final["final_action"] == "rejected"
    assert "provider down" in final["analysis_error"]
    assert gh.posts == []


async def test_fallback_questions_exist_when_the_llm_returns_none() -> None:
    """A low-confidence analysis with no questions must still ask something specific."""
    app = build_graph(
        mode=Mode.EXERCISE_3,
        checkpointer=memory_checkpointer(),
        llm_factory=FakeLLMFactory(make_analysis(0.2, questions=[])),
        github=FakeGitHub(),
    )
    result = await app.ainvoke({"pr_url": PR_URL, "thread_id": "esc"}, config_for("esc"))
    payload = result["__interrupt__"][0].value
    assert len(payload["questions"]) >= 2
    assert all(q.strip() for q in payload["questions"])
