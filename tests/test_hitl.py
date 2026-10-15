"""Exercise 2 — ``interrupt()`` / ``Command(resume=...)`` behaviour.

Drives the real graph with a ``MemorySaver`` checkpointer and a fake LLM/GitHub,
so the pause-and-resume mechanics are exercised for real: payload shape, all
three choices, thread identity, state preservation, no duplicate LLM calls and
no comment posted before (or twice after) approval.
"""

from __future__ import annotations

import pytest
from common.db import memory_checkpointer
from langgraph.types import Command

from exercises.agent import Mode, build_graph, config_for
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis

OPEN_PR = {"pr_url": PR_URL, "thread_id": "t-hitl"}


@pytest.fixture
def graph_factory():
    """Builds an exercise-2 graph with a fresh MemorySaver per test."""

    def _make(*analyses, github=None, mode=Mode.EXERCISE_2, dry_run=False):
        factory = FakeLLMFactory(*analyses)
        gh = github or FakeGitHub()
        app = build_graph(
            mode=mode, checkpointer=memory_checkpointer(), llm_factory=factory, github=gh, dry_run=dry_run
        )
        return app, factory, gh

    return _make


async def test_interrupt_payload_shape_is_exactly_the_approval_request(graph_factory) -> None:
    app, _, gh = graph_factory(make_analysis(0.65, reasoning="One open schema question."))
    result = await app.ainvoke(OPEN_PR, config_for("t-hitl"))

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["kind"] == "approval_request"
    assert payload["confidence"] == pytest.approx(0.65)
    assert payload["confidence_reasoning"] == "One open schema question."
    assert payload["summary"]
    assert payload["comments"] and {"file", "line", "severity", "body"} <= set(payload["comments"][0])
    assert payload["diff_preview"]
    assert payload["pr_url"] == PR_URL
    # Nothing has been posted just because the graph paused.
    assert gh.posts == []


async def test_approve_resume_posts_the_comment_once(graph_factory) -> None:
    app, factory, gh = graph_factory(make_analysis(0.65))
    first = await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    assert "__interrupt__" in first

    final = await app.ainvoke(
        Command(resume={"choice": "approve", "feedback": "lgtm"}), config_for("t-hitl")
    )
    assert final["final_action"] == "committed"
    assert final["human_choice"] == "approve"
    assert final["human_feedback"] == "lgtm"
    assert len(gh.posts) == 1
    body = gh.posts[0]["body"]
    assert "Reviewer note: lgtm" in body
    assert "confidence 65%" in body
    # The LLM ran exactly once: resume must not re-run analyze.
    assert factory.call_count == 1


async def test_reject_resume_posts_nothing(graph_factory) -> None:
    app, factory, gh = graph_factory(make_analysis(0.65))
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    final = await app.ainvoke(
        Command(resume={"choice": "reject", "feedback": "too risky"}), config_for("t-hitl")
    )
    assert final["final_action"] == "rejected"
    assert final["human_choice"] == "reject"
    assert gh.posts == []
    assert factory.call_count == 1


async def test_edit_choice_is_recorded_and_does_not_post_on_its_own(graph_factory) -> None:
    """``edit`` is its own outcome — never silently collapsed into ``reject``."""
    app, _, gh = graph_factory(make_analysis(0.65))
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    final = await app.ainvoke(
        Command(resume={"choice": "edit", "feedback": "soften the migration comment"}),
        config_for("t-hitl"),
    )
    assert final["human_choice"] == "edit"
    assert final["human_feedback"] == "soften the migration comment"
    assert final["final_action"] == "rejected"
    assert gh.posts == []


async def test_edit_then_approve_posts_the_reviewer_feedback(graph_factory) -> None:
    """The edit path can still end in a posted review once the human confirms."""
    app, _, gh = graph_factory(make_analysis(0.65))
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    final = await app.ainvoke(
        Command(resume={"choice": "edit", "feedback": "rewritten by reviewer"}),
        config_for("t-hitl"),
    )
    assert final["final_action"] == "rejected"

    # Confirming the edited review is a second run on the same thread: the
    # reviewer approves with the edited feedback attached.
    app2, _, gh2 = graph_factory(make_analysis(0.65))
    await app2.ainvoke(OPEN_PR, config_for("t-hitl"))
    final2 = await app2.ainvoke(
        Command(resume={"choice": "approve", "feedback": "rewritten by reviewer"}),
        config_for("t-hitl"),
    )
    assert final2["final_action"] == "committed"
    assert "Reviewer note: rewritten by reviewer" in gh2.posts[0]["body"]


async def test_state_is_preserved_across_the_resume(graph_factory) -> None:
    app, _, _ = graph_factory(make_analysis(0.65, summary="A precise summary."))
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    final = await app.ainvoke(Command(resume={"choice": "approve", "feedback": ""}), config_for("t-hitl"))

    # fetch_pr/analyze output survived the interrupt.
    assert final["pr_title"] == "feat: add priority field to tasks"
    assert final["pr_files"] == ["app/auth.py"]
    assert final["pr_head_sha"] == "abcdef1234567890"
    assert final["analysis"].summary == "A precise summary."
    assert final["decision"] == "human_approval"


async def test_resume_with_a_different_thread_id_restarts_instead_of_resuming(graph_factory) -> None:
    app, factory, _ = graph_factory(make_analysis(0.65), make_analysis(0.65))
    await app.ainvoke(OPEN_PR, config_for("thread-a"))
    # A different thread_id has no state → the graph starts over (and pauses again).
    other = await app.ainvoke(
        {"pr_url": PR_URL, "thread_id": "thread-b"}, config_for("thread-b")
    )
    assert other.get("final_action") != "committed"
    assert "__interrupt__" in other
    assert factory.call_count == 2  # analyze ran once per thread


async def test_resuming_without_state_fails_loudly_instead_of_crashing(graph_factory) -> None:
    """A resume on a thread that was never started must not raise KeyError."""
    app, factory, gh = graph_factory(make_analysis(0.65))
    result = await app.ainvoke(Command(resume={"choice": "approve"}), config_for("never-started"))
    assert result["final_action"] == "fetch_failed"
    assert "pr_url" in result["fetch_error"]
    assert gh.posts == [] and factory.call_count == 0


async def test_checkpoint_can_be_read_back_and_the_thread_recovered(graph_factory) -> None:
    app, _, _ = graph_factory(make_analysis(0.65))
    await app.ainvoke(OPEN_PR, config_for("t-recover"))

    snapshot = await app.aget_state(config_for("t-recover"))
    assert snapshot.next  # still paused
    assert snapshot.values["pr_url"] == PR_URL

    payload = None
    for task in snapshot.tasks:
        for intr in task.interrupts or ():
            payload = intr.value
    assert payload is not None and payload["kind"] == "approval_request"

    # Resume straight from the recovered payload, as the Streamlit UI does.
    final = await app.ainvoke(Command(resume={"choice": "approve", "feedback": ""}), config_for("t-recover"))
    assert final["final_action"] == "committed"


async def test_interrupt_is_not_reached_when_confidence_is_high(graph_factory) -> None:
    app, _, gh = graph_factory(make_analysis(0.95))
    final = await app.ainvoke(OPEN_PR, config_for("t-auto"))
    assert "__interrupt__" not in final
    assert final["final_action"] == "auto_approved"
    assert len(gh.posts) == 1


async def test_auto_approve_never_posts_a_security_pr(graph_factory) -> None:
    from common.schemas import ReviewComment

    analysis = make_analysis(
        0.99,
        comments=[ReviewComment(file="a.py", line=3, severity="blocker", body="MD5 hashing")],
        risk_factors=["Weak password hashing"],
    )
    app, _, gh = graph_factory(analysis)
    result = await app.ainvoke(OPEN_PR, config_for("t-sec"))
    # Route sends it to a human instead, so the graph pauses and posts nothing.
    assert "__interrupt__" in result
    assert gh.posts == []


async def test_github_post_failure_is_reported_not_swallowed(graph_factory) -> None:
    from common.github import GitHubPermissionError

    gh = FakeGitHub(post_error=GitHubPermissionError("403 forbidden"))
    app, _, _ = graph_factory(make_analysis(0.65), github=gh)
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    final = await app.ainvoke(Command(resume={"choice": "approve", "feedback": ""}), config_for("t-hitl"))
    assert final["final_action"] == "commit_failed"
    assert final["post_error"] == "403 forbidden"


async def test_thread_id_is_used_as_the_posting_idempotency_key() -> None:
    """The idempotency key travels with the post so a retry cannot duplicate it."""
    factory = FakeLLMFactory(make_analysis(0.65))
    gh = FakeGitHub()
    app = build_graph(mode=Mode.EXERCISE_2, checkpointer=memory_checkpointer(), llm_factory=factory, github=gh)
    await app.ainvoke({"pr_url": PR_URL, "thread_id": "t-dup"}, config_for("t-dup"))
    await app.ainvoke(Command(resume={"choice": "approve"}), config_for("t-dup"))
    assert len(gh.posts) == 1
    assert gh.posts[0]["thread_id"] == "t-dup"
    assert gh.posts[0]["pr_url"] == PR_URL


async def test_dry_run_never_calls_github_post(graph_factory) -> None:
    gh = FakeGitHub()
    app, _, _ = graph_factory(make_analysis(0.65), github=gh, dry_run=True)
    await app.ainvoke(OPEN_PR, config_for("t-dry"))
    final = await app.ainvoke(Command(resume={"choice": "approve"}), config_for("t-dry"))
    assert final["final_action"] == "committed_dry_run"
    assert gh.posts == []
    assert final["posted_comment_body"]


async def test_low_confidence_escalates_and_never_posts(graph_factory) -> None:
    """The <58% bucket must ask questions, not offer approve/reject."""
    initial = make_analysis(0.2, questions=["app/auth.py:12 — why MD5?"])
    refined = make_analysis(0.6)
    gh = FakeGitHub()
    app, factory, _ = graph_factory(initial, refined, github=gh, mode=Mode.EXERCISE_2)

    paused = await app.ainvoke(OPEN_PR, config_for("t-esc"))
    payload = paused["__interrupt__"][0].value
    assert payload["kind"] == "escalation"
    assert payload["questions"] == ["app/auth.py:12 — why MD5?"]

    final = await app.ainvoke(
        Command(resume={"app/auth.py:12 — why MD5?": "legacy"}), config_for("t-esc")
    )
    assert final["escalation_answers"] == {"app/auth.py:12 — why MD5?": "legacy"}
    assert final["analysis"].confidence == pytest.approx(0.6)  # refined
    assert gh.posts == []  # exercise 2 stops after synthesize


@pytest.mark.parametrize(
    ("mode", "expects_final_approval"),
    [
        (Mode.EXERCISE_2, False),   # stops at synthesize
        (Mode.EXERCISE_3, False),   # posts the refined review directly
        (Mode.EXERCISE_4, True),    # asks for confirmation first
        (Mode.EXERCISE_5, True),
    ],
)
def test_escalation_wiring_per_mode(mode: Mode, expects_final_approval: bool) -> None:
    """The final-confirmation gate exists exactly where the README requires it."""
    app = build_graph(
        mode=mode,
        checkpointer=memory_checkpointer(),
        llm_factory=FakeLLMFactory(make_analysis(0.2)),
        github=FakeGitHub(),
    )
    assert ("final_approval" in app.nodes) is expects_final_approval
    assert "synthesize" in app.nodes
