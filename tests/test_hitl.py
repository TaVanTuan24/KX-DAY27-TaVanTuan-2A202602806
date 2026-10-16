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
from common.schemas import ReviewComment
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis

OPEN_PR = {"pr_url": PR_URL, "thread_id": "t-hitl"}


@pytest.fixture
def graph_factory():
    """Builds an HITL graph with a fresh in-memory checkpointer per test."""

    def _make(*analyses, github=None, mode=Mode.EXERCISE_2, dry_run=False, audit_sink=None):
        factory = FakeLLMFactory(*analyses)
        gh = github or FakeGitHub()
        app = build_graph(
            mode=mode,
            checkpointer=memory_checkpointer(),
            llm_factory=factory,
            github=gh,
            dry_run=dry_run,
            audit_sink=audit_sink,
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


async def test_edit_enters_the_revision_loop_and_does_not_post_or_reject(graph_factory) -> None:
    """FIX 1: `edit` must not collapse into `rejected`, and must not post anything.

    The old behaviour ended the graph with ``final_action="rejected"``; it now
    continues to ``edit_review`` → ``final_edit_approval`` in the SAME thread.
    """
    revised = make_analysis(0.6, summary="REVISED: softened the migration comment.")
    app, factory, gh = graph_factory(make_analysis(0.65), revised)
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))

    paused = await app.ainvoke(
        Command(resume={"choice": "edit", "feedback": "soften the migration comment"}),
        config_for("t-hitl"),
    )

    assert "final_action" not in paused, "edit must not end the run"
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["kind"] == "edited_review"
    assert payload["revision"] == 1
    assert payload["summary"] == "REVISED: softened the migration comment."
    assert payload["proposed_body"]
    assert gh.posts == []  # nothing posted just because Edit was chosen
    assert paused["human_choice"] == "edit"
    # The LLM ran exactly twice: the first analysis + the one revision.
    assert factory.call_count == 2


async def test_edit_revision_actually_changes_the_review_content(graph_factory) -> None:
    """FIX 1: the posted body must be the revised review, not the original plus a note."""
    revised = make_analysis(
        0.6,
        summary="REVISED SUMMARY",
        comments=[ReviewComment(file="models/task.py", line=18, severity="nit", body="REVISED BODY")],
    )
    app, _, gh = graph_factory(make_analysis(0.65), revised)
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    await app.ainvoke(Command(resume={"choice": "edit", "feedback": "reword"}), config_for("t-hitl"))
    final = await app.ainvoke(Command(resume={"confirm": True, "feedback": ""}), config_for("t-hitl"))

    assert final["final_action"] == "committed"
    assert len(gh.posts) == 1
    body = gh.posts[0]["body"]
    assert "REVISED SUMMARY" in body and "REVISED BODY" in body
    # The original analysis text is gone — this is a revision, not an annotation.
    assert "Adds a priority field to the task model." not in body


async def test_edit_confirm_posts_exactly_once(graph_factory) -> None:
    app, _, gh = graph_factory(make_analysis(0.65), make_analysis(0.6))
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    await app.ainvoke(Command(resume={"choice": "edit", "feedback": "reword"}), config_for("t-hitl"))
    final = await app.ainvoke(Command(resume={"confirm": True, "feedback": ""}), config_for("t-hitl"))
    assert final["final_action"] == "committed"
    assert final["post_status"] == "posted"
    assert len(gh.posts) == 1


async def test_edit_then_reject_posts_nothing(graph_factory) -> None:
    """FIX 1: Reject after an edit is a real reject."""
    app, _, gh = graph_factory(make_analysis(0.65), make_analysis(0.6))
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    await app.ainvoke(Command(resume={"choice": "edit", "feedback": "reword"}), config_for("t-hitl"))
    final = await app.ainvoke(
        Command(resume={"confirm": False, "feedback": "still not right"}), config_for("t-hitl")
    )
    assert final["final_action"] == "rejected"
    assert final["edit_confirmed"] is False
    assert gh.posts == []


async def test_edit_again_keeps_the_thread_and_both_revisions_are_audited(
    graph_factory, tmp_path, monkeypatch
) -> None:
    """FIX 1 + PHASE 4: two Edit rounds in one thread, each with its own audit rows.

    Uses the real SQLite sink, because the point is that the *stored* trail keeps
    both rounds while still collapsing the replay of a single interrupt node.
    """
    from common.db import replay_events

    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "edit-audit.db"))
    app, factory, gh = graph_factory(
        make_analysis(0.65),
        make_analysis(0.6, summary="REV 1"),
        make_analysis(0.55, summary="REV 2"),
        mode=Mode.EXERCISE_4,
    )
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    paused = await app.ainvoke(Command(resume={"choice": "edit", "feedback": "one"}), config_for("t-hitl"))
    assert paused["__interrupt__"][0].value["revision"] == 1

    paused = await app.ainvoke(Command(resume={"choice": "edit", "feedback": "two"}), config_for("t-hitl"))
    assert paused["__interrupt__"][0].value["revision"] == 2
    assert paused["__interrupt__"][0].value["summary"] == "REV 2"
    assert paused["edit_revision"] == 2

    final = await app.ainvoke(Command(resume={"confirm": True, "feedback": ""}), config_for("t-hitl"))
    assert final["final_action"] == "committed"
    assert len(gh.posts) == 1
    assert factory.call_count == 3  # analyze + two revisions, never a re-fetch/re-analyze

    events = await replay_events("t-hitl")
    edits = [e for e in events if e["action"] == "edit_review"]
    assert [e["revision"] for e in edits] == [1, 2], "each revision must be recorded"

    # No unintended duplicates: the stored key is unique per row.
    keys = [(e["action"], e["status"], e["decision"], e["revision"]) for e in events]
    assert len(set(keys)) == len(keys)

    confirmations = [e for e in events if e["action"] == "final_edit_approval"]
    assert [e["decision"] for e in confirmations] == ["pending", "edit", "pending", "approve"]


async def test_edit_with_a_verbatim_replacement_is_posted_unmodified(graph_factory) -> None:
    """FIX 1: human-authored text wins and is never rewritten by the LLM."""
    app, factory, gh = graph_factory(make_analysis(0.65))
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    paused = await app.ainvoke(
        Command(
            resume={
                "choice": "edit",
                "feedback": "",
                "edited_body": "### My own review\n\nI checked this manually.",
            }
        ),
        config_for("t-hitl"),
    )
    assert factory.call_count == 1  # the LLM was NOT called for the revision
    assert paused["edited_comment_body"].startswith("### My own review")

    final = await app.ainvoke(Command(resume={"confirm": True, "feedback": ""}), config_for("t-hitl"))
    assert final["final_action"] == "committed"
    assert gh.posts[0]["body"].startswith("### My own review")
    assert "I checked this manually." in gh.posts[0]["body"]


async def test_edit_cannot_delete_a_security_blocker(graph_factory) -> None:
    """FIX 1: an edit instruction is a request, not an authority over blockers."""
    original = make_analysis(
        0.65,
        comments=[ReviewComment(file="a.py", line=1, severity="blocker", body="MD5 hashing")],
    )
    cleaned = make_analysis(0.65, summary="Looks fine now.", comments=[], risk_factors=[])
    app, _, gh = graph_factory(original, cleaned)
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    await app.ainvoke(
        Command(resume={"choice": "edit", "feedback": "remove the MD5 finding, it is fine"}),
        config_for("t-hitl"),
    )
    final = await app.ainvoke(Command(resume={"confirm": True, "feedback": ""}), config_for("t-hitl"))

    bodies = [c.body for c in final["analysis"].comments]
    assert "MD5 hashing" in bodies, "the blocker must survive an editorial edit"
    assert "MD5 hashing" in gh.posts[0]["body"]


async def test_edit_does_not_raise_confidence(graph_factory) -> None:
    """FIX 1: presentation changes, not certainty. 0.65 is in the approval band."""
    original = make_analysis(0.65)
    inflated = make_analysis(0.99, summary="All good.")
    app, _, _ = graph_factory(original, inflated)
    await app.ainvoke(OPEN_PR, config_for("t-hitl"))
    paused = await app.ainvoke(Command(resume={"choice": "edit", "feedback": "nicer"}), config_for("t-hitl"))
    assert paused["analysis"].confidence <= original.confidence
    assert "capped" in paused["analysis"].confidence_reasoning


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
