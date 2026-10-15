"""End-to-end: PR URL → fetch → analyze → route → interrupt → human input →
resume → commit/reject → audit → replay.

One graph, one SQLite file, fakes at the two system boundaries (LLM and GitHub).
The checkpointer is a real ``AsyncSqliteSaver``, so this covers the full lab
pipeline including crash recovery — and never touches the network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import list_threads, replay_events
from common.schemas import ReviewComment
from exercises.agent import Mode, build_graph, config_for
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis

THREAD = "e2e-thread"


def _hitl_analysis():
    return make_analysis(
        0.65,
        summary="Adds an optional priority column to the task model.",
        risk_factors=["Schema migration needs a backfill plan"],
        comments=[ReviewComment(file="models/task.py", line=18, severity="issue", body="add a migration test")],
    )


def _escalation_analyses():
    initial = make_analysis(
        0.31,
        summary="Adds user login and cloud sync.",
        risk_factors=["MD5 password hashing", "SYNC_URL is plain HTTP"],
        comments=[ReviewComment(file="app/auth.py", line=12, severity="blocker", body="MD5 password hashing")],
        questions=["app/auth.py:12 — why MD5?", "sync/store.py:40 — is SYNC_URL HTTPS?"],
    )
    refined = make_analysis(
        0.62,
        summary="Adds login and sync; the transport is still plain HTTP.",
        risk_factors=["SYNC_URL is plain HTTP"],
        comments=[ReviewComment(file="app/auth.py", line=12, severity="blocker", body="MD5 password hashing")],
    )
    return initial, refined


async def _start(db: Path, *, analyses, github=None, mode=Mode.EXERCISE_5, dry_run=False):
    """Open a checkpointer + graph, run until the first interrupt, then stop.

    Returns ``(cm, app, paused_result)`` — deliberately mimicking a process that
    dies while a session is paused, keeping the context manager so the caller can
    close it (or simulate a crash by closing it and opening a new one).
    """
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    cp = await cm.__aenter__()
    await cp.setup()
    app = build_graph(mode=mode, checkpointer=cp, llm_factory=FakeLLMFactory(*analyses),
                      github=github or FakeGitHub(), dry_run=dry_run)
    result = await app.ainvoke(
        {"pr_url": PR_URL, "thread_id": THREAD, "reviewer_id": "tavan"}, config_for(THREAD)
    )
    return cm, app, result


async def _close(cm) -> None:
    await cm.__aexit__(None, None, None)


async def _resume(app, *answers):
    result = None
    for answer in answers:
        result = await app.ainvoke(Command(resume=answer), config_for(THREAD))
    return result


# ─── the four documented paths ─────────────────────────────────────────────


async def test_e2e_human_approval_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "e2e.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    gh = FakeGitHub()

    cp, app, paused = await _start(db, analyses=[_hitl_analysis()], github=gh)
    try:
        assert "__interrupt__" in paused
        payload = paused["__interrupt__"][0].value
        assert payload["kind"] == "approval_request"
        assert payload["confidence"] == pytest.approx(0.65)
        assert gh.posts == []  # nothing posted before approval

        final = await _resume(app, {"choice": "approve", "feedback": "ship it"})
    finally:
        await _close(cp)

    assert final["final_action"] == "committed"
    assert len(gh.posts) == 1
    body = gh.posts[0]["body"]
    assert "Automated review (confidence 65%)" in body
    assert "models/task.py:18" in body
    assert "Reviewer note: ship it" in body
    assert "Revert" not in body  # no accidental formal-approval wording

    # ── audit ──────────────────────────────────────────────────────────────
    events = await replay_events(THREAD)
    assert [e["action"] for e in events] == [
        "fetch_pr", "analyze", "route", "human_approval", "human_approval", "commit",
    ]
    assert all(e["execution_time_ms"] >= 0 for e in events)
    assert {e["risk_level"] for e in events} <= {"low", "med", "high"}
    responded = [e for e in events if e["action"] == "human_approval" and e["status"] == "responded"][0]
    assert responded["reviewer_id"] == "tavan"  # came from the graph state
    assert responded["decision"] == "approve"

    # ── replay / list ──────────────────────────────────────────────────────
    rows = [r for r in await list_threads() if r["thread_id"] == THREAD]
    assert rows and rows[0]["worst_risk"] == "med"
    assert rows[0]["events"] == 6
    assert rows[0]["last_action"] == "commit"

    # ── schema invariants, checked with plain sqlite3 ──────────────────────
    con = sqlite3.connect(db)
    try:
        bad = con.execute(
            "SELECT COUNT(*) FROM audit_events WHERE risk_level='high' AND decision='auto'"
        ).fetchone()[0]
        assert bad == 0  # the README's "shouldn't happen" query
        assert con.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0] == len(events)
        # Both mechanisms live in one file.
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "audit_events" in tables
        assert any(t.startswith("checkpoints") for t in tables)
    finally:
        con.close()


async def test_e2e_escalation_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "e2e-esc.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    gh = FakeGitHub()
    initial, refined = _escalation_analyses()

    cp, app, paused = await _start(db, analyses=[initial, refined], github=gh)
    try:
        payload = paused["__interrupt__"][0].value
        assert payload["kind"] == "escalation"
        assert len(payload["questions"]) == 2
        assert any("MD5" in rf for rf in payload["risk_factors"])

        # Answer the questions → the agent re-analyses and asks for confirmation.
        after_answers = await app.ainvoke(
            Command(resume={q: "answered" for q in payload["questions"]}), config_for(THREAD)
        )
        assert after_answers["__interrupt__"][0].value["kind"] == "final_confirmation"
        assert gh.posts == []  # still nothing posted

        final = await app.ainvoke(
            Command(resume={"confirm": True, "feedback": "post it"}), config_for(THREAD)
        )
    finally:
        await _close(cp)

    assert final["final_action"] == "committed"
    assert len(gh.posts) == 1
    body = gh.posts[0]["body"]
    assert "Reviewer answers to escalation questions" in body
    assert "MD5" in body

    events = await replay_events(THREAD)
    assert [e["action"] for e in events] == [
        "fetch_pr", "analyze", "route",
        "escalate", "escalate",
        "synthesize", "final_approval", "final_approval",
        "commit",
    ]
    assert next(e for e in events if e["action"] == "route")["reason"].startswith("confidence=0.31")
    synth = next(e for e in events if e["action"] == "synthesize")
    assert "0.31 → 0.62" in synth["reason"]
    # The unresolved blocker is still there after synthesis.
    assert not final["analysis"].comments == []
    assert any(c.severity == "blocker" for c in final["analysis"].comments)


async def test_e2e_auto_approve_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "e2e-auto.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    gh = FakeGitHub()

    cp, app, result = await _start(db, analyses=[make_analysis(0.94)], github=gh)
    try:
        assert "__interrupt__" not in result
    finally:
        await _close(cp)

    assert result["final_action"] == "auto_approved"
    assert len(gh.posts) == 1
    events = await replay_events(THREAD)
    assert [e["action"] for e in events] == ["fetch_pr", "analyze", "route", "auto_approve"]
    auto = events[-1]
    assert auto["decision"] == "auto"
    assert auto["status"] == "done"
    assert auto["reviewer_id"] is None
    assert auto["risk_level"] == "low"


async def test_e2e_reject_flow_posts_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "e2e-rej.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    gh = FakeGitHub()

    cp, app, paused = await _start(db, analyses=[_hitl_analysis()], github=gh)
    try:
        assert "__interrupt__" in paused
        final = await _resume(app, {"choice": "reject", "feedback": "needs a migration first"})
    finally:
        await _close(cp)

    assert final["final_action"] == "rejected"
    assert gh.posts == []
    events = await replay_events(THREAD)
    assert events[-1]["action"] == "commit"
    assert events[-1]["decision"] == "reject"
    assert "needs a migration first" == events[-2]["reason"]


# ─── durability ────────────────────────────────────────────────────────────


async def test_e2e_resumes_after_a_simulated_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brand-new process (new checkpointer, new graph) resumes the paused thread."""
    db = tmp_path / "e2e-restart.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    gh = FakeGitHub()

    cm1, app1, paused = await _start(db, analyses=[_hitl_analysis()], github=gh)
    await _close(cm1)  # the process "dies"

    # Second process: fresh connection and fresh graph object, same file.
    cm2 = AsyncSqliteSaver.from_conn_string(str(db))
    cp2 = await cm2.__aenter__()
    await cp2.setup()
    try:
        app2 = build_graph(
            mode=Mode.EXERCISE_5,
            checkpointer=cp2,
            llm_factory=FakeLLMFactory(),  # no analyses left: analyze must NOT re-run
            github=gh,
        )
        snapshot = await app2.aget_state(config_for(THREAD))
        assert snapshot.next, "checkpoint should still be paused"
        pending = [i.value for t in snapshot.tasks for i in (t.interrupts or ())]
        assert pending and pending[0]["kind"] == "approval_request"

        final = await app2.ainvoke(
            Command(resume={"choice": "approve", "feedback": "resumed"}), config_for(THREAD)
        )
    finally:
        await _close(cm2)

    assert final["final_action"] == "committed"
    assert len(gh.posts) == 1
    assert "Reviewer note: resumed" in gh.posts[0]["body"]
    # 6 events total — the resume did not re-run fetch/analyze in the new process.
    assert len(await replay_events(THREAD)) == 6
