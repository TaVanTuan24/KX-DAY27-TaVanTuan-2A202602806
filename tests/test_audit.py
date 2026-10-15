"""Exercise 4 — structured SQLite audit trail, idempotency and replay.

These tests run against a real ``AsyncSqliteSaver`` in a temp directory, so the
checkpointer, the ``audit_events`` schema, the idempotency migration and the
replay queries are all exercised — with no API key and nothing posted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import (
    IDEMPOTENCY_COLUMNS,
    db_conn,
    list_threads,
    replay_events,
    write_audit_event,
)
from common.schemas import AuditEntry, risk_level_for
from exercises.agent import Mode, build_graph, config_for
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis

EXPECTED_ACTIONS = {
    "fetch_pr",
    "analyze",
    "route",
    "human_approval",
    "escalate",
    "synthesize",
    "final_approval",
    "commit",
}


async def _run(
    tmp_path: Path,
    thread_id: str,
    analyses,
    *,
    resume_values=None,
    github=None,
    mode=Mode.EXERCISE_4,
    dry_run=False,
):
    """Run the exercise-4 graph on its own DB, resuming through the interrupts."""
    db = tmp_path / "hitl.db"
    factory = FakeLLMFactory(*analyses)
    gh = github or FakeGitHub()
    app = None
    results = []
    async with AsyncSqliteSaver.from_conn_string(str(db)) as cp:
        await cp.setup()
        app = build_graph(
            mode=mode, checkpointer=cp, llm_factory=factory, github=gh, dry_run=dry_run
        )
        result = await app.ainvoke({"pr_url": PR_URL, "thread_id": thread_id}, config_for(thread_id))
        results.append(result)
        for resume in resume_values or []:
            result = await app.ainvoke(Command(resume=resume), config_for(thread_id))
            results.append(result)
    return results, factory, gh, db


# ─── the audit() helper itself ─────────────────────────────────────────────


async def test_write_audit_event_round_trips_every_field(audit_db) -> None:
    entry = AuditEntry(
        agent_id="pr-review-agent@v0.1",
        action="route",
        confidence=0.61,
        risk_level="med",
        reviewer_id="tavan",
        decision="pending",
        status="done",
        reason="confidence=0.61 → human_approval",
        execution_time_ms=42,
    )
    await write_audit_event(thread_id="t1", pr_url=PR_URL, entry=entry)
    events = await replay_events("t1")
    assert len(events) == 1
    ev = events[0]
    assert ev["action"] == "route"
    assert ev["confidence"] == pytest.approx(0.61)
    assert ev["risk_level"] == "med"
    assert ev["reviewer_id"] == "tavan"
    assert ev["decision"] == "pending"
    assert ev["status"] == "done"
    assert ev["reason"] == "confidence=0.61 → human_approval"
    assert ev["execution_time_ms"] == 42
    assert ev["timestamp"]


async def test_duplicate_audit_event_is_ignored(audit_db) -> None:
    entry = AuditEntry(
        agent_id="a", action="human_approval", confidence=0.6, risk_level="med",
        decision="pending", status="waiting", reason="requested", execution_time_ms=1,
    )
    for _ in range(3):
        await write_audit_event(thread_id="t1", pr_url=PR_URL, entry=entry)
    assert len(await replay_events("t1")) == 1


async def test_a_different_outcome_is_not_deduplicated(audit_db) -> None:
    """Only *identical* repeats collapse — a changed decision is a new event."""
    for decision in ("pending", "approve"):
        await write_audit_event(
            thread_id="t1", pr_url=PR_URL,
            entry=AuditEntry(
                agent_id="a", action="human_approval", confidence=0.6, risk_level="med",
                decision=decision, status="responded" if decision != "pending" else "waiting",
                reason=decision, execution_time_ms=1,
            ),
        )
    assert len(await replay_events("t1")) == 2


async def test_schema_has_the_idempotency_index(audit_db) -> None:
    await write_audit_event(
        thread_id="t1", pr_url=PR_URL,
        entry=AuditEntry(
            agent_id="a", action="analyze", confidence=0.5, risk_level="high",
            decision="pending", status="done", reason="r", execution_time_ms=1,
        ),
    )
    async with db_conn() as conn:
        async with conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='audit_events_idempotency_idx'"
        ) as cur:
            row = await cur.fetchone()
    assert row is not None
    for column in IDEMPOTENCY_COLUMNS:
        assert column in row["sql"]


def test_legacy_database_is_migrated_and_deduplicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old DB (no `status` column, duplicate rows) is repaired on first use."""
    import asyncio

    db = tmp_path / "legacy.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    legacy_ddl = """
        CREATE TABLE audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            pr_url TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            action TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            reviewer_id TEXT,
            decision TEXT NOT NULL,
            reason TEXT,
            execution_time_ms INTEGER NOT NULL
        );
    """
    con = sqlite3.connect(db)
    con.executescript(legacy_ddl)
    for _ in range(3):  # the duplicate bug the index prevents
        con.execute(
            "INSERT INTO audit_events (timestamp, thread_id, pr_url, agent_id, action,"
            " confidence, risk_level, decision, reason, execution_time_ms)"
            " VALUES ('2026-01-01T00:00:00Z','t1','u','a','human_approval',0.6,'med',"
            "'pending','r',1)"
        )
    con.commit()
    con.close()

    async def migrate():
        async with db_conn() as conn:
            async with conn.execute("SELECT COUNT(*) AS n FROM audit_events") as cur:
                return (await cur.fetchone())["n"]

    assert asyncio.run(migrate()) == 1  # collapsed to the earliest row


# ─── the full exercise-4 run ───────────────────────────────────────────────


async def test_human_approval_run_writes_the_expected_audit_trail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    results, factory, gh, _ = await _run(
        tmp_path,
        "t-audit",
        [make_analysis(0.65)],
        resume_values=[{"choice": "approve", "feedback": "ship it"}],
    )
    assert results[-1]["final_action"] == "committed"
    assert len(gh.posts) == 1

    events = await replay_events("t-audit")
    actions = [e["action"] for e in events]
    assert actions == ["fetch_pr", "analyze", "route", "human_approval", "human_approval", "commit"]

    waiting = next(e for e in events if e["action"] == "human_approval" and e["status"] == "waiting")
    responded = next(e for e in events if e["action"] == "human_approval" and e["status"] == "responded")
    assert waiting["decision"] == "pending"
    assert responded["decision"] == "approve"
    assert responded["reason"] == "ship it"
    assert responded["reviewer_id"] == "octocat"  # from GitHubService.current_user()

    commit = events[-1]
    assert commit["decision"] == "committed"
    assert commit["risk_level"] == "med"
    assert commit["confidence"] == pytest.approx(0.65)


async def test_resumed_node_does_not_duplicate_the_waiting_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core idempotency requirement: `interrupt()` re-runs the node body."""
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    await _run(
        tmp_path,
        "t-dup",
        [make_analysis(0.65)],
        resume_values=[{"choice": "reject", "feedback": ""}],
    )
    events = await replay_events("t-dup")
    waiting = [e for e in events if e["action"] == "human_approval" and e["status"] == "waiting"]
    assert len(waiting) == 1
    assert len(events) == len({(e["action"], e["status"], e["decision"]) for e in events})


async def test_escalation_run_records_the_whole_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    initial = make_analysis(
        0.3,
        risk_factors=["MD5 password hashing"],
        questions=["app/auth.py:12 — why MD5?", "sync/store.py:40 — is SYNC_URL HTTPS?"],
    )
    refined = make_analysis(0.62, risk_factors=["MD5 password hashing"])
    answers = {
        "app/auth.py:12 — why MD5?": "legacy",
        "sync/store.py:40 — is SYNC_URL HTTPS?": "staging only",
    }
    results, _, gh, _ = await _run(
        tmp_path,
        "t-esc",
        [initial, refined],
        resume_values=[answers, {"confirm": True, "feedback": "ok"}],
    )
    assert results[-1]["final_action"] == "committed"
    assert len(gh.posts) == 1

    events = await replay_events("t-esc")
    actions = [e["action"] for e in events]
    assert actions == [
        "fetch_pr", "analyze", "route",
        "escalate", "escalate",
        "synthesize", "final_approval", "final_approval",
        "commit",
    ]
    assert EXPECTED_ACTIONS >= {e["action"] for e in events}
    synth = next(e for e in events if e["action"] == "synthesize")
    assert synth["confidence"] == pytest.approx(0.62)
    assert "0.30 → 0.62" in synth["reason"]
    assert synth["risk_level"] == "high"  # MD5 still unresolved


async def test_escalation_without_final_confirmation_posts_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    initial = make_analysis(0.3, questions=["q1"])
    refined = make_analysis(0.6)
    results, _, gh, _ = await _run(
        tmp_path, "t-noconfirm", [initial, refined],
        resume_values=[{"q1": "a"}, {"confirm": False, "feedback": "not yet"}],
    )
    assert results[-1]["final_action"] == "rejected"
    assert gh.posts == []
    commit = [e for e in await replay_events("t-noconfirm") if e["action"] == "commit"][0]
    assert commit["decision"] == "reject"
    assert "did not confirm" in commit["reason"]


async def test_auto_approve_run_is_audited_as_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    results, factory, gh, _ = await _run(tmp_path, "t-auto", [make_analysis(0.95)])
    assert results[-1]["final_action"] == "auto_approved"
    assert len(gh.posts) == 1
    events = await replay_events("t-auto")
    assert [e["action"] for e in events] == ["fetch_pr", "analyze", "route", "auto_approve"]
    auto = events[-1]
    assert auto["decision"] == "auto"
    assert auto["risk_level"] == risk_level_for(0.95) == "low"
    assert auto["reviewer_id"] is None


async def test_failed_post_is_audited_as_failed_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from common.github import GitHubPermissionError

    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    gh = FakeGitHub(post_error=GitHubPermissionError("403 — token lacks scope"))
    results, _, _, _ = await _run(tmp_path, "t-fail", [make_analysis(0.95)], github=gh)
    assert results[-1]["final_action"] == "auto_approve_failed"
    assert "403" in results[-1]["post_error"]
    commit = [e for e in await replay_events("t-fail") if e["action"] == "auto_approve"][0]
    assert commit["decision"] == "failed"
    assert commit["status"] == "failed"


async def test_dry_run_is_never_audited_as_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An auditor filtering on `decision` must not read a rehearsal as a post."""
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    gh = FakeGitHub()
    results, _, _, _ = await _run(
        tmp_path, "t-dry", [make_analysis(0.95)], github=gh, dry_run=True
    )
    assert results[-1]["final_action"] == "auto_approved_dry_run"
    assert gh.posts == []

    events = await replay_events("t-dry")
    terminal = events[-1]
    assert terminal["decision"] == "dry_run"
    assert terminal["decision"] not in {"committed", "auto"}
    assert "would be posted" in terminal["reason"]
    assert not any(e["decision"] in {"committed", "auto"} for e in events)


# ─── replay ────────────────────────────────────────────────────────────────


async def test_replay_orders_events_by_insertion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    await _run(
        tmp_path, "t-order", [make_analysis(0.65)],
        resume_values=[{"choice": "approve", "feedback": ""}],
    )
    events = await replay_events("t-order")
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)


async def test_list_threads_orders_risk_by_severity_not_alphabetically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`MAX(risk_level)` over TEXT would say "med" > "low" > "high" — wrong."""
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    for thread, conf, risk in (
        ("t-high", 0.1, "high"),
        ("t-med", 0.6, "med"),
        ("t-low", 0.95, "low"),
    ):
        await write_audit_event(
            thread_id=thread, pr_url=PR_URL,
            entry=AuditEntry(
                agent_id="a", action="analyze", confidence=conf, risk_level=risk,
                decision="pending", status="done", reason="r", execution_time_ms=1,
            ),
        )
        await write_audit_event(
            thread_id=thread, pr_url=PR_URL,
            entry=AuditEntry(
                agent_id="a", action="analyze", confidence=conf, risk_level="low",
                decision="auto", status="done", reason="second, lower risk", execution_time_ms=1,
            ),
        )

    rows = {r["thread_id"]: r for r in await list_threads()}
    assert rows["t-high"]["worst_risk"] == "high"
    assert rows["t-med"]["worst_risk"] == "med"
    assert rows["t-low"]["worst_risk"] == "low"
    assert rows["t-high"]["events"] == 2
    assert rows["t-high"]["last_action"] == "analyze"
    assert rows["t-high"]["last_decision"] == "auto"


async def test_list_threads_is_empty_without_events(audit_db) -> None:
    assert await list_threads() == []


# ─── the exercise-4 CLI's own resume/recovery helpers ──────────────────────


async def test_exercise_4_pending_interrupt_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--thread` on a paused thread must replay the *same* questions.

    This is the exercise-4 CLI's resume path: read the pending interrupt from the
    checkpoint (not from `snapshot.interrupts`, which is empty) and hand it back
    to the reviewer without re-running the LLM.
    """
    from exercises.exercise_4_audit import _first_interrupt_payload, _pending_interrupts

    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    db = tmp_path / "hitl.db"
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    cp = await cm.__aenter__()
    await cp.setup()
    try:
        factory = FakeLLMFactory(make_analysis(0.3, questions=["app/auth.py:12 — why MD5?"]))
        app = build_graph(
            mode=Mode.EXERCISE_4, checkpointer=cp, llm_factory=factory, github=FakeGitHub()
        )
        await app.ainvoke({"pr_url": PR_URL, "thread_id": "t-resume"}, config_for("t-resume"))

        snapshot = await app.aget_state(config_for("t-resume"))
        pending = _pending_interrupts(snapshot)
        assert len(pending) == 1, "a paused thread must expose its interrupt"
        payload = _first_interrupt_payload({"__interrupt__": pending})
        assert payload["kind"] == "escalation"
        assert payload["questions"] == ["app/auth.py:12 — why MD5?"]

        # Resuming from that recovered payload completes the run and does not
        # re-run analyze (the fake has no analyses left).
        final = await app.ainvoke(
            Command(resume={payload["questions"][0]: "legacy"}), config_for("t-resume")
        )
        assert final["escalation_answers"] == {"app/auth.py:12 — why MD5?": "legacy"}
    finally:
        await cm.__aexit__(None, None, None)


async def test_exercise_4_detects_a_finished_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finished thread has no pending interrupts, so the CLI starts a new one."""
    from exercises.exercise_4_audit import _pending_interrupts

    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "hitl.db"))
    db = tmp_path / "hitl.db"
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    cp = await cm.__aenter__()
    await cp.setup()
    try:
        app = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.95)),
            github=FakeGitHub(),
        )
        result = await app.ainvoke({"pr_url": PR_URL, "thread_id": "t-done"}, config_for("t-done"))
        assert result["final_action"] == "auto_approved"

        snapshot = await app.aget_state(config_for("t-done"))
        assert _pending_interrupts(snapshot) == []
        assert snapshot.values  # the CLI uses this to decide "already finished"
    finally:
        await cm.__aexit__(None, None, None)
