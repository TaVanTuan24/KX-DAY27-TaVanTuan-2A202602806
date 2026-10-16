"""FIX 2 — dry-run is an immutable property of a review session.

The dangerous shape this guards against: a session starts in dry-run, the operator
flips the toggle, the graph is rebuilt with ``dry_run=False`` on the resume, and a
rehearsal silently becomes a real comment on a public PR.

The rules under test:

* ``state["dry_run"]`` wins over anything the caller passes to ``build_graph``;
* seeding happens once, when the session starts;
* a checkpoint that records no mode fails safe (no POST);
* a dry-run never calls the GitHub POST at all and is audited as ``dry_run``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import replay_events
from exercises.agent import Agent, Mode, build_graph, config_for
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis

THREAD = "dry-thread"
OPEN = {"pr_url": PR_URL, "thread_id": THREAD}


# ─── the precedence rule itself ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("state", "agent_default", "expected"),
    [
        ({"dry_run": True}, False, True),    # state wins over config
        ({"dry_run": False}, True, False),   # state wins over config
        ({"dry_run": True}, None, True),
        ({"dry_run": False}, None, False),
        ({}, False, False),                  # no state value → explicit config
        ({}, True, True),
        ({}, None, True),                    # unknown provenance → FAIL SAFE
    ],
)
def test_effective_dry_run_precedence(state, agent_default, expected) -> None:
    agent = Agent(mode=Mode.EXERCISE_2, dry_run=agent_default)
    assert agent.effective_dry_run(state) is expected


def test_unknown_provenance_is_reported_as_fail_safe() -> None:
    agent = Agent(mode=Mode.EXERCISE_2, dry_run=None)
    assert "fail-safe" in agent.dry_run_origin({})
    assert "session" in agent.dry_run_origin({"dry_run": False})


# ─── end-to-end: start dry-run, resume asking for a real post ──────────────


async def _open(db: Path) -> tuple:
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    cp = await cm.__aenter__()
    await cp.setup()
    return cm, cp


async def test_dry_run_session_resumed_with_live_config_never_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 2: the resume is rebuilt with dry_run=False, and must still not POST."""
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "dry.db"))
    db = tmp_path / "dry.db"
    gh = FakeGitHub()

    cm, cp = await _open(db)
    try:
        app = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.65)),
            github=gh,
            dry_run=None,  # state is authoritative
        )
        await app.ainvoke({**OPEN, "dry_run": True}, config_for(THREAD))

        # A *new* graph object, built as if the operator had turned the toggle off.
        app_live = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(),
            github=gh,
            dry_run=False,
        )
        final = await app_live.ainvoke(
            Command(resume={"choice": "approve", "feedback": ""}), config_for(THREAD)
        )
    finally:
        await cm.__aexit__(None, None, None)

    assert gh.posts == [], "a dry-run session must never POST, even on resume"
    assert final["final_action"] == "committed_dry_run"
    assert final["post_status"] == "dry_run"

    events = await replay_events(THREAD)
    assert events[-1]["decision"] == "dry_run"
    assert not any(e["decision"] in {"committed", "auto"} for e in events)


async def test_live_session_resumed_with_dry_run_config_is_not_downgraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stored mode is not overridden in either direction."""
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "live.db"))
    db = tmp_path / "live.db"
    gh = FakeGitHub()

    cm, cp = await _open(db)
    try:
        app = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.65)),
            github=gh,
            dry_run=None,
        )
        await app.ainvoke({**OPEN, "dry_run": False}, config_for(THREAD))

        # Rebuilt as if the operator had turned dry-run ON mid-session.
        app_dry = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(),
            github=gh,
            dry_run=True,
        )
        final = await app_dry.ainvoke(
            Command(resume={"choice": "approve", "feedback": ""}), config_for(THREAD)
        )
    finally:
        await cm.__aexit__(None, None, None)

    assert final["final_action"] == "committed"
    assert final["post_status"] == "posted"
    assert len(gh.posts) == 1


async def test_legacy_checkpoint_without_a_mode_fails_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 2.7: an older checkpoint records no dry-run → refuse to POST."""
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "legacy.db"))
    db = tmp_path / "legacy.db"
    gh = FakeGitHub()

    cm, cp = await _open(db)
    try:
        # Seed a paused session *without* persisting dry_run (simulating the old
        # schema), by starting it and then rewriting the checkpoint state.
        app = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.65)),
            github=gh,
            dry_run=None,
        )
        await app.ainvoke(OPEN, config_for(THREAD))  # no "dry_run" key at all

        resumed = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(),
            github=gh,
            dry_run=None,  # the UI's "trust the state" mode
        )
        final = await resumed.ainvoke(
            Command(resume={"choice": "approve", "feedback": ""}), config_for(THREAD)
        )
    finally:
        await cm.__aexit__(None, None, None)

    assert gh.posts == [], "unknown provenance must fail safe"
    assert final["final_action"] == "committed_dry_run"
    assert final["post_status"] == "dry_run"


async def test_dry_run_never_calls_the_github_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "auto.db"))
    db = tmp_path / "auto.db"
    gh = FakeGitHub()

    cm, cp = await _open(db)
    try:
        app = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.95)),
            github=gh,
            dry_run=None,
        )
        final = await app.ainvoke({**OPEN, "dry_run": True}, config_for(THREAD))
    finally:
        await cm.__aexit__(None, None, None)

    assert gh.posts == []
    assert final["final_action"] == "auto_approved_dry_run"
    events = await replay_events(THREAD)
    assert events[-1]["decision"] == "dry_run"
    assert "dry-run" in events[-1]["reason"]


async def test_the_audit_names_where_the_mode_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "origin.db"))
    db = tmp_path / "origin.db"
    cm, cp = await _open(db)
    try:
        app = build_graph(
            mode=Mode.EXERCISE_4,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.95)),
            github=FakeGitHub(),
            dry_run=None,
        )
        await app.ainvoke({**OPEN, "dry_run": True}, config_for(THREAD))
    finally:
        await cm.__aexit__(None, None, None)

    reason = (await replay_events(THREAD))[-1]["reason"]
    assert "session" in reason  # not "config", not "fail-safe"
