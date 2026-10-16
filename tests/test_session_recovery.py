"""FIX 6 — the sidebar can reopen a *finished* session.

The bug: "Open this session" only restored a pending interrupt. For a completed
(or rejected, or dry-run) session ``load_pending`` returned ``None`` and the UI
showed nothing, so the audit trail was the only way back to the result.

The view is rebuilt from the LangGraph checkpoint and is strictly read-only:
opening a session must not re-run the LLM, must not POST, and must not modify the
checkpoint.

Note on the async style: ``app`` calls the graph/database through
``asyncio.run`` (correct for Streamlit's thread, which has no loop), so tests that
invoke those helpers are *synchronous*; tests that only read a checkpoint await it
directly.

The Streamlit stub is shared with ``test_streamlit_app.py`` rather than copied.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import replay_events, write_audit_event
from common.schemas import AuditEntry
from exercises.agent import Mode, build_graph, config_for
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis
from tests.test_streamlit_app import StubStreamlit  # shared, not duplicated


async def _finish(
    db: Path,
    thread: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    confidence: float = 0.65,
    dry_run: bool = False,
):
    """Run a session to completion on a real SQLite checkpointer (then close it).

    ``HITL_DB_PATH`` is set FIRST: the audit rows must land in the same file as
    the checkpoints, otherwise a test reads one DB and writes to another.
    """
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    gh = FakeGitHub()
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    cp = await cm.__aenter__()
    await cp.setup()
    try:
        app = build_graph(
            mode=Mode.EXERCISE_5,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(confidence)),
            github=gh,
            dry_run=None,
        )
        cfg = config_for(thread)
        result = await app.ainvoke(
            {"pr_url": PR_URL, "thread_id": thread, "dry_run": dry_run}, cfg
        )
        if "__interrupt__" in result:
            result = await app.ainvoke(
                Command(resume={"choice": "approve", "feedback": ""}), cfg
            )
        return gh, result
    finally:
        await cm.__aexit__(None, None, None)


async def _pause(db: Path, thread: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    cp = await cm.__aenter__()
    await cp.setup()
    try:
        app = build_graph(
            mode=Mode.EXERCISE_5,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.65)),
            github=FakeGitHub(),
            dry_run=None,
        )
        await app.ainvoke(
            {"pr_url": PR_URL, "thread_id": thread, "dry_run": False}, config_for(thread)
        )
    finally:
        await cm.__aexit__(None, None, None)


def _open_module(db: Path, monkeypatch: pytest.MonkeyPatch):
    """Import ``app`` with a stubbed streamlit, pointing the DB at ``db``."""
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    stub = StubStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    sys.modules.pop("app", None)
    return importlib.import_module("app"), stub


# ─── reading a session back ────────────────────────────────────────────────


async def test_finished_session_is_readable_from_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "done.db"
    await _finish(db, "t-done", monkeypatch)
    module, _ = _open_module(db, monkeypatch)

    view = await module.load_session("t-done")

    assert view.has_checkpoint
    assert view.is_finished and not view.is_pending
    assert view.values["final_action"] == "committed"
    assert view.values["analysis"].summary  # confidence/summary available to render
    assert view.values["decision"] == "human_approval"
    assert view.dry_run is False


async def test_pending_session_is_still_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "pending.db"
    await _pause(db, "t-pending", monkeypatch)
    module, _ = _open_module(db, monkeypatch)

    view = await module.load_session("t-pending")

    assert view.is_pending and not view.is_finished
    assert view.pending["kind"] == "approval_request"
    assert view.values["pr_url"] == PR_URL  # state so far is available too


async def test_rejected_session_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "rejected.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    # Finish it as a reject.
    cm = AsyncSqliteSaver.from_conn_string(str(db))
    cp = await cm.__aenter__()
    await cp.setup()
    try:
        app = build_graph(
            mode=Mode.EXERCISE_5,
            checkpointer=cp,
            llm_factory=FakeLLMFactory(make_analysis(0.65)),
            github=FakeGitHub(),
            dry_run=None,
        )
        cfg = config_for("t-rej")
        await app.ainvoke(
            {"pr_url": PR_URL, "thread_id": "t-rej", "dry_run": False}, cfg
        )
        await app.ainvoke(
            Command(resume={"choice": "reject", "feedback": "needs a migration"}), cfg
        )
    finally:
        await cm.__aexit__(None, None, None)

    module, _ = _open_module(db, monkeypatch)
    view = await module.load_session("t-rej")

    assert view.is_finished
    assert view.values["final_action"] == "rejected"
    assert view.values["human_choice"] == "reject"


async def test_dry_run_session_is_restored_in_dry_run_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "dry.db"
    gh, _ = await _finish(db, "t-dry", monkeypatch, confidence=0.95, dry_run=True)
    module, _ = _open_module(db, monkeypatch)

    view = await module.load_session("t-dry")

    assert gh.posts == []
    assert view.is_finished
    assert view.values["final_action"] == "auto_approved_dry_run"
    assert view.dry_run is True, "reopening must restore the original mode"


async def test_unknown_thread_has_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _ = _open_module(tmp_path / "empty.db", monkeypatch)
    view = await module.load_session("never-existed")
    assert view.has_checkpoint is False
    assert view.is_pending is False and view.is_finished is False
    assert view.dry_run is None


# ─── reopening must not do anything ────────────────────────────────────────


async def test_reopening_a_finished_session_runs_nothing_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No LLM call, no POST, and the checkpoint (hence the audit) is unchanged."""
    db = tmp_path / "noop.db"
    gh, _ = await _finish(db, "t-noop", monkeypatch)
    posts_after_run = len(gh.posts)
    events_after_run = await replay_events("t-noop")

    module, _ = _open_module(db, monkeypatch)
    first = await module.load_session("t-noop")
    second = await module.load_session("t-noop")  # "reload the page and reopen"

    assert first.values["final_action"] == "committed"
    assert second.values["final_action"] == "committed"
    assert len(gh.posts) == posts_after_run, "viewing history must never POST"
    events_now = await replay_events("t-noop")
    assert len(events_now) == len(events_after_run), "reading must not write audit rows"


def test_opening_a_finished_session_fills_the_ui_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "ui.db"
    asyncio.run(_finish(db, "t-ui", monkeypatch, dry_run=True))
    module, stub = _open_module(db, monkeypatch)
    module.init_state()

    view = asyncio.run(module.load_session("t-ui"))
    module.open_session(view, "t-ui", PR_URL)

    assert stub.session_state["thread_id"] == "t-ui"
    assert stub.session_state["final"] is not None
    assert stub.session_state["final"]["final_action"] == "committed_dry_run"
    assert stub.session_state["interrupt_payload"] is None
    assert stub.session_state["session_finished"] is True
    assert stub.session_state["session_dry_run"] is True
    assert stub.session_state["history_only"] is False


def test_opening_a_pending_session_fills_the_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "ui2.db"
    asyncio.run(_pause(db, "t-p", monkeypatch))
    module, stub = _open_module(db, monkeypatch)
    module.init_state()

    view = asyncio.run(module.load_session("t-p"))
    module.open_session(view, "t-p", PR_URL)

    assert stub.session_state["interrupt_payload"]["kind"] == "approval_request"
    assert stub.session_state["final"] is None
    assert stub.session_state["session_finished"] is False


def test_opening_a_session_with_no_checkpoint_warns_but_keeps_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit trail survives even if the checkpoint does not."""
    db = tmp_path / "auditonly.db"
    monkeypatch.setenv("HITL_DB_PATH", str(db))
    asyncio.run(
        write_audit_event(
            thread_id="t-ghost",
            pr_url=PR_URL,
            entry=AuditEntry(
                agent_id="a", action="commit", confidence=0.65, risk_level="med",
                decision="committed", status="done", reason="r", execution_time_ms=1,
            ),
        )
    )

    module, stub = _open_module(db, monkeypatch)
    module.init_state()
    view = asyncio.run(module.load_session("t-ghost"))
    assert view.has_checkpoint is False
    module.open_session(view, "t-ghost", PR_URL)

    assert stub.session_state["history_only"] is True
    assert stub.session_state["final"] is None
    assert len(asyncio.run(replay_events("t-ghost"))) == 1


def test_sidebar_lists_a_finished_session_and_its_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sidebar renders (and offers to reopen) a session that already finished."""
    db = tmp_path / "sidebar.db"
    asyncio.run(_finish(db, "t-side", monkeypatch, confidence=0.95, dry_run=True))
    module, stub = _open_module(db, monkeypatch)
    module.init_state()

    module._sidebar_recent_sessions()  # uses asyncio.run internally

    # The stub records `expander` through `form`, so assert on everything rendered.
    rendered = " ".join(str(call) for call in stub.calls)
    assert "t-side" in rendered, "session not listed"
    assert "dry-run" in rendered, "the sidebar should say the session was a dry-run"
    assert "Open this session" in rendered
