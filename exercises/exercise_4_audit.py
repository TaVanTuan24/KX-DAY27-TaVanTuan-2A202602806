"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Two storage mechanisms in one ``./hitl_audit.db`` file:

* ``AsyncSqliteSaver`` — the LangGraph checkpointer. Binary blobs; lets the graph
  resume after the process dies (that is why ``--thread`` works here and not in
  exercise 2).
* ``audit_events`` — the structured decision log built from
  ``common.schemas.AuditEntry``: first-class columns an auditor can query.

Every meaningful node writes exactly one row (two for the interrupt nodes — one
*before* the human answers with ``status="waiting"``, one *after* with
``status="responded"``):

    fetch_pr, analyze, route, auto_approve, human_approval ×2,
    escalate ×2, synthesize, final_approval ×2, commit

**Duplicate-safe.** LangGraph re-executes a node from the top when the graph
resumes after ``interrupt()``, so the "before interrupt" write runs twice. The
table carries a UNIQUE index on ``(thread_id, action, status, decision)`` and
``common.db.write_audit_event`` uses ``INSERT OR IGNORE``, which makes the repeat
a no-op instead of a duplicate row (``common/db.py:_ensure_idempotency_index``
also migrates older databases and collapses pre-existing duplicates).

Run:
    uv run python exercises/exercise_4_audit.py --pr <PR URL> [--dry-run]
    uv run python exercises/exercise_4_audit.py --pr <PR URL> --thread <id>   # resume
    uv run python -m audit.replay --thread <id>
    uv run python -m audit.replay --list
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from common.db import db_path, open_checkpointer
from common.github import GitHubError, is_valid_pr_url
from exercises.agent import Mode, build_graph, config_for, console
from exercises.cli import handle_interrupt


async def run(pr_url: str, thread_id: str | None, *, dry_run: bool = False) -> None:
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]")
    console.print(f"[dim]audit db   = {db_path()}[/dim]")
    if dry_run:
        console.print("[yellow]dry-run: no comment will be posted to GitHub[/yellow]")
    console.print()

    async with open_checkpointer() as cp:
        app = build_graph(mode=Mode.EXERCISE_4, checkpointer=cp, dry_run=dry_run)

        try:
            # Does this thread already exist (and is it still paused)? A finished
            # thread has no pending nodes; an interrupted one has `next` set and
            # carries the original interrupt payload — which means we can show
            # the *same* questions again without re-running fetch/analyze/LLM.
            snapshot = await app.aget_state(config_for(thread_id))
            pending = _pending_interrupts(snapshot)
            if pending:
                console.print("[dim]resuming the checkpointed session…[/dim]")
                result = {"__interrupt__": pending}
            else:
                if snapshot.values:
                    # Re-using a finished thread id would mix two sessions'
                    # audit rows and checkpoints, so start a clean one.
                    console.print(
                        "[yellow]note[/yellow] this thread already finished — "
                        "starting a new session under a fresh thread_id"
                    )
                    thread_id = str(uuid.uuid4())
                    console.print(f"[dim]thread_id = {thread_id}[/dim]")
                result = await app.ainvoke(
                    {"pr_url": pr_url, "thread_id": thread_id},
                    config_for(thread_id),
                )

            while "__interrupt__" in result:
                payload = _first_interrupt_payload(result)
                result = await app.ainvoke(
                    Command(resume=handle_interrupt(payload)),
                    config_for(thread_id),
                )
        except GitHubError as exc:
            console.print(f"[red]✗ GitHub error:[/red] {exc.friendly}")
            raise SystemExit(1) from exc
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow] State is checkpointed.")
            console.print(
                f"  Resume: uv run python exercises/exercise_4_audit.py --pr {pr_url} "
                f"--thread {thread_id}"
            )
            raise SystemExit(130)

        # Make sure the audit rows are flushed and readable before we print the
        # replay command.
        await _report(thread_id, result)


def _first_interrupt_payload(result: dict) -> dict:
    """Normalise the interrupt payload.

    ``ainvoke`` returns ``{"__interrupt__": [Interrupt(...)]}`` while
    ``aget_state`` returns plain ``Interrupt`` objects in ``snapshot.interrupts``;
    both expose ``.value``.
    """
    first = result["__interrupt__"][0]
    return getattr(first, "value", first)


def _pending_interrupts(snapshot) -> list:
    """Interrupts a checkpointed thread is currently waiting on ([] if none).

    ``StateSnapshot.interrupts`` is empty for a paused graph in current
    LangGraph; the interrupts live on the pending tasks, so read them there.
    """
    return [
        intr for task in (snapshot.tasks or ()) for intr in (task.interrupts or ())
    ]


async def _report(thread_id: str, result: dict) -> None:
    from common.db import replay_events

    console.rule("Final")
    console.print(f"decision     = {result.get('decision')}")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")
    if result.get("post_error"):
        console.print(f"[red]post_error[/red]   = {result['post_error']}")

    events = await replay_events(thread_id)
    console.print(f"\n[bold]{len(events)} audit event(s) written[/bold]")
    for ev in events:
        console.print(
            f"  [dim]{ev['timestamp']}[/dim] [cyan]{ev['action']:<15}[/cyan] "
            f"conf={ev['confidence']:.2f} risk={ev['risk_level']:<4} "
            f"decision={ev['decision']:<10} status={ev['status']}"
        )

    if result.get("fetch_error") or result.get("analysis_error") or result.get("post_error"):
        console.print(
            "[red]The run did not post a comment[/red] — see the audit trail above "
            "(decision='failed')."
        )

    console.rule("Replay")
    console.print(f"uv run python -m audit.replay --thread {thread_id}")
    console.print("uv run python -m audit.replay --list")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True, help="GitHub PR URL")
    parser.add_argument("--thread", help="Resume an existing thread_id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Never POST to GitHub (audit trail is still written)",
    )
    args = parser.parse_args()

    if not is_valid_pr_url(args.pr):
        console.print(f"[red]✗ Not a GitHub PR URL:[/red] {args.pr}")
        raise SystemExit(2)

    asyncio.run(run(args.pr, args.thread, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
