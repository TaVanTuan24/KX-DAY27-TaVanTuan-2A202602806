"""Replay a review session from the audit trail.

Usage:
    uv run python -m audit.replay --thread <thread_id>
    uv run python -m audit.replay --list                # list recent threads

The script reads `audit_events` (human-readable timeline). The LangGraph
SqliteSaver checkpoint tables live in the same .db file but are queried
separately by the LangGraph runtime, not by this tool.

Risk ordering is explicit: the "worst risk" of a thread is `high > med > low`,
computed with a CASE expression rather than `MAX(risk_level)` over TEXT (which
would order alphabetically: "med" > "low" > "high" — i.e. wrong).
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.table import Table

from common.db import list_threads as _list_threads_data
from common.db import replay_events
from common.prompts import force_utf8_stdio

RISK_COLOR = {"low": "green", "med": "yellow", "high": "red"}
STATUS_COLOR = {
    "pending": "white",
    "waiting": "yellow",
    "responded": "cyan",
    "done": "green",
    "failed": "red",
}

force_utf8_stdio()
console = Console()


async def list_threads() -> None:
    """Print a table of recent review sessions (CLI view)."""
    rows = await _list_threads_data()

    table = Table(title="Recent review sessions")
    for col in (
        "thread_id", "pr_url", "started", "last_event",
        "worst_risk", "events", "state",
    ):
        table.add_column(col)
    for r in rows:
        risk = str(r["worst_risk"] or "?")
        table.add_row(
            r["thread_id"],
            r["pr_url"],
            str(r["started"]),
            str(r["last_event"]),
            f"[{RISK_COLOR.get(risk, 'white')}]{risk}[/]",
            str(r["events"]),
            f"{r['last_action']}/{r['last_decision']}",
        )
    if not rows:
        console.print("[yellow]No audit events yet — run an exercise first.[/yellow]")
        return
    console.print(table)


async def replay(thread_id: str) -> None:
    events = await replay_events(thread_id)

    if not events:
        console.print(f"[red]No events found for thread {thread_id}[/red]")
        return

    console.rule(f"[bold]Replay {thread_id}")
    for ev in events:
        risk = ev["risk_level"]
        risk_colored = f"[{RISK_COLOR.get(risk, 'white')}]{risk:<4}[/]"
        status = ev.get("status") or "?"
        status_colored = f"[{STATUS_COLOR.get(status, 'white')}]{status:<9}[/]"
        reviewer = ev["reviewer_id"] or "-"
        reason = (ev["reason"] or "")[:60]
        # `rev` is only meaningful once a review has been edited (revision >= 1);
        # showing it lets an auditor tell two Edit rounds apart.
        revision = ev.get("revision") or 0
        rev_colored = f"rev=[bold]{revision}[/bold] " if revision else "        "
        console.print(
            f"[dim]{ev['timestamp']}[/dim]  "
            f"[cyan]{ev['action']:<18}[/cyan] "
            f"conf=[bold]{ev['confidence']:.2f}[/bold] "
            f"risk={risk_colored} "
            f"decision=[magenta]{ev['decision']:<10}[/magenta] "
            f"status={status_colored} "
            f"{rev_colored}"
            f"reviewer={reviewer:<14} "
            f"{ev['execution_time_ms']:>5}ms  "
            f"[dim]{reason}[/dim]"
        )
    console.print(f"\n[dim]{len(events)} event(s)[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", help="Replay one thread by id")
    parser.add_argument("--list", action="store_true", help="List recent threads")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_threads())
    elif args.thread:
        asyncio.run(replay(args.thread))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
