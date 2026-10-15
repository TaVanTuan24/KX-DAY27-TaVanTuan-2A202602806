"""Exercise 2 — HITL with ``interrupt()`` + ``Command(resume=...)``.

Turn the ``human_approval`` node from a placeholder into a real pause-and-ask:

* ``node_human_approval`` calls ``interrupt(payload)`` with
  ``kind="approval_request"``, the confidence, the reasoning, the summary, the
  proposed comments and a diff preview.
* The graph is compiled with an in-process ``MemorySaver``
  (``common.db.memory_checkpointer()``) — mandatory: without a checkpointer
  ``interrupt()`` raises instead of pausing.
* ``main()`` loops while ``"__interrupt__" in result``, renders the payload and
  resumes with ``Command(resume=answer)`` **on the same thread_id**.

``approve`` → the comment is posted. ``reject`` → nothing is posted. ``edit`` →
the reviewer's note is attached to the posted review; edit is never silently
treated as a reject.

The node bodies live in ``exercises/agent.py`` so exercises 1–5 share one
implementation; this file is the exercise-2 view of that graph. The nodes are
coroutines, so the graph is driven with the async API (``ainvoke``) — LangGraph
does not provide a synchronous entry point for async nodes.

Run:
    uv run python exercises/exercise_2_hitl.py --pr <PR URL> [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from dotenv import load_dotenv
from common.db import memory_checkpointer
from langgraph.types import Command

from common.github import GitHubError, is_valid_pr_url
from exercises.agent import Mode, build_graph, config_for, console
from exercises.cli import handle_interrupt


async def run(pr_url: str, thread_id: str, *, dry_run: bool = False) -> dict:
    app = build_graph(
        mode=Mode.EXERCISE_2, checkpointer=memory_checkpointer(), dry_run=dry_run
    )
    cfg = config_for(thread_id)

    result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)
    return result


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True, help="GitHub PR URL")
    parser.add_argument(
        "--thread", help="Reuse an existing thread_id (resume a paused session)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Never POST to GitHub — prints what would be posted instead",
    )
    args = parser.parse_args()

    if not is_valid_pr_url(args.pr):
        console.print(f"[red]✗ Not a GitHub PR URL:[/red] {args.pr}")
        raise SystemExit(2)

    console.rule("[bold]Exercise 2 — HITL with interrupt()[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    thread_id = args.thread or str(uuid.uuid4())
    console.print(f"[dim]thread_id = {thread_id}[/dim]")
    if args.thread:
        console.print(
            "[yellow]note[/yellow] exercise 2 uses MemorySaver: a checkpointer that lives "
            "in this process only. In-process resumes work; a thread created by a "
            "*previous* process cannot be resumed here — use exercise 4 "
            "(AsyncSqliteSaver) for that."
        )
    if args.dry_run:
        console.print("[yellow]dry-run: no comment will be posted to GitHub[/yellow]")
    console.print()

    try:
        result = asyncio.run(run(args.pr, thread_id, dry_run=args.dry_run))
    except GitHubError as exc:
        console.print(f"[red]✗ GitHub error:[/red] {exc.friendly}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        console.print(
            f"\n[yellow]Interrupted.[/yellow] Resume with:\n"
            f"  uv run python exercises/exercise_2_hitl.py --pr {args.pr} --thread {thread_id}"
        )
        raise SystemExit(130)

    console.rule("Done")
    console.print(f"thread_id    = {thread_id}")
    console.print(f"final_action = {result.get('final_action')}")
    if result.get("fetch_error") or result.get("analysis_error"):
        # The graph caught a provider/API failure and stopped before posting.
        console.print(
            f"[red]error[/red]        = {result.get('fetch_error') or result.get('analysis_error')}"
        )
        console.print("[red]Nothing was posted to GitHub.[/red]")
        raise SystemExit(1)
    if result.get("post_error"):
        console.print(f"[red]error[/red]        = {result['post_error']}")
        console.print("[red]GitHub rejected the comment — the run is NOT reported as posted.[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
