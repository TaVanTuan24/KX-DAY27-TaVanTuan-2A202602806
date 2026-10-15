"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 0.58 the agent must not ask "approve / reject". It asks the
reviewer *specific* questions grounded in the diff, collects the answers, then
re-runs the analysis with those answers to produce a **refined** review:

    route ── escalate ──► synthesize ──► commit ──► END

* ``node_analyze``'s prompt (``common/prompts.py``) requires 2–4 concrete
  escalation questions naming the file/function at risk whenever confidence is
  below the auto-approve threshold.
* ``node_escalate`` calls ``interrupt({"kind": "escalation", ...})`` and stores
  the answers in ``escalation_answers``.
* ``node_synthesize`` re-prompts the model with the diff + original findings +
  the reviewer's answers, keeps any risk the reviewer did not actually resolve,
  and refuses to inflate confidence without a concrete reason (unresolved
  ``blocker`` findings are re-instated even if the model drops them).

Exercise 3 posts the refined review immediately after synthesis (the README's
minimum flow). Exercise 4 adds the mandatory final confirmation before posting.

Run:
    uv run python exercises/exercise_3_escalation.py --pr <PR URL> [--dry-run]
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
        mode=Mode.EXERCISE_3, checkpointer=memory_checkpointer(), dry_run=dry_run
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
    parser.add_argument("--thread", help="Reuse an existing thread_id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Never POST to GitHub — prints what would be posted instead",
    )
    args = parser.parse_args()

    if not is_valid_pr_url(args.pr):
        console.print(f"[red]✗ Not a GitHub PR URL:[/red] {args.pr}")
        raise SystemExit(2)

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    thread_id = args.thread or str(uuid.uuid4())
    console.print(f"[dim]thread_id = {thread_id}[/dim]")
    if args.dry_run:
        console.print("[yellow]dry-run: no comment will be posted to GitHub[/yellow]")
    console.print()

    try:
        result = asyncio.run(run(args.pr, thread_id, dry_run=args.dry_run))
    except GitHubError as exc:
        console.print(f"[red]✗ GitHub error:[/red] {exc.friendly}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted.[/yellow] thread_id = {thread_id}")
        raise SystemExit(130)

    console.rule("Final")
    console.print(f"decision     = {result.get('decision')}")
    console.print(f"final_action = {result.get('final_action')}")
    if result.get("fetch_error") or result.get("analysis_error"):
        console.print(
            f"[red]error[/red]        = {result.get('fetch_error') or result.get('analysis_error')}"
        )
        console.print("[red]Nothing was posted to GitHub.[/red]")
        raise SystemExit(1)
    if result.get("post_error"):
        console.print(f"[red]error[/red]        = {result['post_error']}")
        raise SystemExit(1)
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")
        console.print(
            f"unresolved risk factors = {len(result['analysis'].risk_factors)}"
        )
    if result.get("escalation_answers"):
        console.print("[dim]reviewer answers recorded:[/dim]")
        for q, a in result["escalation_answers"].items():
            console.print(f"  [dim]• {q}[/dim] → {a or '(no answer)'}")


if __name__ == "__main__":
    main()
