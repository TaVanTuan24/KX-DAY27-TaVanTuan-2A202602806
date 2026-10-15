"""Exercise 1 — Confidence scoring + routing.

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of
three terminal nodes by confidence.

    START → fetch_pr → analyze → route ─┬─ auto_approve  → END
                                        ├─ human_approval → END
                                        └─ escalate       → END

Thresholds come from ``common/schemas.py`` (``AUTO_APPROVE_THRESHOLD = 0.73``,
``ESCALATE_THRESHOLD = 0.58``); routing is delegated to
``common.schemas.route_for`` so this exercise and exercises 2–5 can never drift
apart. Nothing here is keyed off the PR URL — the branch is decided by the
model's own (honest, calibrated) confidence plus a security override.

Run:
    uv run python exercises/exercise_1_confidence.py --pr <PR URL> [--dry-run]
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from rich.console import Console

from common.github import GitHubError, fetch_pr
from common.llm import structured_llm
from common.prompts import build_analyze_messages, force_utf8_stdio
from common.schemas import (
    PRAnalysis,
    ReviewState,
    audit_risk_level,
    has_blocking_risk,
    route_for,
)

force_utf8_stdio()
console = Console()


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_description": pr.description,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    """Call the LLM with structured output and return a valid ``PRAnalysis``."""
    console.print("[cyan]→ analyze[/cyan]")
    llm = structured_llm(PRAnalysis)  # get_llm().with_structured_output(PRAnalysis)
    messages = build_analyze_messages(
        title=state.get("pr_title", ""),
        diff=state.get("pr_diff", ""),
        description=state.get("pr_description", ""),
        files=state.get("pr_files") or [],
    )
    try:
        with console.status("[dim]LLM thinking...[/dim]"):
            analysis = llm.invoke(messages)
    except Exception as exc:
        # Provider problems (missing key, no credits, rate limit, timeout) must
        # surface as one readable line — not a traceback through LangGraph.
        raise RuntimeError(
            f"LLM analysis failed ({type(exc).__name__}): {exc}"
        ) from exc

    if not isinstance(analysis, PRAnalysis):
        raise TypeError(f"LLM returned {type(analysis).__name__}, expected PRAnalysis")

    console.print(
        f"  [green]✓[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.comments)} comment(s), "
        f"{len(analysis.risk_factors)} risk factor(s)"
    )
    console.print(f"  [dim]{analysis.summary}[/dim]")
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    """Map the analysis to one of the three decisions."""
    console.print("[cyan]→ route[/cyan]")
    analysis = state["analysis"]
    decision = route_for(analysis)
    override = ""
    if decision == "human_approval" and has_blocking_risk(analysis):
        override = "  [red](security/blocker override)[/red]"
    console.print(
        f"  [green]✓[/green] decision=[bold]{decision}[/bold] "
        f"(confidence={analysis.confidence:.0%}){override}"
    )
    return {"decision": decision}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
    return {"final_action": "auto_approved"}


def node_human_approval(state: ReviewState) -> dict:
    console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — exercise 2 pauses here")
    return {"final_action": "pending_human_approval"}


def node_escalate(state: ReviewState) -> dict:
    console.print("[red]✓ ESCALATE[/red] — exercise 3 asks the reviewer questions")
    if state["analysis"].escalation_questions:
        for i, question in enumerate(state["analysis"].escalation_questions, start=1):
            console.print(f"  [yellow]Q{i}[/yellow] {question}")
    return {"final_action": "pending_escalation"}


def build_graph():
    """START → fetch_pr → analyze → route → {auto_approve|human_approval|escalate} → END."""
    g = StateGraph(ReviewState)
    g.add_node("fetch_pr", node_fetch_pr)
    g.add_node("analyze", node_analyze)
    g.add_node("route", node_route)
    g.add_node("auto_approve", node_auto_approve)
    g.add_node("human_approval", node_human_approval)
    g.add_node("escalate", node_escalate)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "human_approval",
            "escalate": "escalate",
        },
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", END)
    g.add_edge("escalate", END)
    return g.compile()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True, help="GitHub PR URL")
    parser.add_argument(
        "--dry-run", action="store_true", help="accepted for symmetry (this exercise posts nothing)"
    )
    args = parser.parse_args()

    console.rule("[bold]Exercise 1 — confidence routing[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    try:
        final = app.invoke({"pr_url": args.pr})
    except GitHubError as exc:
        console.print(f"[red]✗ GitHub error:[/red] {exc.friendly}")
        raise SystemExit(1) from exc
    except (RuntimeError, TypeError) as exc:
        console.print(f"[red]✗ {exc}[/red]")
        console.print(
            "[dim]Nothing was posted. Check OPENROUTER_API_KEY / LLM_MODEL in .env, "
            "and your OpenRouter credit balance.[/dim]"
        )
        raise SystemExit(1) from exc
    except ValueError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        raise SystemExit(2) from exc

    console.rule("Final")
    analysis = final.get("analysis")
    if analysis is None:
        console.print(f"[red]✗ run stopped early:[/red] {final.get('final_action')}")
        console.print("[red]Nothing was posted to GitHub.[/red]")
        raise SystemExit(1)
    console.print(f"confidence = {analysis.confidence:.0%} (risk={audit_risk_level(analysis)})")
    console.print(f"decision   = {final.get('decision')}")
    console.print(f"action     = {final.get('final_action')}")


if __name__ == "__main__":
    main()
