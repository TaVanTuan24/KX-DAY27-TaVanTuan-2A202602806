"""Terminal helpers shared by the exercise CLIs.

Exercises 1–4 all need the same things: render an ``interrupt()`` payload as a
Rich panel, collect the reviewer's answer, and map the finished run onto a
consistent exit code. Keeping that here removes the copy-paste the skeletons had,
and gives one place to handle the ``approval_request`` / ``escalation`` /
``final_confirmation`` / ``edited_review`` payload kinds.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from common.prompts import force_utf8_stdio

force_utf8_stdio()
console = Console()

VALID_CHOICES = {"approve", "reject", "edit"}
VALID_EDIT_ACTIONS = {"confirm", "edit", "reject"}

#: Exit codes shared by every exercise CLI, so scripts and CI can rely on them.
#:
#: 0   the workflow finished as designed (posted, already posted, dry-run, or a
#:     deliberate human reject)
#: 1   a runtime failure (GitHub / LLM / posting), an unconfirmed POST outcome, or
#:     a mandatory audit write that failed
#: 2   invalid input (not a PR URL)
#: 130 the operator interrupted with Ctrl+C
EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_BAD_INPUT = 2
EXIT_INTERRUPTED = 130

#: Terminal actions that mean the workflow did what it set out to do.
SUCCESS_ACTIONS = frozenset(
    {
        "committed",
        "committed_already",
        "committed_dry_run",
        "auto_approved",
        "auto_approved_dry_run",
        "rejected",                # the reviewer said no — not a runtime failure
        "pending_human_approval",  # exercise 1's designed terminal
        "pending_escalation",      # exercise 1's designed terminal
    }
)

#: Terminal actions that mean it did not. An unconfirmed POST and a failed audit
#: write are BOTH failures for automation even though nothing crashed.
FAILURE_ACTIONS = frozenset(
    {
        "fetch_failed",
        "analyze_failed",
        "commit_failed",
        "commit_unconfirmed",
        "auto_approve_failed",
        "auto_refused_security",
    }
)


def workflow_exit_code(result: dict | None) -> int:
    """Map a finished run's state onto the shared exit code.

    Deliberately conservative:

    * a *deliberate* ``rejected`` is success — the human decided, nothing broke;
    * a dry-run is success — it is a designed outcome and is reported as such;
    * an ``unconfirmed`` POST or a missing audit row is **failure**: automation
      must not read "we don't know" as "posted".
    """
    result = result or {}
    action = result.get("final_action") or ""
    if action in FAILURE_ACTIONS:
        return EXIT_FAILURE
    if action not in SUCCESS_ACTIONS:
        # Unknown or absent terminal action: never claim success by default.
        return EXIT_FAILURE
    if result.get("post_status") == "unconfirmed":
        return EXIT_FAILURE
    if result.get("audit_error"):
        return EXIT_FAILURE
    return EXIT_OK


def workflow_exit_message(result: dict | None) -> str | None:
    """One line explaining a non-zero exit, or None when the run was fine."""
    result = result or {}
    action = result.get("final_action") or "(no final_action)"
    if result.get("audit_error"):
        return (
            f"the audit trail is incomplete ({result['audit_error']}) — refusing to "
            "report success without evidence of the decision"
        )
    if action in FAILURE_ACTIONS:
        detail = (
            result.get("post_error")
            or result.get("fetch_error")
            or result.get("analysis_error")
            or ""
        )
        return f"{action}: {detail}".rstrip(": ") if detail else action
    if action not in SUCCESS_ACTIONS:
        return f"unexpected terminal state: {action}"
    return None


def report_exit(result: dict | None) -> int:
    """Print the exit reason (when there is one) and return the exit code."""
    message = workflow_exit_message(result)
    if message:
        console.print(f"[red]✗ {message}[/red]")
    return workflow_exit_code(result)


def ask(prompt: str, **kwargs) -> str:
    """Ask the reviewer one question.

    A single, deliberately narrow seam: every prompt in this module goes through
    it, so the whole interactive surface can be driven from a test (or replaced
    by a different UI) without patching Rich's global console.
    """
    return Prompt.ask(prompt, **kwargs)


def render_approval_request(payload: dict) -> dict:
    """Render the 58–72% approval card and collect the reviewer's choice.

    Returns the dict passed to ``Command(resume=...)``:
    ``{"choice": "approve"|"reject"|"edit", "feedback": str}``.

    For ``edit`` the reviewer may either describe the change (``feedback``, applied
    by the LLM in ``edit_review``) or paste the replacement review
    (``edited_body``, posted verbatim). Both keep the graph in the same thread; a
    later prompt confirms the result before anything is posted.
    """
    body = [
        f"[bold]Confidence:[/bold] {payload['confidence']:.0%}"
        f"   [dim](risk factors: {len(payload.get('risk_factors') or [])})[/dim]",
        f"[dim]{payload.get('confidence_reasoning', '')}[/dim]",
        "",
        f"[bold]Summary:[/bold] {payload.get('summary', '')}",
    ]
    comments = payload.get("comments") or []
    if comments:
        body.append("")
        body.append("[bold]Proposed comments:[/bold]")
        for c in comments:
            location = f"{c['file']}:{c.get('line') or '?'}"
            body.append(f"  [{c['severity']}] {location} — {c['body']}")
    if payload.get("risk_factors"):
        body += ["", "[bold]Risk factors:[/bold]"]
        body += [f"  • {rf}" for rf in payload["risk_factors"]]

    console.print(Panel("\n".join(body), title="Approval request", border_style="green"))

    if payload.get("diff_preview"):
        console.print("\n[dim]--- diff preview (truncated) ---[/dim]")
        console.print(payload["diff_preview"], markup=False)
        console.print("[dim]--- end diff preview ---[/dim]\n")

    choice = ""
    while choice not in VALID_CHOICES:
        choice = ask(
            "[bold]Choice[/bold]", choices=sorted(VALID_CHOICES), default="reject"
        ).strip().lower()

    answer: dict = {"choice": choice, "feedback": ""}
    if choice == "edit":
        console.print(
            "[dim]Edit: describe what to change (the LLM revises the review), or paste "
            "a full replacement review — it is posted verbatim. Nothing is posted "
            "until you confirm the result.[/dim]"
        )
        answer["feedback"] = ask("[bold]Your edit instructions[/bold]", default="")
        replacement = ask(
            "[bold]Replacement review text (Enter to let the LLM revise it)[/bold]",
            default="",
        )
        if replacement.strip():
            answer["edited_body"] = replacement
    elif choice == "reject":
        answer["feedback"] = ask("[bold]Reason for rejecting (optional)[/bold]", default="")
    return answer


def render_edited_review(payload: dict) -> dict:
    """Post-Edit gate: show the revised review and confirm / edit again / reject.

    Returns ``{"confirm": bool, "feedback": str}`` or
    ``{"choice": "edit", "feedback": str, "edited_body": str|None}`` to loop back
    into another revision.
    """
    revision = payload.get("revision") or 1
    body = [
        f"[bold]Revised confidence:[/bold] {payload['confidence']:.0%}"
        f"   [dim](revision {revision})[/dim]",
        f"[dim]{payload.get('confidence_reasoning', '')}[/dim]",
        "",
        f"[bold]Summary:[/bold] {payload.get('summary', '')}",
    ]
    comments = payload.get("comments") or []
    if comments:
        body.append("")
        body.append("[bold]Findings:[/bold]")
        for c in comments:
            body.append(
                f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}"
            )
    if payload.get("risk_factors"):
        body += ["", "[bold]Risk factors:[/bold]"]
        body += [f"  • {rf}" for rf in payload["risk_factors"]]
    console.print(
        Panel("\n".join(body), title=f"Edited review (revision {revision})", border_style="magenta")
    )

    if payload.get("proposed_body"):
        console.print("[dim]--- comment that would be posted ---[/dim]")
        console.print(payload["proposed_body"], markup=False)
        console.print("[dim]--- end comment ---[/dim]")

    action = ""
    while action not in VALID_EDIT_ACTIONS:
        action = ask(
            "[bold]Post this revised review?[/bold]",
            choices=sorted(VALID_EDIT_ACTIONS),
            default="reject",
        ).strip().lower()

    if action == "edit":
        instructions = ask("[bold]Further edit instructions[/bold]", default="")
        replacement = ask(
            "[bold]Replacement review text (Enter to let the LLM revise it)[/bold]",
            default="",
        )
        return {
            "choice": "edit",
            "feedback": instructions,
            "edited_body": replacement.strip() or None,
        }
    feedback = ""
    if action == "reject":
        feedback = ask("[bold]Reason (optional)[/bold]", default="")
    return {"confirm": action == "confirm", "feedback": feedback}


def render_escalation(payload: dict) -> dict[str, str]:
    """Render the <58% escalation card and collect one answer per question."""
    body = [
        f"[bold]Confidence:[/bold] {payload['confidence']:.0%}  [red]strong escalation[/red]",
        f"[dim]{payload.get('confidence_reasoning', '')}[/dim]",
        "",
        f"[bold]Summary:[/bold] {payload.get('summary', '')}",
    ]
    if payload.get("risk_factors"):
        body += ["", "[bold red]Risk factors:[/bold red]"]
        body += [f"  • {rf}" for rf in payload["risk_factors"]]
    console.print(Panel("\n".join(body), title="Escalation", border_style="yellow"))

    if payload.get("diff_preview"):
        console.print("[dim]--- diff preview (truncated) ---[/dim]")
        console.print(payload["diff_preview"], markup=False)
        console.print("[dim]--- end diff preview ---[/dim]\n")

    answers: dict[str, str] = {}
    for i, question in enumerate(payload.get("questions") or [], start=1):
        console.print(f"[bold yellow]Q{i}.[/bold yellow] {question}")
        answers[question] = ask("[dim]Answer[/dim]", default="").strip()
    return answers


def render_final_confirmation(payload: dict) -> dict:
    """Refined-review gate after escalation (exercises 4–5)."""
    body = [
        f"[bold]Refined confidence:[/bold] {payload['confidence']:.0%}",
        f"[dim]{payload.get('confidence_reasoning', '')}[/dim]",
        "",
        f"[bold]Summary:[/bold] {payload.get('summary', '')}",
    ]
    comments = payload.get("comments") or []
    if comments:
        body.append("")
        body.append("[bold]Comments to post:[/bold]")
        for c in comments:
            body.append(
                f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}"
            )
    if payload.get("escalation_answers"):
        body += ["", "[bold]Your answers:[/bold]"]
        body += [f"  • {q} → {a}" for q, a in payload["escalation_answers"].items()]
    console.print(Panel("\n".join(body), title="Post this review?", border_style="blue"))

    confirm = ask(
        "[bold]Post this comment to the PR?[/bold]", choices=["yes", "no"], default="no"
    )
    feedback = ""
    if confirm == "no":
        feedback = ask("[bold]Reason (optional)[/bold]", default="")
    return {"confirm": confirm == "yes", "feedback": feedback}


def handle_interrupt(payload: dict) -> dict:
    """Dispatch on ``payload["kind"]`` — the single entry point for the CLIs."""
    kind = payload.get("kind")
    if kind == "approval_request":
        return render_approval_request(payload)
    if kind == "escalation":
        return render_escalation(payload)
    if kind == "final_confirmation":
        return render_final_confirmation(payload)
    if kind == "edited_review":
        return render_edited_review(payload)
    raise ValueError(f"Unknown interrupt payload kind: {kind!r}")


__all__ = [
    "EXIT_BAD_INPUT",
    "EXIT_FAILURE",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "FAILURE_ACTIONS",
    "SUCCESS_ACTIONS",
    "VALID_CHOICES",
    "VALID_EDIT_ACTIONS",
    "console",
    "handle_interrupt",
    "render_approval_request",
    "render_edited_review",
    "render_escalation",
    "render_final_confirmation",
    "report_exit",
    "workflow_exit_code",
    "workflow_exit_message",
]
