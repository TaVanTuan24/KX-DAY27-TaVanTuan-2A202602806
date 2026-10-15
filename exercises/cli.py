"""Terminal helpers shared by the exercise CLIs.

Exercises 2, 3 and 4 all need the same things: render an ``interrupt()`` payload
as a Rich panel and collect the reviewer's answer. Keeping that here removes the
copy-paste that the skeletons had, and gives one place to handle the
``approval_request`` / ``escalation`` / ``final_confirmation`` payload kinds.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from common.prompts import force_utf8_stdio

force_utf8_stdio()
console = Console()

VALID_CHOICES = {"approve", "reject", "edit"}


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

    feedback = ""
    if choice == "edit":
        console.print(
            "[dim]Edit: describe what to change. The review is NOT posted as a "
            "reject — the comment body will carry your note.[/dim]"
        )
        feedback = ask("[bold]Your edit / feedback[/bold]", default="")
    elif choice == "reject":
        feedback = ask("[bold]Reason for rejecting (optional)[/bold]", default="")
    return {"choice": choice, "feedback": feedback}


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
    raise ValueError(f"Unknown interrupt payload kind: {kind!r}")


__all__ = [
    "VALID_CHOICES",
    "console",
    "handle_interrupt",
    "render_approval_request",
    "render_escalation",
    "render_final_confirmation",
]
