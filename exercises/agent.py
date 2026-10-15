"""Shared LangGraph wiring for the HITL PR-review agent (exercises 1–5).

Exercises 1–4 are *views* of one graph, so the graph — and every node — lives
here once instead of being copy-pasted four times:

===========  ==========================================================
``EXERCISE_1``  fetch → analyze → route → 3 terminal placeholders (no HITL)
``EXERCISE_2``  + ``interrupt()`` approval, ``commit`` posts on approve
``EXERCISE_3``  + escalation Q&A (``escalate`` → ``synthesize`` → ``commit``)
``EXERCISE_4``  + structured SQLite audit trail and a final confirmation
``EXERCISE_5``  the same full graph, driven by the Streamlit UI
===========  ==========================================================

Dependency injection is deliberate: ``llm_factory``, ``github``, ``audit_sink``
and ``clock`` can all be replaced, which is what lets the test suite exercise
every branch — including both ``interrupt()`` kinds — without an API key and
without posting anything to a public PR.

Safety properties enforced here (lab §9):

* PR content is passed to the model fenced as untrusted data (``common.prompts``).
* A security finding or a ``blocker`` comment forces ``human_approval`` even at
  high confidence (``common.schemas.route_for``).
* Comments are only posted by ``commit``/``auto_approve`` — never before a
  decision — and posting is idempotent per ``thread_id``.
* A failed GitHub call is reported as ``commit_failed`` / recorded as
  ``decision="failed"``; the run is never reported as posted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from rich.console import Console

from common.db import memory_checkpointer
from common.github import (
    GitHubError,
    PullRequest,
    current_user,
    fetch_pr,
    post_review_comment,
)
from common.llm import StructuredOutputLLM, get_llm, structured_llm
from common.prompts import build_analyze_messages, build_synthesize_messages, force_utf8_stdio
from common.schemas import (
    AuditEntry,
    PRAnalysis,
    ReviewState,
    audit_risk_level,
    confidence_only_route as route_for_by_confidence_only,
    has_blocking_risk,
    risk_level_for,
    route_for,
)

logger = logging.getLogger(__name__)
force_utf8_stdio()
console = Console()

AGENT_ID = "pr-review-agent@v0.1"
COMMENT_MARKER_PREFIX = "day27-hitl"

#: Longest ``reason`` string written to the audit trail (keeps one row readable).
MAX_AUDIT_REASON = 600
#: How much of the diff is shown to the reviewer in an interrupt payload.
DIFF_PREVIEW_CHARS = 2000


class Mode(str, Enum):
    """Which exercise's shape of the graph to build."""

    EXERCISE_1 = "exercise_1"
    EXERCISE_2 = "exercise_2"
    EXERCISE_3 = "exercise_3"
    EXERCISE_4 = "exercise_4"
    EXERCISE_5 = "exercise_5"

    @property
    def has_hitl(self) -> bool:
        return self is not Mode.EXERCISE_1

    @property
    def has_audit(self) -> bool:
        return self in {Mode.EXERCISE_4, Mode.EXERCISE_5}

    @property
    def has_final_confirmation(self) -> bool:
        """Post-escalation "may I post this?" gate (exercises 4–5, README §6)."""
        return self in {Mode.EXERCISE_4, Mode.EXERCISE_5}


class LLMBinding(Protocol):
    """The slice of ``StructuredOutputLLM`` the nodes use."""

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> PRAnalysis: ...


class LLMFactory(Protocol):
    def __call__(self, *, temperature: float = 0.2) -> LLMBinding: ...


class GitHubService(Protocol):
    """The slice of ``common.github`` the nodes use."""

    def fetch_pr(self, pr_url: str) -> PullRequest: ...

    def post_review_comment(self, pr_url: str, body: str, *, thread_id: str) -> str: ...

    def current_user(self) -> str | None: ...


class RealLLMFactory:
    """Default :class:`LLMFactory` — reads model config from the environment."""

    def __call__(self, *, temperature: float = 0.2) -> LLMBinding:
        return structured_llm(PRAnalysis, llm=get_llm(temperature))


class RealGitHubService:
    """Default :class:`GitHubService` — talks to the GitHub REST API."""

    def fetch_pr(self, pr_url: str) -> PullRequest:
        return fetch_pr(pr_url)

    def post_review_comment(self, pr_url: str, body: str, *, thread_id: str) -> str:
        return post_review_comment(pr_url, body, thread_id=thread_id)

    def current_user(self) -> str | None:
        return current_user()


AuditSink = Callable[[dict, AuditEntry], "Any"]


async def _default_audit_sink(state: dict, entry: AuditEntry) -> None:
    from common.db import write_audit_event

    await write_audit_event(
        thread_id=state.get("thread_id", "unknown"),
        pr_url=state.get("pr_url", ""),
        entry=entry,
    )


class _Clock:
    """Monotonic wall-clock used to fill ``execution_time_ms``."""

    def now(self) -> float:
        return time.monotonic() * 1000.0

    def elapsed_ms(self, start_ms: float) -> int:
        return max(0, int(self.now() - start_ms))


def _clip(text: str | None, limit: int = MAX_AUDIT_REASON) -> str | None:
    if text is None:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reviewer_id(state: dict, github: GitHubService) -> str | None:
    explicit = state.get("reviewer_id")
    if explicit:
        return str(explicit)
    env = os.environ.get("GITHUB_USER")
    if env:
        return env
    return github.current_user()


def build_comment_body(state: ReviewState) -> str:
    """Markdown body posted back to the PR.

    Escalation answers may be long; a comment also gets unwieldy with every
    answer inline, but they are the evidence for the refined review, so they are
    included (quoted) rather than dropped.
    """
    analysis: PRAnalysis = state["analysis"]
    lines = [
        f"### Automated review (confidence {analysis.confidence:.0%})",
        "",
        analysis.summary,
        "",
    ]
    if analysis.comments:
        lines.append("**Findings**")
        for c in analysis.comments:
            location = f"`{c.file}:{c.line}`" if c.line else f"`{c.file}`"
            lines.append(f"- **[{c.severity}]** {location} — {c.body}")
        lines.append("")
    if analysis.risk_factors:
        lines.append("**Risk factors**")
        lines += [f"- {rf}" for rf in analysis.risk_factors]
        lines.append("")

    answers = state.get("escalation_answers") or {}
    if answers:
        lines.append("<details><summary>Reviewer answers to escalation questions</summary>")
        lines.append("")
        for q, a in answers.items():
            lines.append(f"- **{q}**  \n  {a or '_(no answer given)_'}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    if state.get("human_feedback"):
        lines.append(f"_Reviewer note: {state['human_feedback']}_")
    if state.get("refine_confirmed") is not None:
        lines.append("_Reviewer confirmed posting after escalation._")
    return "\n".join(lines).strip()


# ─── nodes ────────────────────────────────────────────────────────────────


class Agent:
    """Holds one graph's dependencies and exposes the node functions."""

    def __init__(
        self,
        *,
        mode: Mode,
        llm_factory: LLMFactory | None = None,
        github: GitHubService | None = None,
        audit_sink: AuditSink | None = None,
        clock: _Clock | None = None,
        dry_run: bool = False,
    ) -> None:
        self.mode = mode
        self.llm_factory: LLMFactory = llm_factory or RealLLMFactory()
        self.github: GitHubService = github or RealGitHubService()
        self.audit_sink: AuditSink = audit_sink or _default_audit_sink
        self.clock = clock or _Clock()
        self.dry_run = dry_run

    # -- audit -------------------------------------------------------------
    async def audit(
        self,
        state: dict,
        *,
        action: str,
        confidence: float,
        risk_level: str,
        decision: str,
        status: str = "pending",
        reason: str | None = None,
        execution_time_ms: int = 0,
        reviewer_id: str | None = None,
    ) -> None:
        """Emit one audit row (no-op unless the mode has audit enabled)."""
        if not self.mode.has_audit:
            return
        entry = AuditEntry(
            agent_id=AGENT_ID,
            action=action,
            confidence=confidence,
            risk_level=risk_level,
            reviewer_id=reviewer_id,
            decision=decision,
            status=status,
            reason=_clip(reason),
            execution_time_ms=execution_time_ms,
        )
        try:
            await self.audit_sink(state, entry)
        except Exception as exc:  # auditing must never break a review session
            logger.warning("audit write failed for %s: %s", action, exc)
            console.print(f"  [yellow]![/yellow] audit write failed: {exc}")

    # -- fetch_pr ----------------------------------------------------------
    async def node_fetch_pr(self, state: ReviewState) -> dict:
        console.print("[cyan]→ fetch_pr[/cyan]")
        t0 = self.clock.now()
        pr_url = state.get("pr_url")
        if not pr_url:
            message = "no pr_url in the graph state (nothing to fetch)"
            console.print(f"  [red]✗[/red] {message}")
            await self.audit(
                state,
                action="fetch_pr",
                confidence=0.0,
                risk_level="med",
                decision="failed",
                status="failed",
                reason=message,
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {"fetch_error": message, "final_action": "fetch_failed"}
        try:
            pr = await asyncio.to_thread(self.github.fetch_pr, pr_url)
        except GitHubError as exc:
            console.print(f"  [red]✗[/red] {exc.friendly}")
            await self.audit(
                state,
                action="fetch_pr",
                confidence=0.0,
                risk_level="med",
                decision="failed",
                status="failed",
                reason=exc.friendly,
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {"fetch_error": exc.friendly, "final_action": "fetch_failed"}
        except Exception as exc:  # network/TLS/parsing problems
            message = f"{type(exc).__name__}: {exc}"
            console.print(f"  [red]✗[/red] {message}")
            await self.audit(
                state,
                action="fetch_pr",
                confidence=0.0,
                risk_level="med",
                decision="failed",
                status="failed",
                reason=message,
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {"fetch_error": message, "final_action": "fetch_failed"}

        console.print(
            f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}"
        )
        note = f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}"
        if pr.draft:
            note += " (draft)"
        await self.audit(
            state,
            action="fetch_pr",
            confidence=0.0,
            risk_level="med",
            decision="pending",
            status="done",
            reason=note,
            execution_time_ms=self.clock.elapsed_ms(t0),
        )
        return {
            "pr_title": pr.title,
            "pr_author": pr.author,
            "pr_description": pr.description,
            "pr_diff": pr.diff,
            "pr_files": pr.files_changed,
            "pr_head_sha": pr.head_sha,
        }

    async def node_analyze(self, state: ReviewState) -> dict:
        console.print("[cyan]→ analyze[/cyan]")
        t0 = self.clock.now()
        messages = build_analyze_messages(
            title=state.get("pr_title", ""),
            diff=state.get("pr_diff", ""),
            description=state.get("pr_description", ""),
            files=state.get("pr_files") or [],
        )
        try:
            with console.status("[dim]LLM reviewing the diff...[/dim]"):
                analysis = await self.llm_factory(temperature=0.2).ainvoke(messages)
        except Exception as exc:
            message = f"LLM analysis failed: {type(exc).__name__}: {exc}"
            console.print(f"  [red]✗[/red] {message}")
            await self.audit(
                state,
                action="analyze",
                confidence=0.0,
                risk_level="med",
                decision="failed",
                status="failed",
                reason=message,
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {"analysis_error": message, "final_action": "analyze_failed"}

        console.print(
            f"  [green]✓[/green] confidence={analysis.confidence:.0%}, "
            f"{len(analysis.comments)} comment(s), "
            f"{len(analysis.escalation_questions)} question(s)"
        )
        await self.audit(
            state,
            action="analyze",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="pending",
            status="done",
            reason=analysis.confidence_reasoning,
            execution_time_ms=self.clock.elapsed_ms(t0),
        )
        return {"analysis": analysis}

    async def node_route(self, state: ReviewState) -> dict:
        console.print("[cyan]→ route[/cyan]")
        t0 = self.clock.now()
        analysis: PRAnalysis = state["analysis"]
        decision = route_for(analysis)
        forced = decision != route_for_by_confidence_only(analysis)
        console.print(
            f"  [green]✓[/green] decision=[bold]{decision}[/bold] "
            f"(confidence={analysis.confidence:.0%}"
            + (", forced to human by a security/blocker finding" if forced else "")
            + ")"
        )
        await self.audit(
            state,
            action="route",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="pending",
            status="done",
            reason=f"confidence={analysis.confidence:.2f} → {decision}"
            + (" (security/blocker override)" if forced else ""),
            execution_time_ms=self.clock.elapsed_ms(t0),
        )
        return {"decision": decision}

    # -- terminals ---------------------------------------------------------
    def node_auto_approve_placeholder(self, state: ReviewState) -> dict:
        console.print("[green]✓ AUTO APPROVE[/green] — high confidence, no human needed")
        return {"final_action": "auto_approved"}

    def node_human_approval_placeholder(self, state: ReviewState) -> dict:
        console.print("[yellow]✓ HUMAN APPROVAL[/yellow] — exercise 2 pauses here")
        return {"final_action": "pending_human_approval"}

    def node_escalate_placeholder(self, state: ReviewState) -> dict:
        console.print("[red]✓ ESCALATE[/red] — exercise 3 asks the reviewer questions")
        return {"final_action": "pending_escalation"}

    # -- HITL --------------------------------------------------------------
    async def node_human_approval(self, state: ReviewState) -> dict:
        """Pause and ask the reviewer to approve / reject / edit."""
        console.print("[cyan]→ human_approval[/cyan] (interrupt)")
        t0 = self.clock.now()
        analysis: PRAnalysis = state["analysis"]

        await self.audit(
            state,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="pending",
            status="waiting",
            reason="Approval requested: "
            + ("security/blocker finding" if has_blocking_risk(analysis) else "medium confidence"),
            execution_time_ms=self.clock.elapsed_ms(t0),
        )

        response = interrupt(
            {
                "kind": "approval_request",
                "pr_url": state["pr_url"],
                "thread_id": state.get("thread_id"),
                "confidence": analysis.confidence,
                "confidence_reasoning": analysis.confidence_reasoning,
                "summary": analysis.summary,
                "risk_factors": list(analysis.risk_factors),
                "comments": [c.model_dump() for c in analysis.comments],
                "diff_preview": (state.get("pr_diff") or "")[:DIFF_PREVIEW_CHARS],
            }
        )
        response = response or {}
        choice = response.get("choice") or "reject"
        feedback = response.get("feedback") or ""
        reviewer = _reviewer_id(state, self.github)
        console.print(f"  [green]✓[/green] reviewer chose [bold]{choice}[/bold]")

        await self.audit(
            state,
            action="human_approval",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision=str(choice),
            status="responded",
            reason=feedback or f"human chose {choice}",
            execution_time_ms=self.clock.elapsed_ms(t0),
            reviewer_id=reviewer,
        )
        return {
            "human_choice": choice,
            "human_feedback": feedback or None,
            "reviewer_id": reviewer,
        }

    async def node_escalate(self, state: ReviewState) -> dict:
        """Ask the reviewer specific questions; store their answers."""
        console.print("[cyan]→ escalate[/cyan] (interrupt)")
        t0 = self.clock.now()
        analysis: PRAnalysis = state["analysis"]
        questions = list(analysis.escalation_questions)
        if not questions:
            files = state.get("pr_files") or []
            subject = files[0] if files else "this PR"
            questions = [
                f"Why is this change to {subject} needed now, and what is the "
                "rollout / rollback plan?",
                "Which of the risk factors above are already mitigated, and how?",
            ]

        await self.audit(
            state,
            action="escalate",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="pending",
            status="waiting",
            reason=f"Escalated with {len(questions)} question(s): "
            + _clip(" | ".join(questions), 300),
            execution_time_ms=self.clock.elapsed_ms(t0),
        )

        answers = interrupt(
            {
                "kind": "escalation",
                "pr_url": state["pr_url"],
                "thread_id": state.get("thread_id"),
                "confidence": analysis.confidence,
                "confidence_reasoning": analysis.confidence_reasoning,
                "summary": analysis.summary,
                "risk_factors": list(analysis.risk_factors),
                "questions": questions,
                "diff_preview": (state.get("pr_diff") or "")[:DIFF_PREVIEW_CHARS],
            }
        )
        answers = {q: (answers or {}).get(q, "") for q in questions}
        reviewer = _reviewer_id(state, self.github)
        console.print(f"  [green]✓[/green] reviewer answered {len(answers)} question(s)")

        await self.audit(
            state,
            action="escalate",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="escalate",
            status="responded",
            reason="Reviewer answers: "
            + _clip(" | ".join(f"{k} → {v}" for k, v in answers.items())),
            execution_time_ms=self.clock.elapsed_ms(t0),
            reviewer_id=reviewer,
        )
        return {"escalation_answers": answers, "reviewer_id": reviewer}

    async def node_synthesize(self, state: ReviewState) -> dict:
        """Re-analyze with the reviewer's answers, keeping unresolved risks."""
        console.print("[cyan]→ synthesize[/cyan]")
        t0 = self.clock.now()
        initial: PRAnalysis = state["analysis"]
        messages = build_synthesize_messages(
            title=state.get("pr_title", ""),
            diff=state.get("pr_diff", ""),
            initial=initial,
            answers=state.get("escalation_answers") or {},
            files=state.get("pr_files") or [],
        )
        try:
            with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
                refined = await self.llm_factory(temperature=0.2).ainvoke(messages)
        except Exception as exc:
            message = f"LLM synthesis failed: {type(exc).__name__}: {exc}"
            console.print(f"  [red]✗[/red] {message}")
            await self.audit(
                state,
                action="synthesize",
                confidence=initial.confidence,
                risk_level=audit_risk_level(initial),
                decision="failed",
                status="failed",
                reason=message,
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            # Keep the original review rather than losing the session.
            return {"analysis_error": message, "refined": False}

        # An answer must not silently erase an unresolved blocker.
        refined_blocker_locations = {
            (c.file, c.line) for c in refined.comments if c.severity == "blocker"
        }
        dropped = [
            c.body
            for c in initial.comments
            if c.severity == "blocker" and (c.file, c.line) not in refined_blocker_locations
        ]
        if dropped:
            console.print(
                f"  [yellow]![/yellow] re-instating {len(dropped)} unresolved blocker finding(s)"
            )
            refined = _reinstate_blockers(refined, initial)

        if refined.confidence > initial.confidence and not state.get("escalation_answers"):
            refined = refined.model_copy(update={"confidence": initial.confidence})

        console.print(
            f"  [green]✓[/green] refined confidence={refined.confidence:.0%} "
            f"(was {initial.confidence:.0%})"
        )
        await self.audit(
            state,
            action="synthesize",
            confidence=refined.confidence,
            risk_level=audit_risk_level(refined),
            decision="pending",
            status="done",
            reason=_clip(
                f"refined confidence {initial.confidence:.2f} → {refined.confidence:.2f}"
                + (
                    f"; unresolved risks kept: {' | '.join(dropped)}"
                    if dropped
                    else ""
                )
            ),
            execution_time_ms=self.clock.elapsed_ms(t0),
        )
        return {"analysis": refined}

    async def node_final_approval(self, state: ReviewState) -> dict:
        """After escalation: show the refined review and ask to post it."""
        console.print("[cyan]→ final_approval[/cyan] (interrupt)")
        t0 = self.clock.now()
        analysis: PRAnalysis = state["analysis"]

        await self.audit(
            state,
            action="final_approval",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="pending",
            status="waiting",
            reason="Final confirmation requested before posting the refined review",
            execution_time_ms=self.clock.elapsed_ms(t0),
        )

        response = interrupt(
            {
                "kind": "final_confirmation",
                "pr_url": state["pr_url"],
                "thread_id": state.get("thread_id"),
                "confidence": analysis.confidence,
                "confidence_reasoning": analysis.confidence_reasoning,
                "summary": analysis.summary,
                "risk_factors": list(analysis.risk_factors),
                "comments": [c.model_dump() for c in analysis.comments],
                "escalation_answers": state.get("escalation_answers") or {},
                "diff_preview": (state.get("pr_diff") or "")[:DIFF_PREVIEW_CHARS],
            }
        )
        response = response or {}
        confirmed = bool(response.get("confirm"))
        feedback = response.get("feedback") or ""
        reviewer = _reviewer_id(state, self.github)
        console.print(f"  [green]✓[/green] final confirmation: {confirmed}")

        await self.audit(
            state,
            action="final_approval",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="approve" if confirmed else "reject",
            status="responded",
            reason=feedback or ("confirmed posting" if confirmed else "declined posting"),
            execution_time_ms=self.clock.elapsed_ms(t0),
            reviewer_id=reviewer,
        )
        return {"refine_confirmed": confirmed, "human_feedback": feedback or None}

    # -- commit ------------------------------------------------------------
    async def node_commit(self, state: ReviewState) -> dict:
        console.print("[cyan]→ commit[/cyan]")
        t0 = self.clock.now()
        analysis: PRAnalysis = state["analysis"]
        allowed, refusal = self._posting_allowed(state)

        if not allowed:
            console.print(f"  [yellow]·[/yellow] not posting — {refusal}")
            await self.audit(
                state,
                action="commit",
                confidence=analysis.confidence,
                risk_level=audit_risk_level(analysis),
                decision="reject",
                status="done",
                reason=refusal,
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=state.get("reviewer_id"),
            )
            return {"final_action": "rejected"}

        body = build_comment_body(state)
        if self.dry_run:
            console.print("  [yellow]·[/yellow] dry-run: comment NOT posted to GitHub")
            await self.audit(
                state,
                action="commit",
                confidence=analysis.confidence,
                risk_level=audit_risk_level(analysis),
                # NOT "committed": nothing reached GitHub, and an auditor filtering
                # on `decision` must not read this as a posted review.
                decision="dry_run",
                status="done",
                reason=f"dry-run: {len(body)} chars would be posted",
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=state.get("reviewer_id"),
            )
            return {"final_action": "committed_dry_run", "posted_comment_body": body}

        try:
            posted_url = await asyncio.to_thread(
                self.github.post_review_comment,
                state["pr_url"],
                body,
                thread_id=state.get("thread_id", ""),
            )
        except GitHubError as exc:
            console.print(f"  [red]✗[/red] post failed: {exc.friendly}")
            await self.audit(
                state,
                action="commit",
                confidence=analysis.confidence,
                risk_level=audit_risk_level(analysis),
                decision="failed",
                status="failed",
                reason=exc.friendly,
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=state.get("reviewer_id"),
            )
            return {"final_action": "commit_failed", "post_error": exc.friendly}

        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        await self.audit(
            state,
            action="commit",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="committed",
            status="done",
            reason=f"Posted top-level PR comment ({posted_url})",
            execution_time_ms=self.clock.elapsed_ms(t0),
            reviewer_id=state.get("reviewer_id"),
        )
        return {"final_action": "committed", "posted_comment_body": body}

    async def node_auto_approve(self, state: ReviewState) -> dict:
        """High confidence, no human — post directly (still via `commit` shape)."""
        console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
        t0 = self.clock.now()
        analysis: PRAnalysis = state["analysis"]

        if has_blocking_risk(analysis):  # defensive: route should have caught this
            console.print("  [red]✗[/red] security/blocker finding — human required")
            await self.audit(
                state,
                action="auto_approve",
                confidence=analysis.confidence,
                risk_level="high",
                decision="escalate",
                status="done",
                reason="Refused auto-approval: security or blocker finding present",
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {"final_action": "auto_refused_security"}

        body = build_comment_body(state)
        if self.dry_run:
            console.print("  [yellow]·[/yellow] dry-run: comment NOT posted to GitHub")
            await self.audit(
                state,
                action="auto_approve",
                confidence=analysis.confidence,
                risk_level="low",
                decision="dry_run",  # not "auto": nothing reached GitHub
                status="done",
                reason=f"dry-run: {len(body)} chars would be posted",
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {"final_action": "auto_approved_dry_run", "posted_comment_body": body}

        try:
            posted_url = await asyncio.to_thread(
                self.github.post_review_comment,
                state["pr_url"],
                body,
                thread_id=state.get("thread_id", ""),
            )
        except GitHubError as exc:
            console.print(f"  [red]✗[/red] post failed: {exc.friendly}")
            await self.audit(
                state,
                action="auto_approve",
                confidence=analysis.confidence,
                risk_level="low",
                decision="failed",
                status="failed",
                reason=exc.friendly,
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {"final_action": "auto_approve_failed", "post_error": exc.friendly}

        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        await self.audit(
            state,
            action="auto_approve",
            confidence=analysis.confidence,
            risk_level="low",
            decision="auto",
            status="done",
            reason=f"Auto-posted top-level PR comment ({posted_url})",
            execution_time_ms=self.clock.elapsed_ms(t0),
        )
        return {"final_action": "auto_approved", "posted_comment_body": body}

    # -- plumbing ----------------------------------------------------------
    def _posting_allowed(self, state: ReviewState) -> tuple[bool, str]:
        """Decide whether `commit` may post, and why not when it may not."""
        if state.get("refined") is False or state.get("analysis_error"):
            return False, "refined analysis unavailable"
        if state.get("escalation_answers"):
            if self.mode.has_final_confirmation and not state.get("refine_confirmed"):
                return False, "reviewer did not confirm posting after escalation"
            return True, ""
        if state.get("human_choice") == "approve":
            return True, ""
        return False, f"human chose {state.get('human_choice') or 'nothing'}"

    async def node_fetch_failed(self, state: ReviewState) -> dict:
        console.print("[red]✗ aborting: PR could not be fetched[/red]")
        return {"final_action": "fetch_failed"}

    async def node_analyze_failed(self, state: ReviewState) -> dict:
        console.print("[red]✗ aborting: analysis failed — nothing was posted[/red]")
        return {"final_action": "analyze_failed"}


def _reinstate_blockers(refined: PRAnalysis, initial: PRAnalysis) -> PRAnalysis:
    """Put back any blocker finding the refined analysis no longer reports.

    Resolution is judged by *location* (file + line): a blocker at ``file:line``
    counts as resolved only when the refined review still reports a blocker
    there. Softening it to a "suggestion" — or replacing it with a different
    concern at the same spot — is not evidence that the risk went away, and the
    lab explicitly requires an unresolved security risk to survive escalation.
    """
    refined_blocker_locations = {
        (c.file, c.line) for c in refined.comments if c.severity == "blocker"
    }
    restored = [
        c
        for c in initial.comments
        if c.severity == "blocker" and (c.file, c.line) not in refined_blocker_locations
    ]
    risks = list(dict.fromkeys([*refined.risk_factors, *initial.risk_factors]))
    return refined.model_copy(
        update={"comments": [*refined.comments, *restored], "risk_factors": risks}
    )


# ─── graph ────────────────────────────────────────────────────────────────


def build_graph(
    *,
    mode: Mode = Mode.EXERCISE_4,
    checkpointer: Any = None,
    llm_factory: LLMFactory | None = None,
    github: GitHubService | None = None,
    audit_sink: AuditSink | None = None,
    clock: _Clock | None = None,
    dry_run: bool = False,
):
    """Compile the graph for ``mode``.

    ``checkpointer`` is required for any mode with HITL (``interrupt()`` needs
    one). When omitted, a :class:`~langgraph.checkpoint.memory.MemorySaver` is
    used — that is exercise 2's configuration; exercises 4–5 pass in an
    ``AsyncSqliteSaver`` so checkpoints survive a restart.
    """
    mode = Mode(mode)
    agent = Agent(
        mode=mode,
        llm_factory=llm_factory,
        github=github,
        audit_sink=audit_sink,
        clock=clock,
        dry_run=dry_run,
    )
    g = StateGraph(ReviewState)

    g.add_node("fetch_pr", agent.node_fetch_pr)
    g.add_node("fetch_failed", agent.node_fetch_failed)
    g.add_node("analyze", agent.node_analyze)
    g.add_node("analyze_failed", agent.node_analyze_failed)
    g.add_node("route", agent.node_route)

    g.add_edge(START, "fetch_pr")
    g.add_conditional_edges(
        "fetch_pr",
        lambda s: "fetch_failed" if s.get("fetch_error") else "analyze",
        {"fetch_failed": "fetch_failed", "analyze": "analyze"},
    )
    g.add_conditional_edges(
        "analyze",
        lambda s: "analyze_failed" if s.get("analysis_error") else "route",
        {"analyze_failed": "analyze_failed", "route": "route"},
    )
    g.add_edge("fetch_failed", END)
    g.add_edge("analyze_failed", END)

    if mode is Mode.EXERCISE_1:
        g.add_node("auto_approve", agent.node_auto_approve_placeholder)
        g.add_node("human_approval", agent.node_human_approval_placeholder)
        g.add_node("escalate", agent.node_escalate_placeholder)
        g.add_conditional_edges(
            "route",
            lambda s: s["decision"],
            {
                "auto_approve": "auto_approve",
                "human_approval": "human_approval",
                "escalate": "escalate",
            },
        )
        for terminal in ("auto_approve", "human_approval", "escalate"):
            g.add_edge(terminal, END)
        return g.compile(checkpointer=checkpointer)

    # exercises 2–5: HITL + commit
    g.add_node("auto_approve", agent.node_auto_approve)
    g.add_node("human_approval", agent.node_human_approval)
    g.add_node("commit", agent.node_commit)
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
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)

    # `escalate` uses the real Q&A node in every HITL mode (the reviewer answers
    # specific questions) and `synthesize` refines the review with those answers.
    # Exercise 2's skeleton has no `commit` path after escalation, so it stops at
    # `synthesize`; exercises 3–5 continue to commit (with a final confirmation in
    # 4–5). `synthesize` can fail over to `commit`, which refuses to post without
    # a confirmed analysis.
    g.add_node("escalate", agent.node_escalate)
    g.add_node("synthesize", agent.node_synthesize)
    g.add_edge("escalate", "synthesize")
    if mode is Mode.EXERCISE_2:
        g.add_edge("synthesize", END)
        return g.compile(checkpointer=checkpointer or memory_checkpointer())

    if mode.has_final_confirmation:
        g.add_node("final_approval", agent.node_final_approval)
        g.add_conditional_edges(
            "synthesize",
            lambda s: "commit"
            if s.get("refined") is False or s.get("analysis_error")
            else "final_approval",
            {"final_approval": "final_approval", "commit": "commit"},
        )
        g.add_edge("final_approval", "commit")
    else:
        g.add_edge("synthesize", "commit")

    return g.compile(checkpointer=checkpointer or memory_checkpointer())


def config_for(thread_id: str) -> dict:
    """LangGraph runnable config for a thread."""
    return {"configurable": {"thread_id": thread_id}}


__all__ = [
    "AGENT_ID",
    "Agent",
    "AuditSink",
    "GitHubService",
    "LLMBinding",
    "LLMFactory",
    "Mode",
    "RealGitHubService",
    "RealLLMFactory",
    "build_comment_body",
    "build_graph",
    "config_for",
    "console",
    "route_for_by_confidence_only",
]
