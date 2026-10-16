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
from collections.abc import Mapping
from enum import Enum
from typing import Any, Callable, Protocol

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from rich.console import Console

from common.db import memory_checkpointer
from common.github import (
    GitHubError,
    PostResult,
    PostStatus,
    PullRequest,
    current_user,
    fetch_pr,
    post_review_comment,
)
from common.llm import StructuredOutputLLM, get_llm, structured_llm
from common.prompts import (
    build_analyze_messages,
    build_edit_messages,
    build_synthesize_messages,
    force_utf8_stdio,
)
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

    def post_review_comment(
        self, pr_url: str, body: str, *, thread_id: str
    ) -> PostResult: ...

    def current_user(self) -> str | None: ...


class RealLLMFactory:
    """Default :class:`LLMFactory` — reads model config from the environment."""

    def __call__(self, *, temperature: float = 0.2) -> LLMBinding:
        return structured_llm(PRAnalysis, llm=get_llm(temperature))


class RealGitHubService:
    """Default :class:`GitHubService` — talks to the GitHub REST API."""

    def fetch_pr(self, pr_url: str) -> PullRequest:
        return fetch_pr(pr_url)

    def post_review_comment(self, pr_url: str, body: str, *, thread_id: str) -> PostResult:
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

    Precedence:

    1. ``edited_comment_body`` — the reviewer wrote/replaced this text themselves.
       Human-authored content is posted **verbatim**: the LLM never overwrites it
       and nothing is appended except the thread marker added by the client.
    2. Otherwise the review is rendered from ``analysis`` (initial, refined, or
       LLM-edited), with the escalation answers and reviewer notes attached.

    Escalation answers may be long, but they are the evidence for the refined
    review, so they are included (quoted) rather than dropped.
    """
    human_body = (state.get("edited_comment_body") or "").strip()
    if human_body:
        return human_body

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
    if state.get("edit_revision"):
        lines.append(
            f"_Review revised by the reviewer (revision {state['edit_revision']})._"
        )
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
        dry_run: bool | None = False,
    ) -> None:
        self.mode = mode
        self.llm_factory: LLMFactory = llm_factory or RealLLMFactory()
        self.github: GitHubService = github or RealGitHubService()
        self.audit_sink: AuditSink = audit_sink or _default_audit_sink
        self.clock = clock or _Clock()
        #: Fallback only. The authoritative value lives in the graph state so a
        #: resume cannot flip a rehearsal into a real post; see
        #: :meth:`effective_dry_run`.
        self.dry_run = dry_run
        #: Last audit-write failure, surfaced into the state by the terminal nodes
        #: so a "successful" run with no audit evidence cannot pass silently.
        self.audit_error: str | None = None

    # -- configuration -----------------------------------------------------
    def effective_dry_run(self, state: Mapping[str, Any] | dict) -> bool:
        """Whether THIS session may touch GitHub.

        Precedence (this is the whole FIX-2 contract):

        1. ``state["dry_run"]`` — persisted when the thread was created. A resume
           therefore keeps the original mode no matter what the caller passes to
           ``build_graph``.
        2. ``self.dry_run`` — explicit configuration for callers that never seed
           the state (direct API/tests).
        3. ``None`` → **True**: a checkpoint with no recorded mode is of unknown
           provenance, so the safe answer is "do not post".
        """
        if isinstance(state, Mapping) and "dry_run" in state:
            return bool(state["dry_run"])
        if self.dry_run is None:
            return True
        return bool(self.dry_run)

    def dry_run_origin(self, state: Mapping[str, Any] | dict) -> str:
        """Where the effective dry-run value came from (for logs/audit)."""
        if isinstance(state, Mapping) and "dry_run" in state:
            return "session"
        if self.dry_run is None:
            return "fail-safe (no dry_run recorded in this checkpoint)"
        return "config"

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
        revision: int | None = None,
    ) -> None:
        """Emit one audit row (no-op unless the mode has audit enabled).

        A failure is logged *and* recorded in ``self.audit_error`` so the terminal
        node can surface it: a mode that promises an audit trail must not report
        success with no evidence of the decision.
        """
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
            revision=state.get("edit_revision", 0) if revision is None else revision,
            reason=_clip(reason),
            execution_time_ms=execution_time_ms,
        )
        try:
            await self.audit_sink(state, entry)
            self.audit_error = None
        except Exception as exc:
            self.audit_error = f"{action}: {type(exc).__name__}: {exc}"
            logger.warning("audit write failed for %s: %s", action, exc)
            console.print(
                f"  [yellow]![/yellow] audit write failed for {action}: {exc}"
            )

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
        # An `edit` response may carry the reviewer's replacement text directly
        # (`edited_body`). That is human-authored content and is what gets posted,
        # verbatim — see `build_comment_body`.
        edited_body = (response.get("edited_body") or "").strip() or None
        reviewer = _reviewer_id(state, self.github)
        console.print(
            f"  [green]✓[/green] reviewer chose [bold]{choice}[/bold]"
            + (" (with replacement text)" if edited_body else "")
        )

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
            # `edit` continues into the Edit loop instead of ending the graph.
            "edit_instructions": feedback or None if choice == "edit" else None,
            "edited_comment_body": edited_body if choice == "edit" else None,
        }

    # -- Edit loop ---------------------------------------------------------
    async def node_edit_review(self, state: ReviewState) -> dict:
        """Revise the review after the reviewer chose Edit.

        Two ways in, in priority order:

        * ``edited_comment_body`` (the reviewer typed the replacement review) —
          stored as-is and NOT touched by the LLM. Human edits win.
        * otherwise the reviewer's ``edit_instructions`` are applied by the LLM,
          with the same guarantees as synthesis: blocker findings survive, and
          confidence is never raised.
        """
        revision = int(state.get("edit_revision") or 0) + 1
        console.print(f"[cyan]→ edit_review[/cyan] (revision {revision})")
        t0 = self.clock.now()
        current: PRAnalysis = state["analysis"]
        instructions = (state.get("edit_instructions") or "").strip()
        human_body = (state.get("edited_comment_body") or "").strip()

        if human_body:
            console.print(
                "  [green]✓[/green] using the reviewer's replacement text verbatim "
                f"({len(human_body)} chars) — the LLM will not touch it"
            )
            await self.audit(
                state,
                action="edit_review",
                confidence=current.confidence,
                risk_level=audit_risk_level(current),
                decision="edit",
                status="done",
                reason=f"revision {revision}: reviewer supplied the review text verbatim",
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=state.get("reviewer_id"),
                revision=revision,
            )
            return {
                "edit_revision": revision,
                "analysis": current,
                "refined": True,
            }

        if not instructions:
            # Nothing to apply: keep the review as-is rather than guess.
            console.print("  [yellow]![/yellow] no edit instructions — review unchanged")
            await self.audit(
                state,
                action="edit_review",
                confidence=current.confidence,
                risk_level=audit_risk_level(current),
                decision="edit",
                status="done",
                reason=f"revision {revision}: no instructions given, review unchanged",
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=state.get("reviewer_id"),
                revision=revision,
            )
            return {"edit_revision": revision}

        messages = build_edit_messages(
            title=state.get("pr_title", ""),
            diff=state.get("pr_diff", ""),
            current=current,
            instructions=instructions,
            revision=revision,
            files=state.get("pr_files") or [],
        )
        try:
            with console.status("[dim]LLM revising the review...[/dim]"):
                revised = await self.llm_factory(temperature=0.2).ainvoke(messages)
        except Exception as exc:
            message = f"LLM edit failed: {type(exc).__name__}: {exc}"
            console.print(f"  [red]✗[/red] {message}")
            await self.audit(
                state,
                action="edit_review",
                confidence=current.confidence,
                risk_level=audit_risk_level(current),
                decision="failed",
                status="failed",
                reason=message,
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=state.get("reviewer_id"),
                revision=revision,
            )
            return {"edit_revision": revision, "analysis_error": message, "refined": False}

        # Same guarantees as synthesis: an edit instruction cannot delete a blocker
        # or talk the model into being more certain.
        revised = _keep_blockers_revised(revised, current)
        if revised.confidence > current.confidence:
            revised = revised.model_copy(
                update={
                    "confidence": current.confidence,
                    "confidence_reasoning": (
                        f"{revised.confidence_reasoning} "
                        f"(confidence capped at the previous {current.confidence:.2f}: "
                        "an editorial revision does not change how certain the review is)"
                    ),
                }
            )

        console.print(
            f"  [green]✓[/green] revision {revision}: "
            f"{len(revised.comments)} finding(s), confidence={revised.confidence:.0%}"
        )
        await self.audit(
            state,
            action="edit_review",
            confidence=revised.confidence,
            risk_level=audit_risk_level(revised),
            decision="edit",
            status="done",
            reason=_clip(
                f"revision {revision} applied reviewer instructions: {instructions}"
            ),
            execution_time_ms=self.clock.elapsed_ms(t0),
            reviewer_id=state.get("reviewer_id"),
            revision=revision,
        )
        return {"edit_revision": revision, "analysis": revised, "refined": True}

    async def node_final_edit_approval(self, state: ReviewState) -> dict:
        """Show the edited review and ask whether to post it.

        Answer with ``{"confirm": True|False, "feedback": str}``; an
        ``{"choice": "edit"}`` answer loops back into ``edit_review`` for another
        pass. Nothing is posted without a confirmation here.
        """
        console.print("[cyan]→ final_edit_approval[/cyan] (interrupt)")
        t0 = self.clock.now()
        analysis: PRAnalysis = state["analysis"]
        revision = int(state.get("edit_revision") or 0)

        await self.audit(
            state,
            action="final_edit_approval",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="pending",
            status="waiting",
            reason=f"Confirmation requested for revision {revision}",
            execution_time_ms=self.clock.elapsed_ms(t0),
            revision=revision,
        )

        response = interrupt(
            {
                "kind": "edited_review",
                "pr_url": state["pr_url"],
                "thread_id": state.get("thread_id"),
                "confidence": analysis.confidence,
                "confidence_reasoning": analysis.confidence_reasoning,
                "summary": analysis.summary,
                "risk_factors": list(analysis.risk_factors),
                "comments": [c.model_dump() for c in analysis.comments],
                "revision": revision,
                "edited_comment_body": state.get("edited_comment_body"),
                "proposed_body": build_comment_body(state),
                "diff_preview": (state.get("pr_diff") or "")[:DIFF_PREVIEW_CHARS],
            }
        )
        response = response or {}
        reviewer = _reviewer_id(state, self.github)

        # "edit again" → another revision; otherwise this is a yes/no on posting.
        if (response.get("choice") or "").lower() == "edit":
            console.print("  [yellow]↻[/yellow] reviewer wants another revision")
            await self.audit(
                state,
                action="final_edit_approval",
                confidence=analysis.confidence,
                risk_level=audit_risk_level(analysis),
                decision="edit",
                status="responded",
                reason=response.get("feedback") or "another revision requested",
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=reviewer,
                revision=revision,
            )
            return {
                "edit_confirmed": None,
                "edit_instructions": response.get("feedback") or None,
                "edited_comment_body": (response.get("edited_body") or "").strip() or None,
                "reviewer_id": reviewer,
            }

        confirmed = bool(response.get("confirm"))
        feedback = response.get("feedback") or ""
        console.print(f"  [green]✓[/green] revision {revision} confirmed: {confirmed}")
        await self.audit(
            state,
            action="final_edit_approval",
            confidence=analysis.confidence,
            risk_level=audit_risk_level(analysis),
            decision="approve" if confirmed else "reject",
            status="responded",
            reason=feedback or ("confirmed the edited review" if confirmed else "declined posting"),
            execution_time_ms=self.clock.elapsed_ms(t0),
            reviewer_id=reviewer,
            revision=revision,
        )
        return {
            "edit_confirmed": confirmed,
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
            return {
                "final_action": "rejected",
                "post_status": "rejected",
                "audit_error": self.audit_error,
            }

        body = build_comment_body(state)
        if self.effective_dry_run(state):
            console.print(
                "  [yellow]·[/yellow] dry-run: comment NOT posted to GitHub "
                f"[dim]({self.dry_run_origin(state)})[/dim]"
            )
            await self.audit(
                state,
                action="commit",
                confidence=analysis.confidence,
                risk_level=audit_risk_level(analysis),
                # NOT "committed": nothing reached GitHub, and an auditor filtering
                # on `decision` must not read this as a posted review.
                decision="dry_run",
                status="done",
                reason=f"dry-run ({self.dry_run_origin(state)}): {len(body)} chars would be posted",
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=state.get("reviewer_id"),
            )
            return {
                "final_action": "committed_dry_run",
                "posted_comment_body": body,
                "post_status": "dry_run",
                "audit_error": self.audit_error,
            }

        return await self._post(state, body, analysis, t0, action="commit", auto=False)

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
            return {
                "final_action": "auto_refused_security",
                "post_status": "failed",
                "audit_error": self.audit_error,
            }

        body = build_comment_body(state)
        if self.effective_dry_run(state):
            console.print(
                "  [yellow]·[/yellow] dry-run: comment NOT posted to GitHub "
                f"[dim]({self.dry_run_origin(state)})[/dim]"
            )
            await self.audit(
                state,
                action="auto_approve",
                confidence=analysis.confidence,
                risk_level="low",
                decision="dry_run",  # not "auto": nothing reached GitHub
                status="done",
                reason=f"dry-run ({self.dry_run_origin(state)}): {len(body)} chars would be posted",
                execution_time_ms=self.clock.elapsed_ms(t0),
            )
            return {
                "final_action": "auto_approved_dry_run",
                "posted_comment_body": body,
                "post_status": "dry_run",
                "audit_error": self.audit_error,
            }

        return await self._post(state, body, analysis, t0, action="auto_approve", auto=True)

    async def _post(
        self,
        state: ReviewState,
        body: str,
        analysis: PRAnalysis,
        t0: float,
        *,
        action: str,
        auto: bool,
    ) -> dict:
        """Perform the one POST this graph is allowed to make, and report honestly.

        Maps :class:`~common.github.PostResult` onto a final action:

        * ``posted``          → success
        * ``already_posted``  → success (the review IS on the PR, from an earlier
          attempt); reported distinctly so a retry is visible
        * ``unconfirmed``     → neither success nor failure; GitHub may or may not
          have created the comment, so automation must not assume either
        * ``GitHubError``     → definitive failure (401/403/404/429/422)
        """
        prefix = "auto_approve" if auto else "commit"
        risk = "low" if auto else audit_risk_level(analysis)
        reviewer = None if auto else state.get("reviewer_id")

        try:
            result = await asyncio.to_thread(
                self.github.post_review_comment,
                state["pr_url"],
                body,
                thread_id=state.get("thread_id", ""),
            )
        except GitHubError as exc:
            console.print(f"  [red]✗[/red] post failed: {exc.friendly}")
            await self.audit(
                state,
                action=prefix,
                confidence=analysis.confidence,
                risk_level=risk,
                decision="failed",
                status="failed",
                reason=exc.friendly,
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=reviewer,
            )
            return {
                "final_action": f"{prefix}_failed",
                "post_error": exc.friendly,
                "post_status": "failed",
                "audit_error": self.audit_error,
            }

        if result.status == PostStatus.UNCONFIRMED:
            console.print(
                "  [yellow]![/yellow] GitHub accepted the request but the outcome "
                "could not be confirmed — NOT retried"
            )
            await self.audit(
                state,
                action=prefix,
                confidence=analysis.confidence,
                risk_level=risk,
                decision="unconfirmed",
                status="failed",
                reason=result.detail,
                execution_time_ms=self.clock.elapsed_ms(t0),
                reviewer_id=reviewer,
            )
            return {
                "final_action": "commit_unconfirmed",
                "post_error": result.detail,
                "post_status": "unconfirmed",
                "audit_error": self.audit_error,
            }

        already = result.status == PostStatus.ALREADY_POSTED
        if already:
            console.print(
                f"  [green]✓[/green] review already on {state['pr_url']} "
                "(earlier attempt) — not posted again"
            )
            decision = "committed"
            reason = f"Already on the PR, not re-posted ({result.detail})"
        else:
            console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
            decision = "auto" if auto else "committed"
            reason = f"Posted top-level PR comment ({result.comment_url})"

        await self.audit(
            state,
            action=prefix,
            confidence=analysis.confidence,
            risk_level=risk,
            decision=decision,
            status="done",
            reason=reason,
            execution_time_ms=self.clock.elapsed_ms(t0),
            reviewer_id=reviewer,
        )
        final_action = (
            "auto_approved" if auto else ("committed_already" if already else "committed")
        )
        return {
            "final_action": final_action,
            "posted_comment_body": body,
            "post_status": "already_posted" if already else "posted",
            "audit_error": self.audit_error,
        }

    # -- plumbing ----------------------------------------------------------
    def _posting_allowed(self, state: ReviewState) -> tuple[bool, str]:
        """Decide whether `commit` may post, and why not when it may not.

        Exactly one human decision authorises a post, whichever path got here:

        * escalation answered **and** confirmed (when the mode has that gate), or
        * the reviewer approved the original review, or
        * the reviewer confirmed the review produced by the Edit loop.
        """
        if state.get("refined") is False or state.get("analysis_error"):
            return False, "reviewed analysis unavailable"
        if state.get("escalation_answers"):
            if self.mode.has_final_confirmation and not state.get("refine_confirmed"):
                return False, "reviewer did not confirm posting after escalation"
            return True, ""
        if state.get("edit_revision"):
            # The Edit loop was entered: it is the edit confirmation that counts,
            # not the original `edit` choice.
            if state.get("edit_confirmed") is True:
                return True, ""
            return False, "reviewer did not confirm the edited review"
        if state.get("human_choice") == "approve":
            return True, ""
        if state.get("human_choice") == "edit":
            return False, "reviewer chose edit but no confirmed revision exists"
        return False, f"human chose {state.get('human_choice') or 'nothing'}"

    async def node_fetch_failed(self, state: ReviewState) -> dict:
        console.print("[red]✗ aborting: PR could not be fetched[/red]")
        return {
            "final_action": "fetch_failed",
            "post_status": "failed",
            "audit_error": self.audit_error,
        }

    async def node_analyze_failed(self, state: ReviewState) -> dict:
        console.print("[red]✗ aborting: analysis failed — nothing was posted[/red]")
        return {
            "final_action": "analyze_failed",
            "post_status": "failed",
            "audit_error": self.audit_error,
        }


def _reinstate_blockers(refined: PRAnalysis, initial: PRAnalysis) -> PRAnalysis:
    """Put back any blocker finding the newer analysis no longer reports.

    Resolution is judged by *location* (file + line): a blocker at ``file:line``
    counts as resolved only when the new review still reports a blocker there.
    Softening it to a "suggestion" — or replacing it with a different concern at
    the same spot — is not evidence that the risk went away.

    Used by both rewrite paths, because the same rule has to hold whether the
    second pass came from the reviewer's escalation answers or from their Edit
    instructions: a human request cannot silently delete a security blocker.
    """
    new_blocker_locations = {
        (c.file, c.line) for c in refined.comments if c.severity == "blocker"
    }
    restored = [
        c
        for c in initial.comments
        if c.severity == "blocker" and (c.file, c.line) not in new_blocker_locations
    ]
    risks = list(dict.fromkeys([*refined.risk_factors, *initial.risk_factors]))
    return refined.model_copy(
        update={"comments": [*refined.comments, *restored], "risk_factors": risks}
    )


#: The Edit loop applies the same blocker-retention rule as synthesis.
_keep_blockers_revised = _reinstate_blockers


# ─── graph ────────────────────────────────────────────────────────────────


def build_graph(
    *,
    mode: Mode = Mode.EXERCISE_4,
    checkpointer: Any = None,
    llm_factory: LLMFactory | None = None,
    github: GitHubService | None = None,
    audit_sink: AuditSink | None = None,
    clock: _Clock | None = None,
    dry_run: bool | None = False,
):
    """Compile the graph for ``mode``.

    ``checkpointer`` is required for any mode with HITL (``interrupt()`` needs
    one). When omitted, a :class:`~langgraph.checkpoint.memory.MemorySaver` is
    used — that is exercise 2's configuration; exercises 4–5 pass in an
    ``AsyncSqliteSaver`` so checkpoints survive a restart.

    ``dry_run`` is only a **fallback** for callers that never seed the graph
    state: the authoritative value is ``state["dry_run"]``, which the entry points
    persist when the session starts. Pass ``None`` to mean "trust the state, and
    refuse to post when the checkpoint does not record a mode" (the safe answer
    for a resumed, older thread). See :meth:`Agent.effective_dry_run`.
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
    g.add_edge("commit", END)

    # ── the Edit loop ─────────────────────────────────────────────────────
    # human_approval --edit--> edit_review --> final_edit_approval --confirm--> commit
    #                                              |  ^                     |
    #                                     reject --+  +--edit again---------+
    # A bare `edit` choice must NOT end the graph (it used to resolve to
    # "rejected") and must not post anything on its own.
    g.add_node("edit_review", agent.node_edit_review)
    g.add_node("final_edit_approval", agent.node_final_edit_approval)
    g.add_conditional_edges(
        "human_approval",
        lambda s: "edit_review" if s.get("human_choice") == "edit" else "commit",
        {"edit_review": "edit_review", "commit": "commit"},
    )
    g.add_edge("edit_review", "final_edit_approval")
    g.add_conditional_edges(
        "final_edit_approval",
        lambda s: "edit_review"
        if s.get("edit_confirmed") is None
        else "commit",
        {"edit_review": "edit_review", "commit": "commit"},
    )

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
    "PostResult",
    "PostStatus",
    "RealGitHubService",
    "RealLLMFactory",
    "build_comment_body",
    "build_graph",
    "config_for",
    "console",
    "route_for_by_confidence_only",
]
