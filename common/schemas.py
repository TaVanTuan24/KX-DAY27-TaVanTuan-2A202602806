"""Shared types: the graph state, the LLM's structured analysis output,
and the structured audit-trail entry written to the ``audit_events`` table."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


Decision = Literal["auto_approve", "human_approval", "escalate"]
HumanChoice = Literal["approve", "reject", "edit"]

#: Actions accepted by ``audit_events.action`` — one per meaningful graph event.
AuditAction = Literal[
    "fetch_pr",
    "analyze",
    "route",
    "auto_approve",
    "human_approval",
    "escalate",
    "synthesize",
    "final_approval",
    "commit",
]

#: Decisions accepted by ``audit_events.decision``.
AuditDecision = Literal[
    "auto", "approve", "reject", "edit", "escalate", "pending", "committed", "failed",
]

#: Lifecycle phase accepted by ``audit_events.status``.
AuditStatus = Literal[
    "pending",     # an automated step that has no outcome of its own
    "waiting",     # written just before an interrupt(): the human has not answered
    "responded",   # written just after resume: the human's answer is known
    "done",        # the step produced its final outcome
    "failed",      # the step failed (GitHub API error, ...)
]

#: Severities that force human review even when the model is very confident.
BLOCKING_SEVERITIES = frozenset({"blocker"})
#: Ordering of finding severities, worst last — used to sort/report findings
#: without relying on string comparison.
SEVERITY_RANK = {"nit": 0, "suggestion": 1, "issue": 2, "blocker": 3}


# Confidence thresholds
AUTO_APPROVE_THRESHOLD = 0.73   # >= 73% → no human, agent commits the comment directly
ESCALATE_THRESHOLD = 0.58       # < 58% → escalate: agent asks the reviewer specific questions
# 58–72% inclusive → human approval flow (reviewer clicks Approve / Reject / Edit)


def risk_level_for(confidence: float) -> str:
    """Map confidence → risk_level (used in AuditEntry).

    The thresholds invert: higher confidence ↔ lower risk.
    """
    if confidence >= AUTO_APPROVE_THRESHOLD:
        return "low"
    if confidence < ESCALATE_THRESHOLD:
        return "high"
    return "med"


def _looks_security_related(text: str) -> bool:
    """Cheap keyword heuristic for "this risk factor is security-relevant"."""
    lowered = text.lower()
    return any(
        kw in lowered
        for kw in (
            "secur", "auth", "password", "credential", "secret", "token",
            "inject", "xss", "csrf", "ssrf", "sql", "crypto", "hash", "md5",
            "sha1", "tls", "ssl", "permission", "privilege", "sanitiz",
            "vulnerab", "exploit", "pii", "leak",
        )
    )


def has_blocking_risk(analysis: "PRAnalysis") -> bool:
    """True when the analysis contains a security issue or blocker.

    Such PRs always require a human, whatever the self-reported confidence
    (lab requirement §9.2: "với lỗi bảo mật nghiêm trọng hoặc blocker, bắt
    buộc yêu cầu người duyệt trước khi đăng, kể cả confidence cao").
    """
    if any(c.severity in BLOCKING_SEVERITIES for c in analysis.comments):
        return True
    return any(_looks_security_related(rf) for rf in analysis.risk_factors)


def route_for(analysis: "PRAnalysis") -> Decision:
    """Single source of truth for confidence-based routing.

    Boundaries (inclusive on the lower edge of each band):

    * ``confidence >= AUTO_APPROVE_THRESHOLD`` → ``auto_approve``
    * ``ESCALATE_THRESHOLD <= confidence < AUTO_APPROVE_THRESHOLD`` → ``human_approval``
    * ``confidence < ESCALATE_THRESHOLD`` → ``escalate``

    Overridden to ``human_approval`` when the analysis carries a security /
    blocker risk (see :func:`has_blocking_risk`).
    """
    confidence = analysis.confidence
    if confidence < ESCALATE_THRESHOLD:
        return "escalate"
    if confidence >= AUTO_APPROVE_THRESHOLD and not has_blocking_risk(analysis):
        return "auto_approve"
    return "human_approval"


def audit_risk_level(analysis: "PRAnalysis") -> str:
    """risk_level for an ``AuditEntry`` built from an analysis.

    A security/blocker finding is ``high`` risk regardless of confidence — the
    audit trail must not report ``risk_level='low' AND decision='auto'`` for a
    PR that only escaped auto-approval because of a security finding.
    """
    if has_blocking_risk(analysis):
        return "high"
    return risk_level_for(analysis.confidence)


class ReviewComment(BaseModel):
    """A single review comment the agent proposes."""

    file: str = Field(description="Path of the file the comment is about")
    line: int | None = Field(None, description="Line number, when known")
    severity: Literal["nit", "suggestion", "issue", "blocker"]
    body: str


class PRAnalysis(BaseModel):
    """LLM-structured output of the analyzer node."""

    summary: str = Field(description="One-paragraph description of what the PR does")
    risk_factors: list[str] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Self-reported confidence that the review is complete and correct",
    )
    confidence_reasoning: str = Field(
        description="Why the model picked that confidence value"
    )
    escalation_questions: list[str] = Field(
        default_factory=list,
        description="Specific questions to ask a human reviewer if escalating",
    )


class AuditEntry(BaseModel):
    """One row of the PostgreSQL audit trail.

    Designed as a structured *decision log* — one entry per meaningful event
    in a review session (LLM analysis, HITL interrupt, reviewer response,
    final commit). The fields are first-class SQL columns so auditors can
    query directly (e.g. ``SELECT AVG(confidence) WHERE decision = 'approve'``).
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event was recorded (UTC).",
    )
    agent_id: str = Field(
        description="Identifier of the agent that produced the event "
                    "(e.g. 'pr-review-agent@v0.1')."
    )
    action: str = Field(
        description="What the agent did at this step — "
                    "'fetch_pr' | 'analyze' | 'route' | 'human_approval' | "
                    "'escalate' | 'synthesize' | 'commit'."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Current confidence at this step (mirrors PRAnalysis.confidence).",
    )
    risk_level: str = Field(
        description="Derived from confidence — 'low' / 'med' / 'high'. "
                    "Use risk_level_for(confidence) to compute.",
    )
    reviewer_id: str | None = Field(
        default=None,
        description="GitHub username of the human reviewer for HITL events. "
                    "None for fully automated steps.",
    )
    decision: str = Field(
        description="Outcome at this step — "
                    "'auto' | 'approve' | 'reject' | 'edit' | 'escalate' | "
                    "'pending' | 'committed' | 'failed'.",
    )
    status: str = Field(
        default="pending",
        description="Lifecycle phase of this event — "
                    "'pending' | 'waiting' | 'responded' | 'done' | 'failed'. "
                    "Together with (thread_id, action) it forms the UNIQUE "
                    "idempotency key that keeps re-executed nodes from writing "
                    "duplicate audit rows.",
    )
    reason: str | None = Field(
        default=None,
        description="Free-text explanation: confidence_reasoning for analyze, "
                    "human_feedback for HITL, etc.",
    )
    execution_time_ms: int = Field(
        ge=0,
        description="Wall-clock duration of the action in milliseconds.",
    )


class ReviewState(TypedDict, total=False):
    """LangGraph state — every node reads and updates this dict."""

    # Inputs
    pr_url: str
    thread_id: str

    # Populated by fetch_pr
    pr_title: str
    pr_author: str
    pr_description: str
    pr_diff: str
    pr_files: list[str]
    pr_head_sha: str

    # Populated by analyze
    analysis: PRAnalysis

    # Populated by route_by_confidence
    decision: Decision

    # Populated by HITL nodes
    human_choice: HumanChoice | None
    human_feedback: str | None
    escalation_answers: dict[str, str] | None
    reviewer_id: str | None

    # Populated by final_approval (post-escalation confirmation)
    refine_confirmed: bool | None

    # Populated by commit / final nodes
    posted_comment_body: str | None
    final_action: str | None

    # Error channels — used by the failure nodes (nothing is posted when set)
    fetch_error: str | None
    analysis_error: str | None
    post_error: str | None
    refined: bool | None
