"""Shared types: the graph state, the LLM's structured analysis output,
and the structured audit-trail entry written to the ``audit_events`` table."""

from __future__ import annotations

import re
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
#: ``dry_run`` exists so a rehearsal never claims a comment was posted.
AuditDecision = Literal[
    "auto", "approve", "reject", "edit", "escalate", "pending",
    "committed", "failed", "dry_run",
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
    """Cheap, conservative heuristic for "this risk factor is security-relevant".

    Used only as a *secondary* signal: the model's own ``blocker`` severity is the
    primary one (the prompts tell it to use ``blocker`` for anything that must not
    ship). This exists so a security risk factor that was not linked to a blocker
    comment still forces a human.

    Calibration matters — a false positive here mislabels an audit row as
    ``risk_level='high'`` and blocks auto-approval for a safe PR:

    * patterns are **word-boundary** anchored, so ``sqlite3`` does not match
      ``sql`` and ``author`` does not match ``auth``;
    * a match inside a negated clause ("… so there is no injection surface",
      "SQL is fully parameterized") is ignored.
    """
    for pattern in SECURITY_PATTERNS:
        for match in pattern.finditer(text):
            before = _clause_before(text, match.start())
            after = _clause_after(text, match.end())
            if not (
                _NEGATION_RE.search(before) or _TRAILING_NEGATION_RE.search(after)
            ):
                return True
    return False


#: Security-relevant phrases. Deliberately specific: bare ``sql``/``hash``/``auth``
#: match database engines, `hashmap` and `author` respectively.
SECURITY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\binjection\b",
        r"\bsql\s+injection\b",
        r"\braw\s+sql\b",
        r"\bstring[\s-]concatenat",
        r"\bunsaniti[sz]",
        r"\bsaniti[sz]",
        r"\bxss\b",
        r"\bcsrf\b",
        r"\bssrf\b",
        r"\bpath\s+traversal\b",
        r"\bauth\b",
        r"\bauthenticat",
        r"\bauthoriz",
        r"\bpassword\b",
        r"\bcredential",
        r"\bsecret\b",
        r"\bapi[\s_-]?key\b",
        r"\btoken\b",
        r"\bmd5\b",
        r"\bsha[\s-]?1\b",
        r"\bbcrypt\b",
        r"\bargon2\b",
        r"\bplain\s?text\b",
        r"\bplain\s?http\b",
        r"\bhttp://",
        r"\btls\b",
        r"\bssl\b",
        r"\bpermission",
        r"\bprivilege",
        r"\bescalat",
        r"\bvulnerab",
        r"\bexploit",
        r"\bcve-\d{4}",
        r"\bpii\b",
        r"\bleak",
        r"\bexfiltrat",
        r"\bcrypto",
        r"\bweak\s+hash",
    )
)

#: Cues that the surrounding clause *denies* the problem ("no injection",
#: "fully parameterized", "properly escaped").
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|none|nothing|without|isn'?t|aren'?t|doesn'?t|don'?t|didn'?t|"
    r"cannot|can'?t|won'?t|neither|nor|parameteri[sz]ed|escaped|safely|"
    r"saniti[sz]ed)\b",
    re.IGNORECASE,
)

#: Negation that trails the keyword and binds to it ("plain HTTP **is not**
#: reachable"). Deliberately much narrower than :data:`_NEGATION_RE`: a bare "no"
#: in the following clause usually belongs to a *different* phrase
#: ("Weak hash algorithm with no salting" is a real finding), so only a
#: copula/auxiliary immediately followed by a negative counts here.
_TRAILING_NEGATION_RE = re.compile(
    r"^\s*(?:(?:is|are|was|were|has|have|had|will|would|can|cannot|must|should|"
    r"does|do|did)\s+(?:not|never|no\s+longer)\b"
    r"|isn'?t|aren'?t|wasn'?t|weren'?t|doesn'?t|don'?t|didn'?t|never|no\s+longer)\b",
    re.IGNORECASE,
)

#: Clause boundaries used to scope a negation cue to the match it belongs to.
_CLAUSE_SPLIT_RE = re.compile(r"[.;:,!?]\s+|\n")


def _clause_before(text: str, index: int, limit: int = 90) -> str:
    """The clause immediately preceding ``index`` (so negation stays local)."""
    window = text[max(0, index - limit) : index]
    segments = _CLAUSE_SPLIT_RE.split(window)
    return segments[-1] if segments else window


def _clause_after(text: str, index: int, limit: int = 60) -> str:
    """The clause immediately following ``index``.

    Negation can trail the keyword ("plain HTTP **is not** reachable"), so both
    directions are checked — each bounded to its own clause, which keeps an
    unrelated "no" elsewhere in the sentence from suppressing a real finding.
    """
    window = text[index : index + limit]
    segments = _CLAUSE_SPLIT_RE.split(window, maxsplit=1)
    return segments[0] if segments else window


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


def confidence_only_route(analysis: "PRAnalysis") -> Decision:
    """Routing from confidence alone — i.e. without the security override.

    Used to report *why* a PR was pulled down: comparing this with
    :func:`route_for` shows whether a security finding (rather than confidence)
    is what forced a human. The override can only ever move
    ``auto_approve`` → ``human_approval``; ``escalate`` always wins.
    """
    if analysis.confidence < ESCALATE_THRESHOLD:
        return "escalate"
    if analysis.confidence >= AUTO_APPROVE_THRESHOLD:
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
