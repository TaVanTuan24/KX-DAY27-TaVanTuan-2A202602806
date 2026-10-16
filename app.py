"""Exercise 5 — Streamlit approval UI for the HITL PR-review agent.

Run with:
    uv run streamlit run app.py

The UI drives the *same* graph as exercises 1–4
(``exercises.agent.build_graph`` with ``Mode.EXERCISE_5``): an ``AsyncSqliteSaver``
checkpointer so a paused session survives a page reload, a Streamlit restart, or
a process kill, and the structured audit trail in the same SQLite file.

Routing buckets (``common/schemas.py``):

* ``confidence >= 0.73`` and no security/blocker finding → ``auto_approve``:
  a success card; the agent already posted the comment.
* ``0.58 <= confidence < 0.73`` (or a security/blocker finding) →
  ``human_approval``: diff + reasoning + comments + Approve / Reject / Edit.
* ``confidence < 0.58`` → ``escalate``: risk factors + a question form, then a
  refined review and a mandatory final confirmation before anything is posted.

Async integration: every graph call opens *its own* ``AsyncSqliteSaver``
connection inside a *fresh* event loop (``asyncio.run``), then closes it. That
avoids the two classic Streamlit failure modes — a checkpointer bound to a dead
event loop, and a second writer holding SQLite's lock — without leaking
connections across reruns.

This file is importable: ``streamlit run`` executes it top-to-bottom, but the
test suite imports it directly and drives the render helpers with a stubbed
``streamlit`` module instead of a server.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from common.db import db_path, open_checkpointer
from common.db import list_threads as list_threads_data
from common.db import replay_events
from common.github import GitHubError, is_valid_pr_url
from common.schemas import AUTO_APPROVE_THRESHOLD, ESCALATE_THRESHOLD
from exercises.agent import Mode, build_comment_body, build_graph, config_for

load_dotenv()

PAGE_TITLE = "HITL PR Review"
DEFAULT_PR_URL = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"

STATUS_LABEL = {
    "fetch_pr": "fetching PR",
    "analyze": "analyzing diff",
    "route": "routing",
    "human_approval": "waiting for approval",
    "escalate": "waiting for reviewer answers",
    "synthesize": "refining review",
    "final_approval": "waiting for final confirmation",
    "commit": "committing",
    "auto_approve": "auto-approved",
}


# ─── async bridge ──────────────────────────────────────────────────────────


async def run_graph(
    pr_url: str,
    thread_id: str,
    resume_value: Any = None,
    *,
    dry_run: bool | None = None,
    reviewer_id: str | None = None,
) -> dict:
    """Invoke the graph once; return the result (possibly holding ``__interrupt__``).

    A fresh checkpointer connection and event loop per call — see module docstring.

    ``dry_run`` is only used when *starting* a session: it is written into the graph
    state, which is then the single source of truth. The graph is always built with
    ``dry_run=None`` so a resume cannot flip the mode, and a checkpoint that never
    recorded one fails safe (no POST). See ``Agent.effective_dry_run``.

    ``reviewer_id`` is likewise injected on the start call so HITL audit rows can
    attribute the decision to a GitHub username.
    """
    async with open_checkpointer() as cp:
        app = build_graph(mode=Mode.EXERCISE_5, checkpointer=cp, dry_run=None)
        cfg = config_for(thread_id)
        if resume_value is None:
            seed: dict[str, Any] = {
                "pr_url": pr_url,
                "thread_id": thread_id,
                "dry_run": bool(dry_run),
            }
            if reviewer_id:
                seed["reviewer_id"] = reviewer_id
            return await app.ainvoke(seed, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


class SessionView:
    """Read-only view of a thread reconstructed from its checkpoint.

    Exactly one of ``pending`` / ``values`` is meaningful:

    * ``pending`` — the graph is paused on an ``interrupt()``; ``values`` also holds
      the state so far, and the card can be re-rendered to continue.
    * ``values`` — the graph finished; this is its final state, rendered as history.
    * neither — there is no checkpoint (e.g. it was deleted); the audit trail may
      still describe the session.
    """

    def __init__(
        self,
        *,
        pending: dict | None = None,
        values: dict | None = None,
        dry_run: bool | None = None,
        revision: int | None = None,
    ) -> None:
        self.pending = pending
        self.values = values
        self.dry_run = dry_run
        self.revision = revision

    @property
    def has_checkpoint(self) -> bool:
        return bool(self.pending) or bool(self.values)

    @property
    def is_pending(self) -> bool:
        return self.pending is not None

    @property
    def is_finished(self) -> bool:
        return self.pending is None and bool(self.values)


async def load_session(thread_id: str) -> SessionView:
    """Reconstruct a session from its checkpoint — read-only, runs nothing.

    Used both to recover a paused review and (FIX 6) to re-display a *finished*
    one: opening either must not re-run the LLM, must not POST, and must not
    modify the checkpoint.
    """
    async with open_checkpointer() as cp:
        app = build_graph(mode=Mode.EXERCISE_5, checkpointer=cp)
        snapshot = await app.aget_state(config_for(thread_id))
        values = dict(snapshot.values or {})
        pending: dict | None = None
        for task in snapshot.tasks or ():
            for intr in task.interrupts or ():
                pending = getattr(intr, "value", intr)
                break
            if pending is not None:
                break
        if not snapshot.next:
            pending = None
        return SessionView(
            pending=pending,
            values=values,
            dry_run=(values.get("dry_run") if "dry_run" in values else None),
            revision=values.get("edit_revision"),
        )


def call_async(coro):
    """Run a coroutine from Streamlit's thread (no running loop there)."""
    return asyncio.run(coro)


def interrupt_payload(result: dict) -> dict | None:
    """Extract the interrupt payload from an ``ainvoke`` result, if present."""
    if "__interrupt__" not in result:
        return None
    first = result["__interrupt__"][0]
    return getattr(first, "value", first)


# ─── session state ─────────────────────────────────────────────────────────


def init_state() -> None:
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("pr_url", "")
    st.session_state.setdefault("interrupt_payload", None)
    st.session_state.setdefault("final", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("reviewer_id", "")
    st.session_state.setdefault("dry_run", False)
    #: The mode the *current* session was created with (None = no session yet).
    st.session_state.setdefault("session_dry_run", None)
    st.session_state.setdefault("session_finished", False)
    st.session_state.setdefault("history_only", False)
    st.session_state.setdefault("edit_mode", False)


def reset_for(pr_url: str) -> None:
    """Start a brand-new review thread (new thread_id, cleared UI state).

    This is the ONLY place the dry-run of a session is chosen: it is persisted
    into the graph state on the first invoke and locked afterwards.
    """
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.session_state.error = None
    st.session_state.edit_mode = False
    st.session_state.session_finished = False
    st.session_state.history_only = False
    st.session_state.session_dry_run = bool(st.session_state.dry_run)


def open_session(view: "SessionView", thread_id: str, pr_url: str) -> None:
    """Adopt an existing thread in the UI (never writes to its checkpoint)."""
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = pr_url
    st.session_state.error = None
    st.session_state.edit_mode = False
    st.session_state.history_only = not view.has_checkpoint
    st.session_state.session_finished = view.is_finished
    if view.dry_run is not None:
        # Restore the session's original mode; the toggle shows (and locks) it.
        st.session_state.session_dry_run = bool(view.dry_run)
        st.session_state.dry_run = bool(view.dry_run)
    if view.is_pending:
        st.session_state.interrupt_payload = view.pending
        st.session_state.final = None
    elif view.is_finished:
        st.session_state.interrupt_payload = None
        st.session_state.final = view.values
    else:
        # No checkpoint left: show what the audit trail knows, with a warning.
        st.session_state.interrupt_payload = None
        st.session_state.final = None


def apply_result(result: dict) -> None:
    """Fold one graph result into session state."""
    st.session_state.error = result.get("post_error")
    payload = interrupt_payload(result)
    if payload is not None:
        st.session_state.interrupt_payload = payload
        st.session_state.final = None
        st.session_state.session_finished = False
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result
        st.session_state.session_finished = True
    st.session_state.edit_mode = False


# ─── renderers (pure: display + return the resume value) ───────────────────


def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: diff + reasoning + comments + 3 buttons."""
    conf = float(payload.get("confidence", 0.0))
    blocked = bool(payload.get("risk_factors")) or any(
        c.get("severity") == "blocker" for c in payload.get("comments") or []
    )
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    if blocked:
        st.error(
            "This PR carries a security or blocker finding, so it needs a human "
            "even though the model is confident."
        )
        for rf in payload.get("risk_factors") or []:
            st.markdown(f"- ⚠️ {rf}")
    st.caption(payload.get("confidence_reasoning", ""))
    st.markdown(payload.get("summary", ""))

    comments = payload.get("comments") or []
    if comments:
        st.markdown("**Proposed review comments**")
        for c in comments:
            location = f"`{c['file']}:{c.get('line') or '?'}`"
            st.markdown(f"- **[{c['severity']}]** {location} — {c['body']}")

    with st.expander("Diff (truncated)", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    st.markdown("---")
    feedback = st.text_area(
        "Feedback / edit instructions (optional)",
        key="approval_feedback",
        height=80,
        placeholder="For Reject: why not. For Edit: what to change in the review.",
    )
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("⛔ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit", use_container_width=True):
        # Open the edit form instead of resuming straight away: an Edit needs
        # instructions (or a replacement review) before it is actionable.
        st.session_state.edit_mode = True
        st.rerun()

    if st.session_state.get("edit_mode"):
        st.markdown("### ✏️ Edit this review")
        st.caption(
            "Describe what to change and the agent revises the review, or paste the "
            "review you want posted — replacement text is used verbatim and is never "
            "rewritten by the model. Blocker findings are kept either way."
        )
        instructions = st.text_area(
            "Edit instructions (applied by the agent)",
            key="edit_instructions_input",
            height=80,
            placeholder="e.g. drop the ordering nit, keep the migration blocker",
        )
        replacement = st.text_area(
            "Replacement review text (optional — posted verbatim, overwrites the above)",
            key="edit_body_input",
            height=160,
            placeholder="Leave empty to let the agent revise it from your instructions.",
        )
        c1, c2 = st.columns(2)
        if c1.button("▶️ Apply edit", type="primary", use_container_width=True):
            if not instructions.strip() and not replacement.strip():
                st.warning("Enter instructions or paste a replacement review.")
            else:
                return {
                    "choice": "edit",
                    "feedback": instructions,
                    "edited_body": replacement.strip() or None,
                }
        if c2.button("Cancel edit", use_container_width=True):
            st.session_state.edit_mode = False
            st.rerun()
    return None


def render_edited_review_card(payload: dict) -> dict | None:
    """Post-Edit gate: preview the revised review, then confirm / edit again / reject.

    Nothing has been posted at this point; the reviewer must confirm.
    """
    revision = payload.get("revision") or 1
    conf = float(payload.get("confidence", 0.0))
    st.subheader(f"✏️ Edited review — revision {revision} (confidence {conf:.0%})")
    st.caption(payload.get("confidence_reasoning", ""))
    if payload.get("risk_factors"):
        for rf in payload["risk_factors"]:
            st.markdown(f"- ⚠️ {rf}")
    st.markdown(payload.get("summary", ""))
    for c in payload.get("comments") or []:
        st.markdown(
            f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}"
        )

    st.markdown("**Preview of the comment that would be posted**")
    st.code(payload.get("proposed_body") or "(empty)", language="markdown")
    st.info("Nothing has been posted to GitHub yet.")

    col1, col2, col3 = st.columns(3)
    if col1.button("📮 Confirm & post", type="primary", use_container_width=True):
        return {"confirm": True, "feedback": st.session_state.get("edit_review_feedback", "")}
    if col2.button("✏️ Edit again", use_container_width=True):
        st.session_state.edit_mode = True
        st.rerun()
    if col3.button("⛔ Cancel / reject", use_container_width=True):
        return {"confirm": False, "feedback": st.session_state.get("edit_review_feedback", "")}

    if st.session_state.get("edit_mode"):
        st.markdown("### ✏️ Another revision")
        instructions = st.text_area(
            "Further edit instructions", key="edit_again_instructions", height=80
        )
        replacement = st.text_area(
            "Replacement review text (optional — posted verbatim)",
            key="edit_again_body",
            height=140,
        )
        c1, c2 = st.columns(2)
        if c1.button("▶️ Apply another revision", type="primary", use_container_width=True):
            if not instructions.strip() and not replacement.strip():
                st.warning("Enter instructions or paste a replacement review.")
            else:
                return {
                    "choice": "edit",
                    "feedback": instructions,
                    "edited_body": replacement.strip() or None,
                }
        if c2.button("Back", use_container_width=True):
            st.session_state.edit_mode = False
            st.rerun()
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """<58% bucket: risk factors + one answer box per question."""
    conf = float(payload.get("confidence", 0.0))
    st.subheader(f"⚠️ Strong escalation — confidence {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    risks = payload.get("risk_factors") or []
    if risks:
        st.error("Risks the model could not resolve on its own:")
        for rf in risks:
            st.markdown(f"- 🔴 {rf}")
    st.markdown(payload.get("summary", ""))
    with st.expander("Diff (truncated)", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    questions = payload.get("questions") or ["What is the intent of this PR?"]
    st.markdown("**Answer these before the agent re-reviews the PR**")
    with st.form("escalation_form"):
        answers: dict[str, str] = {}
        for i, question in enumerate(questions):
            st.markdown(f"**Q{i + 1}.** {question}")
            answers[question] = st.text_area(
                f"Answer {i + 1}", key=f"escalation_answer_{i}", height=70
            )
        submitted = st.form_submit_button("Submit answers", type="primary")
    if submitted:
        return answers
    return None


def render_final_confirmation_card(payload: dict) -> dict | None:
    """Post-escalation gate: refined review + confirm before posting."""
    conf = float(payload.get("confidence", 0.0))
    st.subheader(f"Refined review ready — confidence {conf:.0%}")
    st.caption(payload.get("confidence_reasoning", ""))
    st.markdown(payload.get("summary", ""))
    for c in payload.get("comments") or []:
        st.markdown(
            f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}"
        )
    answers = payload.get("escalation_answers") or {}
    if answers:
        with st.expander("Your answers", expanded=False):
            for q, a in answers.items():
                st.markdown(f"- **{q}**  \n  {a or '_(no answer)_'}")
    st.info("Nothing has been posted to GitHub yet.")
    feedback = st.text_input("Note for the posted comment (optional)", key="final_feedback")
    col1, col2 = st.columns(2)
    if col1.button("📮 Post this comment", type="primary", use_container_width=True):
        return {"confirm": True, "feedback": feedback}
    if col2.button("🚫 Do not post", use_container_width=True):
        return {"confirm": False, "feedback": feedback}
    return None


def render_interrupt(payload: dict) -> dict | None:
    """Dispatch on ``payload["kind"]``."""
    kind = payload.get("kind")
    if kind == "approval_request":
        return render_approval_card(payload)
    if kind == "escalation":
        return render_escalation_card(payload)
    if kind == "final_confirmation":
        return render_final_confirmation_card(payload)
    if kind == "edited_review":
        return render_edited_review_card(payload)
    st.error(f"Unknown interrupt kind: {kind!r}")
    return None


def render_final(final: dict, pr_url: str, thread_id: str) -> None:
    """Terminal state card."""
    action = final.get("final_action", "?")
    analysis = final.get("analysis")
    if analysis is not None:
        conf = analysis.confidence
        st.metric("Confidence", f"{conf:.0%}")
    if action in {"committed", "auto_approved"}:
        st.success(f"Comment posted to GitHub — {action}")
        st.markdown(f"[View the pull request on GitHub]({pr_url})")
    elif action == "committed_already":
        st.success(
            "The review was already on the PR from an earlier attempt — it was "
            "**not** posted a second time."
        )
        st.markdown(f"[View the pull request on GitHub]({pr_url})")
    elif action in {"committed_dry_run", "auto_approved_dry_run"}:
        st.warning("Dry-run: the review is ready but was NOT posted to GitHub.")
        with st.expander("Comment body that would be posted", expanded=False):
            st.markdown(final.get("posted_comment_body") or "(empty)")
    elif action == "commit_unconfirmed":
        st.error(
            "GitHub accepted the request but the outcome could **not** be confirmed — "
            "the comment may or may not exist, and it was deliberately NOT retried. "
            f"Check the PR before doing anything else. ({final.get('post_error')})"
        )
    elif action == "rejected":
        st.warning("Rejected — nothing was posted to GitHub.")
    elif action in {"commit_failed", "auto_approve_failed"}:
        st.error(f"GitHub rejected the comment: {final.get('post_error')}")
    elif action == "auto_refused_security":
        st.error(
            "Refused to auto-post: a security or blocker finding is present, so this "
            "needs a human."
        )
    elif action in {"fetch_failed", "analyze_failed"}:
        st.error(f"The run stopped early ({action}). Nothing was posted.")
    else:
        st.info(f"final_action = {action}")
    if final.get("audit_error"):
        st.warning(
            "The audit trail is incomplete for this run "
            f"({final['audit_error']}) — automation should not treat it as verified."
        )
    if analysis is not None:
        st.markdown("**Summary**")
        st.markdown(analysis.summary)
        for c in analysis.comments:
            st.markdown(
                f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}"
            )
    st.caption(
        f"thread_id = `{thread_id}` · replay: "
        f"`uv run python -m audit.replay --thread {thread_id}`"
    )


def render_events(thread_id: str) -> None:
    """Audit timeline for the current thread."""
    try:
        events = asyncio.run(replay_events(thread_id))
    except Exception as exc:  # pragma: no cover - defensive
        st.caption(f"(audit trail unavailable: {exc})")
        return
    if not events:
        st.caption("(no audit events yet)")
        return
    with st.expander(f"Audit trail — {len(events)} event(s)", expanded=False):
        st.dataframe(
            [
                {
                    "time": e["timestamp"][11:19],
                    "action": e["action"],
                    "conf": f"{e['confidence']:.2f}",
                    "risk": e["risk_level"],
                    "decision": e["decision"],
                    "status": e["status"],
                    "rev": e.get("revision", 0),
                    "reviewer": e["reviewer_id"] or "-",
                    "ms": e["execution_time_ms"],
                }
                for e in events
            ],
            use_container_width=True,
            hide_index=True,
        )


def _sidebar_recent_sessions() -> None:
    """Recent sessions from ``audit_events`` + session recovery."""
    st.header("Recent sessions")
    try:
        rows = asyncio.run(list_threads_data(25))
    except Exception as exc:  # pragma: no cover - defensive
        st.caption(f"(audit trail unavailable: {exc})")
        rows = []

    if not rows:
        st.caption("No sessions yet.")
        return

    for row in rows:
        tid = row["thread_id"]
        label = STATUS_LABEL.get(row["last_action"] or "", row["last_action"] or "?")
        risk = row["worst_risk"] or "?"
        marker = {"high": "🔴", "med": "🟡", "low": "🟢"}.get(risk, "⚪")
        mode = (
            " · dry-run"
            if row["last_decision"] == "dry_run"
            else " · posted"
            if row["last_decision"] in {"committed", "auto"}
            else ""
        )
        is_current = tid == st.session_state.get("thread_id")
        header = (
            f"{marker} **{'→ ' if is_current else ''}{tid[:8]}** · {label} · {risk}{mode}"
        )
        with st.expander(header, expanded=is_current):
            st.caption(f"PR: {row['pr_url']}")
            st.caption(
                f"events: {row['events']} · last: {row['last_event']} · "
                f"decision: {row['last_decision']}"
            )
            if st.button("Open this session", key=f"open_{tid}"):
                # Read-only: rebuild the view from the checkpoint. Opening a session
                # never runs the graph, never posts, and never touches the
                # checkpoint — whether it is paused or already finished (FIX 6).
                try:
                    view = asyncio.run(load_session(tid))
                    open_session(view, tid, row["pr_url"])
                except Exception as exc:  # pragma: no cover - defensive
                    open_session(SessionView(), tid, row["pr_url"])
                    st.session_state.error = f"Could not read the checkpoint: {exc}"
                st.rerun()


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🔍", layout="wide")
    init_state()

    with st.sidebar:
        st.title("🔍 HITL PR Review")
        st.session_state.reviewer_id = st.text_input(
            "Your GitHub username (audit trail)",
            value=st.session_state.reviewer_id,
            placeholder="tavan",
        )
        # FIX 2: the dry-run mode belongs to the SESSION, not to the widget. It is
        # chosen when a review starts, persisted in the graph state, and locked
        # afterwards so flipping the toggle mid-session cannot turn a rehearsal
        # into a real comment.
        session_active = bool(st.session_state.thread_id)
        st.session_state.dry_run = st.toggle(
            "Dry-run (never POST to GitHub)",
            value=(
                st.session_state.session_dry_run
                if session_active and st.session_state.session_dry_run is not None
                else st.session_state.dry_run
            ),
            disabled=session_active,
            help=(
                "Locked for this session: the mode is fixed when a review starts so a "
                "resume cannot turn a dry-run into a real post. Start a new review to "
                "change it."
                if session_active
                else "Runs the whole graph and writes the audit trail, but skips the "
                "GitHub comment call."
            ),
        )
        if session_active:
            original = "dry-run" if st.session_state.session_dry_run else "live (may POST)"
            st.caption(f"Session mode: **{original}** (locked)")
        st.divider()
        _sidebar_recent_sessions()

    st.title("HITL PR Review Agent")
    st.caption(
        f"auto-approve at ≥ {AUTO_APPROVE_THRESHOLD:.0%} · human approval "
        f"{ESCALATE_THRESHOLD:.0%}–{AUTO_APPROVE_THRESHOLD:.0%} · escalate below "
        f"{ESCALATE_THRESHOLD:.0%}"
    )

    with st.form("start"):
        pr_url = st.text_input(
            "GitHub Pull Request URL",
            value=st.session_state.pr_url or DEFAULT_PR_URL,
            placeholder=DEFAULT_PR_URL,
        )
        submitted = st.form_submit_button("▶️ Run review", type="primary")

    if submitted:
        if not is_valid_pr_url(pr_url):
            st.error(
                "That is not a GitHub PR URL. Expected a link like "
                f"{DEFAULT_PR_URL}"
            )
        else:
            reset_for(pr_url)
            with st.spinner("Fetching the PR and asking the LLM…"):
                try:
                    result = call_async(
                        run_graph(
                            pr_url,
                            st.session_state.thread_id,
                            None,
                            dry_run=st.session_state.dry_run,
                            reviewer_id=st.session_state.reviewer_id or None,
                        )
                    )
                except GitHubError as exc:
                    st.session_state.error = exc.friendly
                except Exception as exc:
                    st.session_state.error = f"{type(exc).__name__}: {exc}"
                else:
                    apply_result(result)
            if st.session_state.error:
                st.error(st.session_state.error)

    payload = st.session_state.get("interrupt_payload")
    if payload is not None:
        st.divider()
        answer = render_interrupt(payload)
        if answer is not None:
            kind = payload.get("kind")
            if kind == "escalation" and not any(answer.values()):
                st.warning("Please answer at least one question before submitting.")
            elif kind == "approval_request" and answer.get("choice") == "edit" and not (
                (answer.get("feedback") or "").strip()
                or (answer.get("edited_body") or "").strip()
            ):
                st.warning("Edit selected — describe what to change, or choose Approve.")
            else:
                with st.spinner("Resuming the graph…"):
                    try:
                        # No `dry_run` here on purpose: the mode was persisted when
                        # the session started and the state is authoritative.
                        result = call_async(
                            run_graph(
                                st.session_state.pr_url,
                                st.session_state.thread_id,
                                answer,
                            )
                        )
                        apply_result(result)
                    except GitHubError as exc:
                        st.session_state.error = exc.friendly
                    except Exception as exc:
                        st.session_state.error = f"{type(exc).__name__}: {exc}"
                if st.session_state.error:
                    st.error(st.session_state.error)
                st.rerun()

    final = st.session_state.get("final")
    if final is not None:
        st.divider()
        thread_id = st.session_state.thread_id or ""
        if st.session_state.get("session_finished"):
            st.caption(
                "Session history (read-only — opening a session never re-runs the "
                "agent or posts anything)."
            )
        if st.session_state.get("history_only"):
            st.warning(
                "No checkpoint is available for this thread any more, so only the "
                "audit trail below can be shown."
            )
        render_final(final, st.session_state.pr_url, thread_id)
        if thread_id:
            render_events(thread_id)
    elif st.session_state.get("history_only") and st.session_state.get("thread_id"):
        st.divider()
        st.warning(
            "No checkpoint is available for this thread any more, so only the audit "
            "trail can be shown."
        )
        render_events(st.session_state.thread_id)

    if st.session_state.get("error") and final is None:
        st.error(st.session_state.error)


def _entrypoint() -> None:
    """Run the Streamlit app, but only when executed by the Streamlit server.

    Importing this module (for tests, or from another script) must not start a
    UI, so the guard checks Streamlit's runtime rather than ``__name__``.
    """
    try:
        running = st.runtime.exists()
    except Exception:  # pragma: no cover - older/newer Streamlit
        running = False
    if running:  # pragma: no cover - exercised only inside `streamlit run`
        main()


_entrypoint()


if __name__ == "__main__":  # `python app.py` — tell the user what to run instead
    print("This is a Streamlit app. Run it with:\n\n    uv run streamlit run app.py\n")
