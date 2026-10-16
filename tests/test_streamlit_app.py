"""Exercise 5 — Streamlit approval UI.

`streamlit run` executes ``app.py`` top-to-bottom, which makes the real module
untestable from pytest. To test the *decision logic* — which card renders for
which ``kind``, what each button returns as the resume value, how a result is
folded into session state, and how the terminal card is worded — the tests
install a recording stub for the ``streamlit`` module before importing ``app``.

The graph itself is exercised in ``test_end_to_end.py``; here the subject is the
UI layer, including the required Approve / Reject / Edit and escalation paths.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest
from common.db import memory_checkpointer
from langgraph.types import Command

from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis


# ─── the stub ──────────────────────────────────────────────────────────────


class StubElement:
    """Return value of ``st.columns(n)`` — delegates widgets back to the stub."""

    def __init__(self, stub: "StubStreamlit") -> None:
        self._stub = stub

    def button(self, label: str, **kwargs: Any) -> bool:
        return self._stub.button(label, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._stub, name)


class SessionState(dict):
    """``st.session_state`` supports both mapping and attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors Streamlit
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def setdefault(self, name: str, default: Any = None) -> Any:  # type: ignore[override]
        return dict.setdefault(self, name, default)


class StubStreamlit(types.ModuleType):
    """Records every call and lets a test decide what each widget returns."""

    def __init__(self) -> None:
        super().__init__("streamlit")
        self.calls: list[tuple[str, tuple]] = []
        self.pressed: set[str] = set()
        self.submitted: set[str] = set()
        self.values: dict[str, Any] = {}
        self.session_state = SessionState()
        self.runtime = types.SimpleNamespace(exists=lambda: False)

    # internals ------------------------------------------------------------
    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def called(self, name: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == name]

    def labels(self, name: str) -> list[str]:
        return [c[1][0] for c in self.calls if c[0] == name and c[1]]

    def _value_for(self, label: str, key: str | None, default: Any) -> Any:
        for candidate in (key, label):
            if candidate and candidate in self.values:
                return self.values[candidate]
        return default

    # widgets --------------------------------------------------------------
    def button(self, label: str, **kwargs: Any) -> bool:
        self._record("button", label, **kwargs)
        return label in self.pressed

    def form_submit_button(self, label: str, **kwargs: Any) -> bool:
        self._record("form_submit_button", label, **kwargs)
        return label in self.submitted

    def text_input(self, label: str, value: Any = "", **kwargs: Any) -> Any:
        self._record("text_input", label, **kwargs)
        return self._value_for(label, kwargs.get("key"), kwargs.get("value", value))

    def text_area(self, label: str, value: Any = "", **kwargs: Any) -> Any:
        self._record("text_area", label, **kwargs)
        return self._value_for(label, kwargs.get("key"), kwargs.get("value", value))

    def toggle(self, label: str, value: bool = False, **kwargs: Any) -> bool:
        self._record("toggle", label, **kwargs)
        return bool(self._value_for(label, kwargs.get("key"), value))

    def columns(self, spec: Any, **kwargs: Any):
        self._record("columns", spec, **kwargs)
        count = spec if isinstance(spec, int) else len(spec)
        return [StubElement(self) for _ in range(count)]

    def form(self, key: str, **kwargs: Any):
        self._record("form", key, **kwargs)
        stub = self

        class _Form:
            def __enter__(self):
                return stub

            def __exit__(self, *exc: Any) -> bool:
                return False

        return _Form()

    def expander(self, label: str, **kwargs: Any):
        return self.form(label, **kwargs)

    def __getattr__(self, name: str):
        """Any other Streamlit API is a recording no-op."""
        if name.startswith("_"):
            raise AttributeError(name)

        def _noop(*args: Any, **kwargs: Any) -> None:
            self._record(name, *args, **kwargs)

        return _noop


@pytest.fixture
def app_stub(monkeypatch: pytest.MonkeyPatch):
    """Import ``app`` with a stubbed ``streamlit`` module."""
    stub = StubStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    try:
        yield module, stub
    finally:
        sys.modules.pop("app", None)
        importlib.import_module("streamlit")  # restore the real module


# ─── helpers ───────────────────────────────────────────────────────────────


APPROVAL_PAYLOAD = {
    "kind": "approval_request",
    "pr_url": PR_URL,
    "thread_id": "t-ui",
    "confidence": 0.65,
    "confidence_reasoning": "One open schema question.",
    "summary": "Adds a priority column.",
    "risk_factors": [],
    "comments": [
        {"file": "app/auth.py", "line": 12, "severity": "suggestion", "body": "add a test"}
    ],
    "diff_preview": "+priority = Column(int)",
}

ESCALATION_PAYLOAD = {
    "kind": "escalation",
    "pr_url": PR_URL,
    "thread_id": "t-ui",
    "confidence": 0.31,
    "confidence_reasoning": "Multiple security red flags.",
    "summary": "Adds login and cloud sync.",
    "risk_factors": ["MD5 password hashing", "plaintext sync token"],
    "questions": ["app/auth.py:12 — why MD5?", "sync/store.py:40 — is SYNC_URL HTTPS?"],
    "diff_preview": "+hashlib.md5(...)",
}

FINAL_PAYLOAD = {
    "kind": "final_confirmation",
    "pr_url": PR_URL,
    "thread_id": "t-ui",
    "confidence": 0.62,
    "confidence_reasoning": "TLS migration still unresolved.",
    "summary": "Adds login and cloud sync.",
    "risk_factors": ["still HTTP"],
    "comments": [
        {"file": "app/auth.py", "line": 12, "severity": "blocker", "body": "MD5 hashing"}
    ],
    "escalation_answers": {"app/auth.py:12 — why MD5?": "legacy"},
    "diff_preview": "+hashlib.md5(...)",
}


# ─── tests: helpers and state ──────────────────────────────────────────────


def test_interrupt_payload_extracts_from_a_real_graph_result(app_stub) -> None:
    module, _ = app_stub
    from exercises.agent import Mode, build_graph, config_for

    app = build_graph(
        mode=Mode.EXERCISE_5,
        checkpointer=memory_checkpointer(),
        llm_factory=FakeLLMFactory(make_analysis(0.65)),
        github=FakeGitHub(),
    )
    import asyncio

    result = asyncio.run(app.ainvoke({"pr_url": PR_URL, "thread_id": "t"}, config_for("t")))
    payload = module.interrupt_payload(result)
    assert payload["kind"] == "approval_request"
    assert module.interrupt_payload({"final_action": "committed"}) is None


def test_apply_result_stores_a_pause_then_a_final_state(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    module.apply_result({"__interrupt__": [types.SimpleNamespace(value=APPROVAL_PAYLOAD)]})
    assert stub.session_state["interrupt_payload"]["kind"] == "approval_request"
    assert stub.session_state["final"] is None

    module.apply_result({"final_action": "committed"})
    assert stub.session_state["interrupt_payload"] is None
    assert stub.session_state["final"]["final_action"] == "committed"


def test_reset_for_creates_a_fresh_thread(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.session_state["interrupt_payload"] = APPROVAL_PAYLOAD
    module.reset_for(PR_URL)
    assert stub.session_state["pr_url"] == PR_URL
    assert stub.session_state["thread_id"]
    assert stub.session_state["interrupt_payload"] is None


def test_unknown_pr_url_is_rejected_before_any_network_call(app_stub) -> None:
    module, _ = app_stub
    from common.github import is_valid_pr_url

    assert is_valid_pr_url("https://example.com/not/a/pr") is False
    assert is_valid_pr_url(PR_URL) is True


# ─── tests: approval card (Approve / Reject / Edit) ────────────────────────


def test_approval_card_renders_reasoning_comments_and_diff(app_stub) -> None:
    module, stub = app_stub
    assert module.render_approval_card(dict(APPROVAL_PAYLOAD)) is None  # nothing clicked
    assert any("One open schema question." in str(c) for c in stub.calls)
    assert stub.labels("button") == ["✅ Approve", "⛔ Reject", "✏️ Edit"]


@pytest.mark.parametrize(
    ("button", "expected_choice"),
    [
        ("✅ Approve", "approve"),
        ("⛔ Reject", "reject"),
    ],
)
def test_approval_card_returns_the_clicked_choice(app_stub, button: str, expected_choice: str) -> None:
    module, stub = app_stub
    stub.pressed = {button}
    stub.values["approval_feedback"] = "my note"
    answer = module.render_approval_card(dict(APPROVAL_PAYLOAD))
    assert answer == {"choice": expected_choice, "feedback": "my note"}


# ─── FIX 1: Edit opens a form, it does not resume immediately ──────────────


def test_edit_button_opens_the_edit_form_instead_of_resuming(app_stub) -> None:
    """The old behaviour sent `{"choice": "edit"}` with no instructions at all."""
    module, stub = app_stub
    module.init_state()
    stub.pressed = {"✏️ Edit"}

    answer = module.render_approval_card(dict(APPROVAL_PAYLOAD))

    assert answer is None, "Edit must not resume the graph on its own"
    assert stub.session_state["edit_mode"] is True
    assert "edit_instructions_input" in {
        c[2].get("key") for c in stub.called("text_area")
    }


def test_edit_form_returns_instructions_and_or_replacement_text(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.session_state["edit_mode"] = True
    stub.pressed = {"▶️ Apply edit"}
    stub.values["edit_instructions_input"] = "drop the ordering nit"
    stub.values["edit_body_input"] = ""

    answer = module.render_approval_card(dict(APPROVAL_PAYLOAD))

    assert answer == {
        "choice": "edit",
        "feedback": "drop the ordering nit",
        "edited_body": None,
    }


def test_edit_form_can_send_verbatim_replacement_text(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.session_state["edit_mode"] = True
    stub.pressed = {"▶️ Apply edit"}
    stub.values["edit_instructions_input"] = ""
    stub.values["edit_body_input"] = "### My own review"

    answer = module.render_approval_card(dict(APPROVAL_PAYLOAD))
    assert answer["edited_body"] == "### My own review"


def test_apply_edit_with_nothing_typed_warns_and_does_not_resume(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.session_state["edit_mode"] = True
    stub.pressed = {"▶️ Apply edit"}  # no values → both boxes empty

    assert module.render_approval_card(dict(APPROVAL_PAYLOAD)) is None
    assert any("Enter instructions or paste" in str(c) for c in stub.called("warning"))


def test_cancel_edit_leaves_the_session_paused(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.session_state["edit_mode"] = True
    stub.pressed = {"Cancel edit"}

    assert module.render_approval_card(dict(APPROVAL_PAYLOAD)) is None
    assert stub.session_state["edit_mode"] is False


# ─── FIX 1: the edited-review gate ─────────────────────────────────────────


EDITED_PAYLOAD = {
    "kind": "edited_review",
    "pr_url": PR_URL,
    "thread_id": "t-ui",
    "confidence": 0.6,
    "confidence_reasoning": "Presented differently; the same risks remain.",
    "summary": "Revised summary.",
    "risk_factors": ["MD5 password hashing"],
    "comments": [
        {"file": "app/auth.py", "line": 12, "severity": "blocker", "body": "MD5 hashing"}
    ],
    "revision": 2,
    "proposed_body": "### Automated review\n\nRevised.",
    "diff_preview": "+hashlib.md5(...)",
}


def test_edited_review_card_confirms_posting(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.pressed = {"📮 Confirm & post"}

    answer = module.render_edited_review_card(dict(EDITED_PAYLOAD))

    assert answer == {"confirm": True, "feedback": ""}
    rendered = " ".join(str(c) for c in stub.calls)
    assert "revision 2" in rendered
    assert "Revised summary." in rendered
    assert "Nothing has been posted to GitHub yet." in rendered


def test_edited_review_card_can_reject(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.pressed = {"⛔ Cancel / reject"}
    answer = module.render_edited_review_card(dict(EDITED_PAYLOAD))
    assert answer == {"confirm": False, "feedback": ""}


def test_edited_review_card_edit_again_opens_another_revision(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.pressed = {"✏️ Edit again"}

    assert module.render_edited_review_card(dict(EDITED_PAYLOAD)) is None
    assert stub.session_state["edit_mode"] is True

    stub.pressed = {"▶️ Apply another revision"}
    stub.values["edit_again_instructions"] = "shorter"
    answer = module.render_edited_review_card(dict(EDITED_PAYLOAD))
    assert answer == {"choice": "edit", "feedback": "shorter", "edited_body": None}


def test_interrupt_dispatch_handles_the_edited_review_kind(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.pressed = {"📮 Confirm & post"}
    assert module.render_interrupt(dict(EDITED_PAYLOAD)) == {"confirm": True, "feedback": ""}


# ─── FIX 2: the dry-run toggle is locked for an active session ─────────────


def test_reset_for_records_the_session_mode(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.session_state["dry_run"] = True
    module.reset_for(PR_URL)
    assert stub.session_state["session_dry_run"] is True

    stub.session_state["dry_run"] = False
    module.reset_for(PR_URL)
    assert stub.session_state["session_dry_run"] is False


def test_open_session_restores_the_stored_mode(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    stub.session_state["dry_run"] = False  # the toggle currently says "live"
    view = module.SessionView(
        values={"final_action": "committed_dry_run", "dry_run": True}, dry_run=True
    )
    module.open_session(view, "t-old", PR_URL)
    assert stub.session_state["session_dry_run"] is True
    assert stub.session_state["dry_run"] is True, "the toggle must show the session mode"


def test_open_session_marks_a_finished_session_as_history(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    view = module.SessionView(values={"final_action": "committed"}, dry_run=False)
    module.open_session(view, "t-done", PR_URL)
    assert stub.session_state["session_finished"] is True
    assert stub.session_state["final"]["final_action"] == "committed"
    assert stub.session_state["interrupt_payload"] is None


def test_open_session_without_a_checkpoint_flags_history_only(app_stub) -> None:
    module, stub = app_stub
    module.init_state()
    module.open_session(module.SessionView(), "t-ghost", PR_URL)
    assert stub.session_state["history_only"] is True
    assert stub.session_state["final"] is None


# ─── terminal card: every outcome, including the new ones ──────────────────


@pytest.mark.parametrize(
    ("action", "expected_marker"),
    [
        ("committed_already", "success"),
        ("commit_unconfirmed", "error"),
        ("auto_refused_security", "error"),
    ],
)
def test_new_terminal_outcomes_are_rendered_honestly(
    app_stub, action: str, expected_marker: str
) -> None:
    module, stub = app_stub
    final = {"final_action": action, "post_error": "detail"}
    module.render_final(final, PR_URL, "t-ui")
    assert stub.called(expected_marker), f"expected st.{expected_marker} for {action}"


def test_unconfirmed_card_says_it_may_or_may_not_exist(app_stub) -> None:
    module, stub = app_stub
    module.render_final({"final_action": "commit_unconfirmed"}, PR_URL, "t")
    rendered = " ".join(str(c) for c in stub.called("error"))
    assert "could **not** be confirmed" in rendered
    assert not stub.called("success")


def test_completed_already_card_does_not_claim_a_fresh_post(app_stub) -> None:
    module, stub = app_stub
    module.render_final({"final_action": "committed_already"}, PR_URL, "t")
    rendered = " ".join(str(c) for c in stub.called("success"))
    assert "already" in rendered and "not** posted a second time" in rendered


def test_audit_error_is_surfaced_on_the_terminal_card(app_stub) -> None:
    module, stub = app_stub
    module.render_final(
        {"final_action": "committed", "audit_error": "commit: OperationalError"}, PR_URL, "t"
    )
    assert any("audit trail is incomplete" in str(c) for c in stub.called("warning"))


def test_approval_card_flags_a_security_pr(app_stub) -> None:
    module, stub = app_stub
    payload = dict(APPROVAL_PAYLOAD, risk_factors=["MD5 password hashing"])
    module.render_approval_card(payload)
    errors = [c for c in stub.calls if c[0] == "error"]
    assert errors and "security or blocker finding" in str(errors[0])


# ─── tests: escalation card ────────────────────────────────────────────────


def test_escalation_card_renders_risks_and_one_box_per_question(app_stub) -> None:
    module, stub = app_stub
    assert module.render_escalation_card(dict(ESCALATION_PAYLOAD)) is None
    text_areas = stub.called("text_area")
    assert len(text_areas) == len(ESCALATION_PAYLOAD["questions"])
    assert any(c[0] == "error" for c in stub.calls)  # risk factors are highlighted


def test_escalation_card_returns_question_to_answer_mapping(app_stub) -> None:
    module, stub = app_stub
    stub.submitted = {"Submit answers"}
    stub.values = {
        "escalation_answer_0": "MD5 is legacy",
        "escalation_answer_1": "staging only",
    }
    answers = module.render_escalation_card(dict(ESCALATION_PAYLOAD))
    assert answers == {
        "app/auth.py:12 — why MD5?": "MD5 is legacy",
        "sync/store.py:40 — is SYNC_URL HTTPS?": "staging only",
    }


def test_escalation_card_falls_back_to_a_default_question(app_stub) -> None:
    module, _ = app_stub
    payload = dict(ESCALATION_PAYLOAD, questions=[])
    module.render_escalation_card(payload)  # must not raise


# ─── tests: final confirmation card ────────────────────────────────────────


def test_final_confirmation_card_confirm_and_decline(app_stub) -> None:
    module, stub = app_stub
    stub.pressed = {"📮 Post this comment"}
    assert module.render_final_confirmation_card(dict(FINAL_PAYLOAD)) == {
        "confirm": True,
        "feedback": "",
    }
    stub.pressed = {"🚫 Do not post"}
    stub.values["final_feedback"] = "not yet"
    assert module.render_final_confirmation_card(dict(FINAL_PAYLOAD)) == {
        "confirm": False,
        "feedback": "not yet",
    }


# ─── tests: dispatch ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("payload", "expected_choice"),
    [
        (APPROVAL_PAYLOAD, "approve"),
        (ESCALATION_PAYLOAD, None),
        (FINAL_PAYLOAD, None),
    ],
)
def test_dispatch_routes_by_kind_and_waits_for_input(app_stub, payload, expected_choice) -> None:
    module, stub = app_stub
    if expected_choice:
        stub.pressed = {"✅ Approve"}
    answer = module.render_interrupt(dict(payload))
    if expected_choice:
        assert answer["choice"] == expected_choice
    else:
        assert answer is None  # rendered but no submission yet


def test_dispatch_reports_an_unknown_kind(app_stub) -> None:
    module, stub = app_stub
    assert module.render_interrupt({"kind": "mystery"}) is None
    assert any("Unknown interrupt kind" in str(c) for c in stub.calls)


# ─── tests: terminal card ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("action", "expected_marker"),
    [
        ("committed", "success"),
        ("auto_approved", "success"),
        ("committed_dry_run", "warning"),
        ("rejected", "warning"),
        ("commit_failed", "error"),
        ("fetch_failed", "error"),
        ("analyze_failed", "error"),
        ("something_else", "info"),
    ],
)
def test_final_card_uses_the_right_severity(app_stub, action: str, expected_marker: str) -> None:
    module, stub = app_stub
    final = {"final_action": action, "post_error": "403 nope"}
    module.render_final(final, PR_URL, "t-ui")
    assert stub.called(expected_marker), f"expected st.{expected_marker} for {action}"


def test_final_card_links_back_to_the_pr_and_shows_the_summary(app_stub) -> None:
    module, stub = app_stub
    analysis = make_analysis(0.95, summary="A tiny mechanical change.")
    module.render_final(
        {"final_action": "auto_approved", "analysis": analysis}, PR_URL, "t-ui"
    )
    markdown = " ".join(str(c[1][0]) for c in stub.called("markdown"))
    assert PR_URL in markdown
    assert "A tiny mechanical change." in markdown
    assert "thread_id" in " ".join(str(c[1][0]) for c in stub.called("caption"))


def test_final_card_does_not_claim_success_for_a_dry_run(app_stub) -> None:
    module, stub = app_stub
    module.render_final(
        {"final_action": "committed_dry_run", "posted_comment_body": "body"}, PR_URL, "t"
    )
    assert stub.called("warning")
    assert not stub.called("success")
    assert any("NOT posted" in str(c) for c in stub.called("warning"))


# ─── tests: the UI's resume loop, against the real graph ───────────────────


def test_ui_resume_values_drive_the_real_graph(app_stub) -> None:
    """The dicts the buttons return are exactly what the graph expects."""
    import asyncio

    from exercises.agent import Mode, build_graph, config_for

    module, _ = app_stub
    gh = FakeGitHub()
    app = build_graph(
        mode=Mode.EXERCISE_5,
        checkpointer=memory_checkpointer(),
        llm_factory=FakeLLMFactory(make_analysis(0.65)),
        github=gh,
    )
    cfg = config_for("t-ui-e2e")

    result = asyncio.run(app.ainvoke({"pr_url": PR_URL, "thread_id": "t-ui-e2e"}, cfg))
    payload = module.interrupt_payload(result)
    assert payload["kind"] == "approval_request"

    final = asyncio.run(
        app.ainvoke(Command(resume={"choice": "approve", "feedback": "ui note"}), cfg)
    )
    assert final["final_action"] == "committed"
    assert "Reviewer note: ui note" in gh.posts[0]["body"]


def test_ui_reject_path_posts_nothing(app_stub) -> None:
    import asyncio

    from exercises.agent import Mode, build_graph, config_for

    module, _ = app_stub
    gh = FakeGitHub()
    app = build_graph(
        mode=Mode.EXERCISE_5,
        checkpointer=memory_checkpointer(),
        llm_factory=FakeLLMFactory(make_analysis(0.65)),
        github=gh,
    )
    cfg = config_for("t-ui-reject")
    asyncio.run(app.ainvoke({"pr_url": PR_URL, "thread_id": "t-ui-reject"}, cfg))
    final = asyncio.run(app.ainvoke(Command(resume={"choice": "reject", "feedback": "no"}), cfg))
    assert final["final_action"] == "rejected"
    assert gh.posts == []
