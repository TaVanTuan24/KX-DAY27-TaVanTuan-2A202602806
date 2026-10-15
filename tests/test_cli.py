"""Terminal prompt helpers (`exercises/cli.py`).

These build the dicts that are handed to `Command(resume=...)`, so a wrong key
name or a crash here would break the interactive path in a way the graph tests
cannot see. `rich.prompt.Prompt.ask` is patched to feed answers.
"""

from __future__ import annotations

import pytest

from exercises import cli

APPROVAL = {
    "kind": "approval_request",
    "pr_url": "https://github.com/o/r/pull/1",
    "confidence": 0.65,
    "confidence_reasoning": "One open question.",
    "summary": "Adds a priority column.",
    "risk_factors": ["Schema migration needs a backfill plan"],
    "comments": [
        {"file": "models/task.py", "line": 18, "severity": "issue", "body": "add a migration test"}
    ],
    "diff_preview": "+priority = Column(int)",
}

ESCALATION = {
    "kind": "escalation",
    "pr_url": "https://github.com/o/r/pull/2",
    "confidence": 0.31,
    "confidence_reasoning": "Multiple red flags.",
    "summary": "Adds login and sync.",
    "risk_factors": ["MD5 password hashing", "plaintext token"],
    "questions": ["app/auth.py:12 — why MD5?", "sync/store.py:40 — is SYNC_URL HTTPS?"],
    "diff_preview": "+hashlib.md5(...)",
}

FINAL = {
    "kind": "final_confirmation",
    "pr_url": "https://github.com/o/r/pull/2",
    "confidence": 0.62,
    "confidence_reasoning": "TLS still unresolved.",
    "summary": "Adds login and sync.",
    "risk_factors": ["still HTTP"],
    "comments": [{"file": "app/auth.py", "line": 12, "severity": "blocker", "body": "MD5"}],
    "escalation_answers": {"app/auth.py:12 — why MD5?": "legacy"},
    "diff_preview": "+hashlib.md5(...)",
}


class Answers:
    """Sequential answers for the module's single `ask()` seam."""

    def __init__(self, *values: str) -> None:
        self._values = list(values)
        self.asked: list[str] = []

    def __call__(self, prompt: str = "", **kwargs) -> str:
        self.asked.append(prompt)
        if not self._values:
            return kwargs.get("default") or ""
        return self._values.pop(0)


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch):
    def install(*values: str) -> Answers:
        stub = Answers(*values)
        monkeypatch.setattr(cli, "ask", stub)
        return stub

    return install


def test_approval_card_returns_the_resume_dict(answers) -> None:
    stub = answers("approve")
    assert cli.render_approval_request(dict(APPROVAL)) == {"choice": "approve", "feedback": ""}
    assert stub.asked  # the reviewer was actually prompted


def test_approval_card_reject_collects_a_reason(answers) -> None:
    answers("reject", "needs a migration test first")
    result = cli.render_approval_request(dict(APPROVAL))
    assert result == {"choice": "reject", "feedback": "needs a migration test first"}


def test_approval_card_edit_collects_the_edit_text(answers) -> None:
    answers("edit", "reword the migration note")
    result = cli.render_approval_request(dict(APPROVAL))
    assert result == {"choice": "edit", "feedback": "reword the migration note"}


def test_escalation_card_maps_answers_to_questions(answers) -> None:
    answers("MD5 is legacy", "staging only")
    result = cli.render_escalation(dict(ESCALATION))
    assert result == {
        "app/auth.py:12 — why MD5?": "MD5 is legacy",
        "sync/store.py:40 — is SYNC_URL HTTPS?": "staging only",
    }


def test_escalation_card_tolerates_an_empty_answer(answers) -> None:
    answers("", "")
    result = cli.render_escalation(dict(ESCALATION))
    assert list(result.values()) == ["", ""]


def test_final_confirmation_yes(answers) -> None:
    answers("yes")
    assert cli.render_final_confirmation(dict(FINAL)) == {"confirm": True, "feedback": ""}


def test_final_confirmation_no_collects_a_reason(answers) -> None:
    answers("no", "not before the TLS fix")
    assert cli.render_final_confirmation(dict(FINAL)) == {
        "confirm": False,
        "feedback": "not before the TLS fix",
    }


def test_handle_interrupt_dispatches_on_kind(answers) -> None:
    answers("approve")
    assert cli.handle_interrupt(dict(APPROVAL))["choice"] == "approve"

    answers("a", "b")
    assert cli.handle_interrupt(dict(ESCALATION)) == {
        ESCALATION["questions"][0]: "a",
        ESCALATION["questions"][1]: "b",
    }

    answers("yes")
    assert cli.handle_interrupt(dict(FINAL))["confirm"] is True


def test_handle_interrupt_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown interrupt payload kind"):
        cli.handle_interrupt({"kind": "mystery"})


def test_rendering_does_not_crash_on_minimal_payloads(answers) -> None:
    """Every optional field must be optional — the UI/CLI share the payload."""
    answers("reject", "")
    cli.render_approval_request({"confidence": 0.6})
    answers("")
    cli.render_escalation({"confidence": 0.3, "questions": ["q1"]})
    answers("no", "")
    cli.render_final_confirmation({"confidence": 0.6})
