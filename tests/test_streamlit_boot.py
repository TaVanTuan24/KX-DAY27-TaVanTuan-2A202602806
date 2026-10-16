"""Streamlit app boot + interaction, executed for real.

`streamlit.testing.v1.AppTest` runs ``app.py`` in-process, so the module body,
session state, our graph call and the widgets all actually execute — the LLM and
GitHub are the only fakes. This is the closest thing to clicking through the UI
without a browser, and it proves the app does not raise at import or on a rerun.

These tests spin up a real Streamlit script runner per rerun, so they are marked
``slow`` (deselect with ``-m "not slow"``); the fast, stubbed UI checks live in
``test_streamlit_app.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from common.schemas import ReviewComment
from tests.conftest import PR_URL, FakeGitHub, FakeLLMFactory, make_analysis

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An ``AppTest`` whose LLM and GitHub boundaries are faked.

    ``exercises.agent.RealLLMFactory`` / ``RealGitHubService`` are the two
    factories ``build_graph`` reaches for by default; replacing them keeps the
    whole app real (checkpointer, graph, audit trail, widgets) with no network.
    """
    import exercises.agent as agent
    import app as app_module

    monkeypatch.setenv("HITL_DB_PATH", str(tmp_path / "ui.db"))
    gh = FakeGitHub()

    def factory_factory(*analyses):
        factory = FakeLLMFactory(*analyses)
        monkeypatch.setattr(
            agent, "RealLLMFactory", lambda: (lambda *, temperature=0.2: factory.llm)
        )
        return factory

    monkeypatch.setattr(agent, "RealGitHubService", lambda: gh)

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30.0)

    def build(*analyses) -> FakeLLMFactory:
        """Point the app at a fresh fake LLM and return that fake.

        `app.py` is already loaded by AppTest; this only swaps the model the
        next graph run will use.
        """
        return factory_factory(*analyses)

    build()
    at.run()
    assert not at.exception, at.exception
    return at, build, gh


def _texts(at) -> str:
    parts = []
    for attr in ("title", "subheader", "markdown", "caption", "error", "warning", "success", "info"):
        for el in getattr(at, attr, []):
            parts.append(str(getattr(el, "value", "")) or str(getattr(el, "body", "")))
    return "\n".join(parts)


def test_app_boots_and_renders_the_input_form(app_test) -> None:
    at, _, _ = app_test
    assert any("GitHub Pull Request URL" in el.label for el in at.text_input)
    assert any("Run review" in el.label for el in at.button)
    sidebar = "\n".join(str(m.value) for m in at.markdown) + "\n".join(
        str(p.value) for p in at.caption
    )
    assert "Recent sessions" in sidebar or "No sessions yet" in sidebar


def test_invalid_pr_url_is_rejected_with_a_friendly_error(app_test) -> None:
    at, _, gh = app_test
    at.text_input[0].set_value("https://example.com/not-a-pr")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("not a GitHub PR URL" in e.value for e in at.error)
    assert gh.fetches == [] and gh.posts == []


def test_approve_flow_through_the_widgets(app_test) -> None:
    at, build, gh = app_test
    factory = build(
        make_analysis(
            0.65,
            summary="Adds an optional priority column.",
            risk_factors=["Schema migration needs a backfill plan"],
            comments=[ReviewComment(file="models/task.py", line=18, severity="issue", body="add a migration test")],
        )
    )
    # ── start a review ─────────────────────────────────────────────────────
    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert factory.call_count == 1
    assert gh.fetches == [PR_URL]
    assert gh.posts == []  # paused, nothing posted

    rendered = _texts(at)
    assert "Approval requested" in rendered
    assert "Schema migration needs a backfill plan" in rendered
    assert "add a migration test" in rendered

    # ── approve ────────────────────────────────────────────────────────────
    at.text_area(key="approval_feedback").set_value("looks good")
    next(b for b in at.button if "Approve" in b.label).click().run()
    assert not at.exception, at.exception
    assert len(gh.posts) == 1
    assert "Reviewer note: looks good" in gh.posts[0]["body"]
    assert "Comment posted to GitHub" in _texts(at)


def test_reject_flow_through_the_widgets(app_test) -> None:
    at, build, gh = app_test
    build(make_analysis(0.65))
    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    assert "Approval requested" in _texts(at)

    next(b for b in at.button if "Reject" in b.label).click().run()
    assert not at.exception, at.exception
    assert gh.posts == []
    assert "Rejected" in _texts(at)


def test_edit_button_opens_a_form_and_posts_nothing(app_test) -> None:
    """FIX 1 in the real UI: Edit shows a form; it does not resume or POST."""
    at, build, gh = app_test
    build(make_analysis(0.65))
    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()

    next(b for b in at.button if "Edit" in b.label).click().run()
    assert not at.exception, at.exception
    assert gh.posts == [], "opening the edit form must not post"
    assert any(ta.key == "edit_instructions_input" for ta in at.text_area), (
        "the edit form should be visible"
    )


def test_applying_an_empty_edit_warns_instead_of_resuming(app_test) -> None:
    at, build, gh = app_test
    build(make_analysis(0.65))
    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    next(b for b in at.button if "Edit" in b.label).click().run()

    next(b for b in at.button if "Apply edit" in b.label).click().run()
    assert not at.exception, at.exception
    assert any("Enter instructions or paste" in w.value for w in at.warning)
    assert gh.posts == []  # still paused


def test_edit_through_the_ui_previews_then_posts_on_confirm(app_test) -> None:
    """FIX 1 end to end in the real UI: Edit → preview → Confirm → one POST."""
    at, build, gh = app_test
    build(
        make_analysis(0.65, summary="ORIGINAL SUMMARY"),
        make_analysis(0.6, summary="REVISED SUMMARY"),
    )
    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()

    next(b for b in at.button if "Edit" in b.label).click().run()
    at.text_area(key="edit_instructions_input").set_value("shorten the summary")
    next(b for b in at.button if "Apply edit" in b.label).click().run()
    assert not at.exception, at.exception

    rendered = _texts(at)
    assert "Edited review" in rendered
    assert "REVISED SUMMARY" in rendered
    assert "Nothing has been posted to GitHub yet." in rendered
    assert gh.posts == [], "the revised review must not be posted before confirmation"

    next(b for b in at.button if "Confirm & post" in b.label).click().run()
    assert not at.exception, at.exception
    assert len(gh.posts) == 1
    assert "REVISED SUMMARY" in gh.posts[0]["body"]
    assert "ORIGINAL SUMMARY" not in gh.posts[0]["body"]


def test_edit_then_cancel_posts_nothing_through_the_ui(app_test) -> None:
    at, build, gh = app_test
    build(make_analysis(0.65), make_analysis(0.6))
    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    next(b for b in at.button if "Edit" in b.label).click().run()
    at.text_area(key="edit_instructions_input").set_value("shorter")
    next(b for b in at.button if "Apply edit" in b.label).click().run()

    next(b for b in at.button if "Cancel / reject" in b.label).click().run()
    assert not at.exception, at.exception
    assert gh.posts == []
    assert "Rejected" in _texts(at)


def _answer_questions(at, *answers: str) -> None:
    """Fill the escalation form and submit it in ONE rerun.

    Streamlit's form semantics matter here: values only reach the form's submit
    callback in the rerun where the button is pressed, so ``set_value`` must not
    trigger a run of its own.
    """
    boxes = [ta for ta in at.text_area if ta.key and ta.key.startswith("escalation_answer_")]
    assert len(boxes) == len(answers)
    for box, answer in zip(boxes, answers):
        box.set_value(answer)
    next(b for b in at.button if "Submit answers" in b.label).click().run()


def test_escalation_flow_through_the_widgets(app_test) -> None:
    at, build, gh = app_test
    initial = make_analysis(
        0.31,
        summary="Adds user login and cloud sync.",
        risk_factors=["MD5 password hashing", "SYNC_URL is plain HTTP"],
        comments=[ReviewComment(file="app/auth.py", line=12, severity="blocker", body="MD5 hashing")],
        questions=["app/auth.py:12 — why MD5?", "sync/store.py:40 — is SYNC_URL HTTPS?"],
    )
    refined = make_analysis(
        0.62,
        summary="Adds login and sync; HTTP transport still unresolved.",
        risk_factors=["SYNC_URL is plain HTTP"],
        comments=[ReviewComment(file="app/auth.py", line=12, severity="blocker", body="MD5 hashing")],
    )
    build(initial, refined)

    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    assert not at.exception, at.exception

    rendered = _texts(at)
    assert "Strong escalation" in rendered
    assert "MD5 password hashing" in rendered
    assert "why MD5?" in rendered

    # Submitting an empty form must not resume the graph.
    next(b for b in at.button if "Submit answers" in b.label).click().run()
    assert any("answer at least one question" in w.value for w in at.warning)
    assert gh.posts == []

    # Answer both questions → refinement → final confirmation gate.
    _answer_questions(at, "legacy, migrating next sprint", "staging only for now")
    assert not at.exception, at.exception

    rendered = _texts(at)
    assert "Refined review ready" in rendered
    assert "Nothing has been posted to GitHub yet." in rendered
    assert gh.posts == []

    next(b for b in at.button if "Post this comment" in b.label).click().run()
    assert not at.exception, at.exception
    assert len(gh.posts) == 1
    assert "Reviewer answers to escalation questions" in gh.posts[0]["body"]


def test_decline_at_final_confirmation_posts_nothing(app_test) -> None:
    at, build, gh = app_test
    build(make_analysis(0.31, questions=["app/auth.py:12 — why MD5?"]), make_analysis(0.62))

    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    _answer_questions(at, "legacy")
    assert "Refined review ready" in _texts(at)

    next(b for b in at.button if "Do not post" in b.label).click().run()
    assert not at.exception, at.exception
    assert gh.posts == []
    assert "Rejected" in _texts(at)


def test_auto_approve_branch_renders_a_success_card(app_test) -> None:
    at, build, gh = app_test
    build(make_analysis(0.95, summary="Renames an internal helper."))

    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(gh.posts) == 1

    rendered = _texts(at)
    assert "Comment posted to GitHub" in rendered
    assert PR_URL in rendered  # link back to the PR
    assert "Renames an internal helper." in rendered


def test_github_failure_is_shown_and_not_reported_as_posted(app_test) -> None:
    from common.github import GitHubPermissionError

    at, build, gh = app_test
    gh.post_error = GitHubPermissionError("403 — token lacks the public_repo scope")
    build(make_analysis(0.95))

    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert gh.posts == []
    rendered = _texts(at)
    assert "403" in rendered
    assert "Comment posted to GitHub" not in rendered


def test_state_survives_a_streamlit_rerun(app_test) -> None:
    """A rerun (what Streamlit does after any widget interaction) keeps session state."""
    at, build, _ = app_test
    build(make_analysis(0.65))
    at.text_input[0].set_value(PR_URL)
    at.button[0].click().run()
    assert "Approval requested" in _texts(at)

    at.run()  # bare rerun
    assert "Approval requested" in _texts(at)
    assert not at.exception, at.exception
