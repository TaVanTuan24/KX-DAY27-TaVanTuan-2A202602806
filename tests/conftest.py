"""Shared test fixtures and fakes.

Nothing in this suite needs an API key: the LLM factory and the GitHub client are
replaced with in-process fakes, so every branch of the graph — including both
``interrupt()`` kinds — runs for real while no HTTP request is ever made and no
comment is ever posted to a public PR.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # make `common`, `audit`, `exercises`, `app` importable
    sys.path.insert(0, str(ROOT))

from common.github import PostResult, PostStatus, PullRequest, comment_marker  # noqa: E402
from common.schemas import PRAnalysis, ReviewComment  # noqa: E402

PR_URL = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
OTHER_PR_URL = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"

SAMPLE_DIFF = """diff --git a/app/auth.py b/app/auth.py
index 1111111..2222222 100644
--- a/app/auth.py
+++ b/app/auth.py
@@ -1,6 +1,12 @@
 import hashlib
+SYNC_URL = "http://sync.example.com/upload"
+
 def hash_password(pw: str) -> str:
-    return hashlib.sha256(pw.encode()).hexdigest()
+    return hashlib.md5(pw.encode()).hexdigest()
"""


def make_analysis(
    confidence: float,
    *,
    summary: str = "Adds a priority field to the task model.",
    risk_factors: list[str] | None = None,
    comments: list[ReviewComment] | None = None,
    questions: list[str] | None = None,
    reasoning: str = "Small mechanical change with one open schema question.",
) -> PRAnalysis:
    return PRAnalysis(
        summary=summary,
        risk_factors=risk_factors or [],
        comments=comments or [
            ReviewComment(
                file="app/auth.py",
                line=12,
                severity="suggestion",
                body="Consider adding a migration test.",
            )
        ],
        confidence=confidence,
        confidence_reasoning=reasoning,
        escalation_questions=questions or [],
    )


def make_pr(pr_url: str = PR_URL, **overrides: Any) -> PullRequest:
    base = dict(
        url=pr_url,
        owner="VinUni-AI20k",
        repo="PR-Demo",
        number=1,
        title="feat: add priority field to tasks",
        author="octocat",
        base_ref="main",
        head_ref="feature/priority",
        head_sha="abcdef1234567890",
        diff=SAMPLE_DIFF,
        files_changed=["app/auth.py"],
        description="Adds an optional priority column.",
        draft=False,
        state="open",
    )
    base.update(overrides)
    return PullRequest(**base)


class FakeLLM:
    """Stands in for ``StructuredOutputLLM``: returns queued replies in order.

    Each queued item is either a ``PRAnalysis`` (returned) or an ``Exception``
    (raised), which lets a test make the *first* call succeed and a later one
    (e.g. synthesis) fail.
    """

    def __init__(self, *replies: Any, error: Exception | None = None) -> None:
        self._queue = list(replies)
        self.error = error
        self.calls: list[list[dict]] = []

    def _next(self) -> PRAnalysis:
        if self.error is not None:
            raise self.error
        if not self._queue:
            raise AssertionError("FakeLLM ran out of queued analyses")
        reply = self._queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> PRAnalysis:
        self.calls.append(list(messages))
        return self._next()

    def invoke(self, messages: list[Any], **kwargs: Any) -> PRAnalysis:
        """Sync twin of ``ainvoke`` — the production nodes are all async."""
        self.calls.append(list(messages))
        return self._next()

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def last_prompt(self) -> str:
        return "\n".join(str(m.get("content", "")) for m in self.calls[-1])


class FakeLLMFactory:
    """``LLMFactory`` that always hands back the same :class:`FakeLLM`."""

    def __init__(self, *replies: Any, error: Exception | None = None) -> None:
        self.llm = FakeLLM(*replies, error=error)

    def __call__(self, *, temperature: float = 0.2) -> FakeLLM:
        return self.llm

    # convenience pass-throughs
    @property
    def call_count(self) -> int:
        return self.llm.call_count

    def last_prompt(self) -> str:
        return self.llm.last_prompt()


class FakeGitHub:
    """In-memory ``GitHubService``: records posts instead of calling the API.

    A confirmed POST returns :attr:`PostStatus.POSTED`; set ``post_status`` to
    exercise ``already_posted`` / ``unconfirmed`` and ``post_error`` for a
    definitive failure.
    """

    def __init__(
        self,
        pr: PullRequest | None = None,
        *,
        fetch_error: Exception | None = None,
        post_error: Exception | None = None,
        reviewer: str | None = "octocat",
        post_status: str = PostStatus.POSTED,
    ) -> None:
        self.pr = pr or make_pr()
        self.fetch_error = fetch_error
        self.post_error = post_error
        self.reviewer = reviewer
        self.post_status = post_status
        self.posts: list[dict] = []
        self.fetches: list[str] = []

    def fetch_pr(self, pr_url: str) -> PullRequest:
        self.fetches.append(pr_url)
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.pr

    def post_review_comment(self, pr_url: str, body: str, *, thread_id: str = "") -> PostResult:
        if self.post_error is not None:
            raise self.post_error
        if self.post_status == PostStatus.UNCONFIRMED:
            # Nothing is recorded: the caller must not treat this as posted.
            return PostResult(
                PostStatus.UNCONFIRMED,
                attempts=1,
                detail=(
                    "simulated ambiguous outcome (timeout after the POST) — the thread "
                    "marker is absent, so whether GitHub applied the POST is unknown; "
                    "not retried"
                ),
            )
        self.posts.append({"pr_url": pr_url, "body": body, "thread_id": thread_id})
        if self.post_status == PostStatus.ALREADY_POSTED:
            return PostResult(
                PostStatus.ALREADY_POSTED,
                comment_url="https://github.com/example/comment/existing",
                attempts=0,
                detail="simulated: the marker was already on the PR",
                marker_found=True,
            )
        return PostResult(
            PostStatus.POSTED,
            comment_url=f"https://github.com/example/comment/{len(self.posts)}",
            attempts=1,
            detail="simulated: GitHub confirmed the comment (201)",
        )

    def current_user(self) -> str | None:
        return self.reviewer


@pytest.fixture(autouse=True)
def _isolate_audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give every test its own audit/checkpoint DB.

    Audit-mode graphs (exercises 4–5) write through ``common.db.db_path()``, which
    reads ``HITL_DB_PATH``. Without this an incidental test would write rows into
    the developer's ``./hitl_audit.db`` — polluting real state and hiding the fact
    that the test forgot to isolate itself.

    Tests that need a specific path set ``HITL_DB_PATH`` themselves (an explicit or
    in-body ``monkeypatch.setenv`` runs after this fixture and wins).
    """
    path = tmp_path / "isolated-audit.db"
    monkeypatch.setenv("HITL_DB_PATH", str(path))
    return path


@pytest.fixture
def audit_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate every test in its own SQLite file."""
    path = tmp_path / "audit.db"
    monkeypatch.setenv("HITL_DB_PATH", str(path))
    monkeypatch.delenv("GITHUB_USER", raising=False)
    return path


@pytest.fixture
def github() -> FakeGitHub:
    return FakeGitHub()


@pytest.fixture
def pr_url() -> str:
    return PR_URL


@pytest.fixture
def marker_helper():
    """Expose ``comment_marker`` so tests can assert on posted bodies."""
    return comment_marker
