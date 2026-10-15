"""GitHub client — every HTTP failure mode, with no real network traffic.

``httpx.MockTransport`` replaces the transport, so the real ``common.github``
code path (headers, status translation, retries, idempotent posting) runs while
nothing leaves the process.
"""

from __future__ import annotations

import json

import httpx
import pytest

from common.github import (
    API,
    GitHubAuthError,
    GitHubNotFoundError,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubServerError,
    GitHubTimeoutError,
    already_posted,
    comment_marker,
    current_user,
    fetch_pr,
    is_valid_pr_url,
    parse_pr_url,
    post_review_comment,
)
from tests.conftest import PR_URL

META = {
    "title": "feat: add priority field to tasks",
    "user": {"login": "octocat"},
    "base": {"ref": "main"},
    "head": {"ref": "feature/priority", "sha": "abcdef1234567890"},
    "body": "Adds an optional priority column.",
    "draft": False,
    "state": "open",
}
DIFF = "diff --git a/app/auth.py b/app/auth.py\n+import hashlib\n"
FILES = [{"filename": "app/auth.py"}, {"filename": "tests/test_auth.py"}]


@pytest.fixture(autouse=True)
def fake_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token must exist, otherwise every request fails before the transport."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_dummy_token_for_tests")


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def ok_handler(request: httpx.Request) -> httpx.Response:
    """Happy path: metadata, diff and files endpoints."""
    if request.url.path.endswith("/files"):
        return httpx.Response(200, json=FILES, request=request)
    if request.headers.get("accept") == "application/vnd.github.v3.diff":
        return httpx.Response(200, text=DIFF, request=request)
    return httpx.Response(200, json=META, request=request)


# ─── URL handling ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (PR_URL, ("VinUni-AI20k", "PR-Demo", 1)),
        ("https://github.com/o/r/pull/42/files", ("o", "r", 42)),
        ("http://github.com/o/r/pull/7", ("o", "r", 7)),
    ],
)
def test_parse_pr_url(url: str, expected: tuple[str, str, int]) -> None:
    assert parse_pr_url(url) == expected


@pytest.mark.parametrize(
    "url",
    ["", "not a url", "https://gitlab.com/o/r/merge_requests/1", "https://github.com/o/r"],
)
def test_parse_pr_url_rejects_junk(url: str) -> None:
    with pytest.raises(ValueError):
        parse_pr_url(url)
    assert is_valid_pr_url(url) is False


def test_is_valid_pr_url_accepts_a_pr_link() -> None:
    assert is_valid_pr_url(PR_URL) is True


# ─── fetch ─────────────────────────────────────────────────────────────────


def test_fetch_pr_returns_metadata_diff_and_files() -> None:
    with make_client(ok_handler) as client:
        pr = fetch_pr(PR_URL, client=client)
    assert pr.title == META["title"]
    assert pr.author == "octocat"
    assert pr.head_sha == "abcdef1234567890"
    assert pr.diff == DIFF
    assert pr.files_changed == ["app/auth.py", "tests/test_auth.py"]
    assert pr.description == "Adds an optional priority column."
    assert pr.number == 1


def test_fetch_pr_sends_the_token_header(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return ok_handler(request)

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    with make_client(handler) as client:
        fetch_pr(PR_URL, client=client)
    assert all(h == "Bearer ghp_test_token" for h in seen)


def test_fetch_pr_without_a_token_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with make_client(ok_handler) as client:
        with pytest.raises(GitHubAuthError, match="GITHUB_TOKEN is not set"):
            fetch_pr(PR_URL, client=client)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, GitHubAuthError),
        (403, GitHubPermissionError),
        (404, GitHubNotFoundError),
        (422, Exception),
    ],
)
def test_fetch_pr_maps_http_errors(status: int, expected: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": "nope"}, request=request)

    with make_client(handler) as client:
        with pytest.raises(expected):
            fetch_pr(PR_URL, client=client)


def test_permission_error_message_mentions_the_scope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Forbidden"}, request=request)

    with make_client(handler) as client:
        with pytest.raises(GitHubPermissionError) as exc:
            fetch_pr(PR_URL, client=client)
    assert "public_repo" in str(exc.value)


def test_rate_limit_is_detected_from_the_403_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000000"},
            request=request,
        )

    with make_client(handler) as client:
        with pytest.raises(GitHubRateLimitError) as exc:
            fetch_pr(PR_URL, client=client)
    assert exc.value.status == 403


def test_429_raises_rate_limit_after_bounded_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("common.github.RETRY_BACKOFF_S", 0.0)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(429, headers={"retry-after": "1"}, request=request)

    with make_client(handler) as client:
        with pytest.raises(GitHubRateLimitError) as exc:
            fetch_pr(PR_URL, client=client)
    assert attempts["n"] == 3  # MAX_ATTEMPTS
    assert exc.value.status == 429


def test_500_is_retried_then_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("common.github.RETRY_BACKOFF_S", 0.0)
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(503, text="unavailable", request=request)

    with make_client(handler) as client:
        with pytest.raises(GitHubServerError):
            fetch_pr(PR_URL, client=client)
    assert attempts["n"] == 3


def test_a_transient_500_then_success_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("common.github.RETRY_BACKOFF_S", 0.0)
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] == 1:
            return httpx.Response(502, request=request)
        return ok_handler(request)

    with make_client(handler) as client:
        pr = fetch_pr(PR_URL, client=client)
    assert pr.title == META["title"]
    assert seen["n"] > 1


def test_timeout_becomes_a_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("common.github.RETRY_BACKOFF_S", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with make_client(handler) as client:
        with pytest.raises(GitHubTimeoutError, match="timed out"):
            fetch_pr(PR_URL, client=client)


def test_connection_error_becomes_a_github_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with make_client(handler) as client:
        with pytest.raises(Exception) as exc:
            fetch_pr(PR_URL, client=client)
    assert "no route to host" in str(exc.value)


# ─── posting ───────────────────────────────────────────────────────────────


def test_post_review_comment_uses_the_issues_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    calls: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[], request=request)  # no existing comments
        body = json.loads(request.content)
        calls.append((request.method, str(request.url), body))
        return httpx.Response(
            201, json={"html_url": "https://github.com/.../comment/1"}, request=request
        )

    with make_client(handler) as client:
        url = post_review_comment(PR_URL, "review body", thread_id="t1", client=client)

    assert url.endswith("/comment/1")
    method, posted_url, body = calls[0]
    assert method == "POST"
    assert posted_url == f"{API}/repos/VinUni-AI20k/PR-Demo/issues/1/comments"
    assert comment_marker("t1") in body["body"]  # idempotency marker
    assert body["body"].startswith("review body")


def test_post_is_skipped_when_the_marker_is_already_present() -> None:
    """A retry / resume must not post the same review twice."""
    existing = [{"body": f"previous review\n\n{comment_marker('t1')}"}]
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=existing, request=request)
        posts["n"] += 1
        return httpx.Response(201, json={"html_url": "x"}, request=request)

    with make_client(handler) as client:
        result = post_review_comment(PR_URL, "review body", thread_id="t1", client=client)
    assert result == "already_posted"
    assert posts["n"] == 0


def test_different_threads_get_different_markers_and_both_post() -> None:
    existing: list[dict] = []
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=existing, request=request)
        posts["n"] += 1
        payload = json.loads(request.content)
        existing.append({"body": payload["body"]})
        return httpx.Response(201, json={"html_url": f"c{posts['n']}"}, request=request)

    with make_client(handler) as client:
        a = post_review_comment(PR_URL, "review", thread_id="thread-a", client=client)
        b = post_review_comment(PR_URL, "review", thread_id="thread-b", client=client)
    assert (a, b) == ("c1", "c2")
    assert comment_marker("thread-a") != comment_marker("thread-b")


def test_post_can_be_forced_even_when_a_marker_exists() -> None:
    existing = [{"body": comment_marker("t1")}]
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=existing, request=request)
        posts["n"] += 1
        return httpx.Response(201, json={"html_url": "c"}, request=request)

    with make_client(handler) as client:
        post_review_comment(
            PR_URL, "review", thread_id="t1", client=client, skip_if_exists=False
        )
    assert posts["n"] == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, GitHubAuthError),
        (403, GitHubPermissionError),
        (404, GitHubNotFoundError),
        (429, GitHubRateLimitError),
    ],
)
def test_post_failures_are_typed(monkeypatch: pytest.MonkeyPatch, status: int, expected: type[Exception]) -> None:
    monkeypatch.setattr("common.github.RETRY_BACKOFF_S", 0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=[], request=request)
        return httpx.Response(status, json={"message": "no"}, request=request)

    with make_client(handler) as client:
        with pytest.raises(expected):
            post_review_comment(PR_URL, "review", thread_id="t1", client=client)


def test_already_posted_helper() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=[{"body": comment_marker("t9")}], request=request
        )

    with make_client(handler) as client:
        assert already_posted(PR_URL, "t9", client=client) is True
        assert already_posted(PR_URL, "other", client=client) is False


def test_current_user_returns_none_instead_of_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "bad credentials"}, request=request)

    with make_client(handler) as client:
        assert current_user(client=client) is None


def test_current_user_returns_the_login() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"login": "tavan"}, request=request)

    with make_client(handler) as client:
        assert current_user(client=client) == "tavan"


def test_token_never_appears_in_an_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors are surfaced in logs/UI, so the token must be scrubbed."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_supersecretvalue")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request ghp_supersecretvalue", request=request)

    with make_client(handler) as client:
        with pytest.raises(Exception) as exc:
            fetch_pr(PR_URL, client=client)
    assert "ghp_supersecretvalue" not in str(exc.value)
    assert "***" in str(exc.value)
