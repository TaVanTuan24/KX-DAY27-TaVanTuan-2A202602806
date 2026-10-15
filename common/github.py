"""Thin wrapper around the GitHub REST API.

We talk to GitHub directly via `httpx` instead of shelling out to `gh`, so
students don't need a `gh` binary — just `GITHUB_TOKEN` in `.env`.

API semantics (see README §"API semantics") — this module deliberately uses the
**Issues comments** endpoint to post a top-level *discussion comment* on the PR.
It does **not** submit a formal GitHub review (Approve / Request changes, which
require collaborator status) and it never merges anything.

Errors are normalised into :class:`GitHubError` subclasses so the graph can tell
a GitHub failure (401/403/404/429/timeout) apart from an LLM failure and never
report a review as posted when the API call failed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

API = "https://api.github.com"
PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
REQUEST_TIMEOUT = 30.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.5


class GitHubError(RuntimeError):
    """Base class for every GitHub failure we surface to the graph."""

    def __init__(self, message: str, *, status: int | None = None, url: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.url = url

    @property
    def friendly(self) -> str:
        return str(self)


class GitHubAuthError(GitHubError):
    """401 — token missing, expired or malformed."""


class GitHubPermissionError(GitHubError):
    """403/404 — token lacks scope, or the repo/PR is not visible to it."""


class GitHubNotFoundError(GitHubError):
    """404 — the repository or pull request does not exist."""


class GitHubRateLimitError(GitHubError):
    """429 (or a 403 caused by a spent rate limit)."""


class GitHubTimeoutError(GitHubError):
    """The request did not complete within the client timeout."""


class GitHubServerError(GitHubError):
    """5xx from GitHub."""


def _scrub(text: str) -> str:
    """Never let a token leak into an exception message or a log line."""
    token = os.environ.get("GITHUB_TOKEN")
    if token and token in text:
        text = text.replace(token, "***")
    return text


def _raise_for_status(resp: httpx.Response, action: str) -> None:
    """Translate an HTTP error response into a typed, token-free exception."""
    if resp.is_success:
        return
    status = resp.status_code
    detail = _scrub(resp.text or "")[:200]
    message = f"GitHub {action} failed: HTTP {status}"

    if status == 401:
        raise GitHubAuthError(
            f"{message} — GITHUB_TOKEN is invalid or expired. "
            "Create a new PAT at https://github.com/settings/tokens/new",
            status=status,
            url=str(resp.url),
        )
    if status == 404:
        raise GitHubNotFoundError(
            f"{message} — repository or pull request not found (or the token "
            "cannot see it). Check the PR URL.",
            status=status,
            url=str(resp.url),
        )
    if status == 403:
        remaining = resp.headers.get("x-ratelimit-remaining")
        if remaining == "0":
            reset = resp.headers.get("x-ratelimit-reset", "?")
            raise GitHubRateLimitError(
                f"{message} — API rate limit exhausted (resets at {reset}). "
                "Wait and retry.",
                status=status,
                url=str(resp.url),
            )
        raise GitHubPermissionError(
            f"{message} — the token lacks the required scope (`public_repo` for a "
            "public repo) or the account may not comment here.",
            status=status,
            url=str(resp.url),
        )
    if status == 429:
        retry_after = resp.headers.get("retry-after", "?")
        raise GitHubRateLimitError(
            f"{message} — rate limited (retry-after={retry_after}s).",
            status=status,
            url=str(resp.url),
        )
    if 500 <= status < 600:
        raise GitHubServerError(f"{message} — {detail}", status=status, url=str(resp.url))
    raise GitHubError(f"{message} — {detail}", status=status, url=str(resp.url))


@dataclass
class PullRequest:
    url: str
    owner: str
    repo: str
    number: int
    title: str
    author: str
    base_ref: str
    head_ref: str
    head_sha: str
    diff: str
    files_changed: list[str]
    description: str = ""
    draft: bool = False
    state: str = "open"


def _token() -> str:
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise GitHubAuthError(
            "GITHUB_TOKEN is not set. Copy .env.example to .env and paste a Personal "
            "Access Token (https://github.com/settings/tokens/new) with `public_repo` scope."
        )
    return tok


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Day27-HITL-Lab",
    }


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.search(pr_url or "")
    if not m:
        raise ValueError(f"Not a PR URL: {pr_url!r}")
    return m.group(1), m.group(2), int(m.group(3))


def is_valid_pr_url(pr_url: str) -> bool:
    """Cheap UI-level validation used by the Streamlit form."""
    return bool(PR_URL_RE.search(pr_url or ""))


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json: dict | None = None,
) -> httpx.Response:
    """One request with bounded retries for 429/5xx/timeout, then typed errors."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.request(method, url, headers=headers, json=json)
        except httpx.TimeoutException as exc:
            last_exc = GitHubTimeoutError(
                f"GitHub request to {url} timed out after {REQUEST_TIMEOUT:.0f}s "
                f"(attempt {attempt}/{MAX_ATTEMPTS}).",
                url=url,
            )
        except httpx.HTTPError as exc:
            last_exc = GitHubError(
                f"GitHub request to {url} failed: {type(exc).__name__}: {_scrub(str(exc))}",
                url=url,
            )
        else:
            retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
            if not retryable or attempt == MAX_ATTEMPTS:
                _raise_for_status(resp, f"{method} {url.rsplit('/', 1)[-1]}")
                return resp
            last_exc = GitHubServerError(
                f"GitHub {method} {url} returned HTTP {resp.status_code}",
                status=resp.status_code,
                url=url,
            )
        logger.warning("github retry %s/%s for %s: %s", attempt, MAX_ATTEMPTS, url, last_exc)
        time.sleep(RETRY_BACKOFF_S * attempt)
    assert last_exc is not None
    raise last_exc


def fetch_pr(pr_url: str, *, client: httpx.Client | None = None) -> PullRequest:
    """Fetch PR metadata + unified diff via the GitHub REST API.

    ``client`` lets callers (and tests) inject a transport; by default a fresh
    ``httpx.Client`` with a 30s timeout is used.
    """
    owner, repo, number = parse_pr_url(pr_url)
    base = f"{API}/repos/{owner}/{repo}/pulls/{number}"

    def _run(client: httpx.Client) -> PullRequest:
        meta = _request(client, "GET", base, headers=_headers()).json()
        diff = _request(
            client, "GET", base, headers=_headers(accept="application/vnd.github.v3.diff")
        ).text
        files = [
            f["filename"]
            for f in _request(client, "GET", f"{base}/files", headers=_headers()).json()
        ]
        return PullRequest(
            url=pr_url,
            owner=owner,
            repo=repo,
            number=number,
            title=meta.get("title", ""),
            author=(meta.get("user") or {}).get("login", "unknown"),
            base_ref=(meta.get("base") or {}).get("ref", ""),
            head_ref=(meta.get("head") or {}).get("ref", ""),
            head_sha=(meta.get("head") or {}).get("sha", ""),
            diff=diff,
            files_changed=files,
            description=meta.get("body") or "",
            draft=bool(meta.get("draft")),
            state=meta.get("state", "open"),
        )

    if client is not None:
        return _run(client)
    with httpx.Client(timeout=REQUEST_TIMEOUT) as owned:
        return _run(owned)


def _list_issue_comments(pr_url: str, client: httpx.Client) -> list[dict]:
    owner, repo, number = parse_pr_url(pr_url)
    url = f"{API}/repos/{owner}/{repo}/issues/{number}/comments"
    resp = _request(client, "GET", url, headers=_headers())
    return resp.json()


def comment_marker(thread_id: str) -> str:
    """Invisible HTML marker embedded in every comment we post.

    It is what makes posting idempotent: before posting we list the PR's
    comments and skip when a comment carrying the same marker is already there.
    """
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:12]
    return f"<!-- day27-hitl:{digest} -->"


def already_posted(pr_url: str, thread_id: str, *, client: httpx.Client | None = None) -> bool:
    """True when a previous run already posted this thread's comment."""
    marker = comment_marker(thread_id)
    if client is not None:
        return any(marker in (c.get("body") or "") for c in _list_issue_comments(pr_url, client))
    with httpx.Client(timeout=REQUEST_TIMEOUT) as owned:
        return any(marker in (c.get("body") or "") for c in _list_issue_comments(pr_url, owned))


def post_review_comment(
    pr_url: str,
    body: str,
    *,
    thread_id: str | None = None,
    client: httpx.Client | None = None,
    skip_if_exists: bool = True,
) -> str:
    """Post a top-level discussion comment on the PR. Returns the comment URL.

    Not a formal GitHub review and not a merge — see the module docstring.

    Idempotent: when ``thread_id`` is given, a deterministic HTML marker is
    appended and the call becomes a no-op if that marker is already present, so
    retries / resumes never post the same review twice.

    Raises a :class:`GitHubError` subclass on any API failure; a truthy return
    value means GitHub confirmed the comment.
    """
    owner, repo, number = parse_pr_url(pr_url)
    url = f"{API}/repos/{owner}/{repo}/issues/{number}/comments"

    payload_body = body
    if thread_id:
        marker = comment_marker(thread_id)
        if marker not in payload_body:
            payload_body = f"{body}\n\n{marker}"

    def _run(client: httpx.Client) -> str:
        if thread_id and skip_if_exists and already_posted(pr_url, thread_id, client=client):
            logger.info("comment for thread %s already present — skipping post", thread_id)
            return "already_posted"
        resp = _request(client, "POST", url, headers=_headers(), json={"body": payload_body})
        return resp.json().get("html_url", url)

    if client is not None:
        return _run(client)
    with httpx.Client(timeout=REQUEST_TIMEOUT) as owned:
        return _run(owned)


def current_user(*, client: httpx.Client | None = None) -> str | None:
    """GitHub login of the authenticated token, or None when unavailable.

    Used to attribute HITL decisions in the audit trail. Never raises: a missing
    identity must not break a review session.
    """
    try:
        if client is not None:
            resp = _request(client, "GET", f"{API}/user", headers=_headers())
        else:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as owned:
                resp = _request(owned, "GET", f"{API}/user", headers=_headers())
        return resp.json().get("login")
    except GitHubError as exc:
        logger.debug("could not resolve GitHub user: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("could not resolve GitHub user: %s", exc)
        return None


__all__ = [
    "API",
    "GitHubAuthError",
    "GitHubError",
    "GitHubNotFoundError",
    "GitHubPermissionError",
    "GitHubRateLimitError",
    "GitHubServerError",
    "GitHubTimeoutError",
    "PullRequest",
    "already_posted",
    "comment_marker",
    "current_user",
    "fetch_pr",
    "is_valid_pr_url",
    "parse_pr_url",
    "post_review_comment",
]
