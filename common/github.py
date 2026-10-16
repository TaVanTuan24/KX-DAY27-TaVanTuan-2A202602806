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

Two reliability properties matter here and are pinned by tests:

* **List endpoints are paginated** (:func:`_paginate`) — a PR with more than one
  page of files/comments must not be silently truncated, and the duplicate-marker
  check must see *every* page. Pagination links are only followed when they point
  at ``https://api.github.com``.
* **A POST is never retried blindly.** A timeout or a 5xx can mean GitHub *did*
  apply the request, so retrying would duplicate the comment. Instead the outcome
  is re-checked with the thread marker and reported as
  :attr:`PostStatus.UNCONFIRMED` when it genuinely cannot be determined. This is
  at-most-once in practice, not exactly-once: two concurrent processes can still
  race (the marker check is not atomic).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

API = "https://api.github.com"
API_HOST = "api.github.com"
PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")
REQUEST_TIMEOUT = 30.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 0.5
#: List endpoints ask for the maximum page size…
PER_PAGE = 100
#: …and pagination stops after this many pages, so a broken/looping Link header
#: cannot spin forever (20 × 100 = 2000 items, well past any real PR or
#: comment thread).
MAX_PAGES = 20

#: Only these are safe to retry: a repeated GET has no side effect.
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD"})

LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="([^"]+)"')


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


class GitHubConnectionError(GitHubError):
    """The transport failed before a response arrived.

    Ambiguous for a POST: the request may have reached GitHub.
    """


class GitHubServerError(GitHubError):
    """5xx from GitHub."""


#: Failures after which a request may still have been applied server-side.
#: A POST hitting one of these must NOT be retried — see `post_review_comment`.
AMBIGUOUS_ERRORS = (GitHubTimeoutError, GitHubConnectionError, GitHubServerError)


class PostStatus:
    """Outcome of an attempted :func:`post_review_comment`."""

    #: GitHub confirmed a new comment.
    POSTED = "posted"
    #: The comment (for this thread) was already on the PR — a retry/resume, or a
    #: previous attempt that we could not confirm at the time.
    ALREADY_POSTED = "already_posted"
    #: The POST may or may not have been applied and we could not verify it.
    #: Never report this as success or as failure.
    UNCONFIRMED = "unconfirmed"

    ALL = (POSTED, ALREADY_POSTED, UNCONFIRMED)


@dataclass
class PostResult:
    """What actually happened when we tried to post a review.

    ``ok`` is only true when the comment is *known* to be on GitHub (either we
    just created it, or we found it). ``unconfirmed`` is deliberately neither
    success nor failure.
    """

    status: str
    comment_url: str | None = None
    attempts: int = 1
    detail: str = ""
    marker_found: bool = False

    @property
    def ok(self) -> bool:
        return self.status in (PostStatus.POSTED, PostStatus.ALREADY_POSTED)

    @property
    def unconfirmed(self) -> bool:
        return self.status == PostStatus.UNCONFIRMED


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
    retry: bool | None = None,
) -> httpx.Response:
    """One request with bounded retries, then typed errors.

    ``retry`` defaults to "yes only for idempotent methods". A POST is never
    retried: a timeout or 5xx may mean GitHub *did* create the comment, and a
    second attempt would duplicate it. Callers that need at-most-once POST
    semantics use :func:`post_review_comment`, which verifies the outcome instead.
    """
    if retry is None:
        retry = method.upper() in IDEMPOTENT_METHODS
    max_attempts = MAX_ATTEMPTS if retry else 1

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.request(method, url, headers=headers, json=json)
        except httpx.TimeoutException:
            last_exc = GitHubTimeoutError(
                f"GitHub request to {url} timed out after {REQUEST_TIMEOUT:.0f}s "
                f"(attempt {attempt}/{max_attempts}).",
                url=url,
            )
        except httpx.HTTPError as exc:
            last_exc = GitHubConnectionError(
                f"GitHub request to {url} failed: {type(exc).__name__}: {_scrub(str(exc))}",
                url=url,
            )
        else:
            retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
            if not retryable or attempt == max_attempts:
                _raise_for_status(resp, f"{method} {url.rsplit('/', 1)[-1]}")
                return resp
            last_exc = GitHubServerError(
                f"GitHub {method} {url} returned HTTP {resp.status_code}",
                status=resp.status_code,
                url=url,
            )
        logger.warning("github retry %s/%s for %s: %s", attempt, max_attempts, url, last_exc)
        time.sleep(RETRY_BACKOFF_S * attempt)
    assert last_exc is not None
    raise last_exc


# ─── pagination ────────────────────────────────────────────────────────────


def _is_trusted_api_url(url: str) -> bool:
    """Only follow pagination links that stay on the GitHub API, over HTTPS.

    A ``Link`` header is server-supplied data: following an arbitrary host would
    leak the ``Authorization`` header (and the token) to a third party.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme == "https" and parsed.hostname == API_HOST


def _next_link(resp: httpx.Response) -> str | None:
    """The ``rel="next"`` URL from a ``Link`` header, if it is safe to follow."""
    header = resp.headers.get("link") or resp.headers.get("Link")
    if not header:
        return None
    for match in LINK_RE.finditer(header):
        url, rel = match.group(1).strip(), match.group(2).strip()
        if rel != "next":
            continue
        if not _is_trusted_api_url(url):
            logger.warning("ignoring untrusted pagination link: %s", url)
            return None
        return url
    return None


def _paginate(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """Collect every page of a list endpoint, following ``rel="next"`` links.

    Bounded by ``max_pages`` so a malformed or looping ``Link`` header cannot spin
    forever; when the cap is hit it is logged and the pages collected so far are
    returned rather than failing the whole request.
    """
    items: list[dict] = []
    next_url: str | None = url
    pages = 0
    while next_url:
        if pages >= max_pages:
            logger.warning(
                "pagination stopped after %s page(s) for %s — result may be truncated",
                pages,
                url,
            )
            break
        pages += 1
        resp = _request(client, "GET", next_url, headers=headers)
        data = resp.json()
        if isinstance(data, list):
            items.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            items.append(data)
        next_url = _next_link(resp)
    return items


def _with_per_page(url: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}per_page={PER_PAGE}"


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
        # Paginated: GitHub caps this endpoint at 3000 files, and one page is 100.
        files = [
            f["filename"]
            for f in _paginate(
                client, _with_per_page(f"{base}/files"), headers=_headers()
            )
            if f.get("filename")
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


def _comments_url(pr_url: str) -> str:
    owner, repo, number = parse_pr_url(pr_url)
    return f"{API}/repos/{owner}/{repo}/issues/{number}/comments"


def _list_issue_comments(
    pr_url: str, client: httpx.Client, *, max_pages: int = MAX_PAGES
) -> list[dict]:
    """Every comment on the PR, across all pages."""
    return _paginate(
        client, _with_per_page(_comments_url(pr_url)), headers=_headers(), max_pages=max_pages
    )


def comment_marker(thread_id: str) -> str:
    """Invisible HTML marker embedded in every comment we post.

    It is what makes posting idempotent: before posting we list the PR's
    comments and skip when a comment carrying the same marker is already there.
    """
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:12]
    return f"<!-- day27-hitl:{digest} -->"


def _find_marker(
    client: httpx.Client, pr_url: str, marker: str, *, max_pages: int = MAX_PAGES
) -> dict | None:
    """The comment carrying ``marker`` (across all pages), or None.

    Returns the comment itself rather than its ``html_url``: a missing ``html_url``
    must not be indistinguishable from "the marker was not found", which is what
    makes posting idempotent.
    """
    for comment in _list_issue_comments(pr_url, client, max_pages=max_pages):
        if marker in (comment.get("body") or ""):
            return comment
    return None


def already_posted(pr_url: str, thread_id: str, *, client: httpx.Client | None = None) -> bool:
    """True when a previous run already posted this thread's comment.

    Scans every page of the PR's comments, not just the first.
    """
    marker = comment_marker(thread_id)
    if client is not None:
        return _find_marker(client, pr_url, marker) is not None
    with httpx.Client(timeout=REQUEST_TIMEOUT) as owned:
        return _find_marker(owned, pr_url, marker) is not None


def post_review_comment(
    pr_url: str,
    body: str,
    *,
    thread_id: str | None = None,
    client: httpx.Client | None = None,
    skip_if_exists: bool = True,
) -> PostResult:
    """Post a top-level discussion comment on the PR.

    Not a formal GitHub review and not a merge — see the module docstring.

    Returns a :class:`PostResult`; check ``.ok`` / ``.unconfirmed``. The comment is
    never re-POSTed to resolve an ambiguous outcome:

    1. When ``thread_id`` is given, a deterministic HTML marker is appended and
       every page of the PR's comments is scanned first — if the marker is already
       there the call is a no-op (``already_posted``).
    2. A POST is attempted exactly **once**. Definitive failures (401/403/404/429/
       422) raise the matching :class:`GitHubError`.
    3. An **ambiguous** outcome (timeout, connection error, 5xx — GitHub may have
       created the comment) is resolved by re-scanning for the marker:
       found → ``already_posted``; not found → ``unconfirmed``. It is not retried
       and it is not reported as either success or failure.
    """
    owner, repo, number = parse_pr_url(pr_url)
    url = _comments_url(pr_url)

    marker: str | None = comment_marker(thread_id) if thread_id else None
    payload_body = body
    if marker and marker not in payload_body:
        payload_body = f"{body}\n\n{marker}"

    def _verify_ambiguous(
        client: httpx.Client, exc: Exception, attempts: int
    ) -> PostResult:
        """Never guess: look for the marker, then report unconfirmed if unsure."""
        if not marker:
            return PostResult(
                PostStatus.UNCONFIRMED,
                attempts=attempts,
                detail=(
                    f"{exc} — the POST outcome is unknown and there is no thread "
                    "marker to verify it with; not retried"
                ),
            )
        try:
            found = _find_marker(client, pr_url, marker)
        except GitHubError as verify_exc:
            return PostResult(
                PostStatus.UNCONFIRMED,
                attempts=attempts,
                detail=f"{exc}; verification also failed ({verify_exc}); not retried",
            )
        if found:
            return PostResult(
                PostStatus.ALREADY_POSTED,
                comment_url=found.get("html_url"),
                attempts=attempts,
                detail=(
                    f"{exc} — but the thread marker is present, so the comment was "
                    "created; not retried"
                ),
                marker_found=True,
            )
        return PostResult(
            PostStatus.UNCONFIRMED,
            attempts=attempts,
            detail=(
                f"{exc} — the thread marker is absent afterwards, so whether GitHub "
                "applied the POST is unknown; not retried"
            ),
        )

    def _run(client: httpx.Client) -> PostResult:
        if marker and skip_if_exists:
            found = _find_marker(client, pr_url, marker)
            if found is not None:
                logger.info(
                    "comment for thread %s already present — skipping post", thread_id
                )
                return PostResult(
                    PostStatus.ALREADY_POSTED,
                    comment_url=found.get("html_url"),
                    attempts=0,
                    detail="the thread marker was already on the PR",
                    marker_found=True,
                )
        try:
            resp = _request(
                client, "POST", url, headers=_headers(), json={"body": payload_body}
            )
        except AMBIGUOUS_ERRORS as exc:
            return _verify_ambiguous(client, exc, attempts=1)
        return PostResult(
            PostStatus.POSTED,
            comment_url=resp.json().get("html_url", url),
            attempts=1,
            detail=f"GitHub confirmed the comment ({resp.status_code})",
        )

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
    "AMBIGUOUS_ERRORS",
    "MAX_PAGES",
    "PER_PAGE",
    "GitHubAuthError",
    "GitHubConnectionError",
    "GitHubError",
    "GitHubNotFoundError",
    "GitHubPermissionError",
    "GitHubRateLimitError",
    "GitHubServerError",
    "GitHubTimeoutError",
    "PostResult",
    "PostStatus",
    "PullRequest",
    "already_posted",
    "comment_marker",
    "current_user",
    "fetch_pr",
    "is_valid_pr_url",
    "parse_pr_url",
    "post_review_comment",
]
