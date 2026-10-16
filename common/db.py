"""SQLite helpers: connection + audit-event writer.

The `audit_events` table lives in the same .db file as the LangGraph
SQLite checkpointer tables — different schemas, same file.

`init_schema()` is idempotent and runs on the first connection, so students
don't need a separate setup step. It also *migrates* an older database that
predates the idempotency index (see `_ensure_idempotency_index`).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite

from common.schemas import AuditEntry, PRAnalysis, ReviewComment

logger = logging.getLogger(__name__)

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "audit" / "schema.sql"

#: Domain types the checkpointer is allowed to deserialize. LangGraph's msgpack
#: serializer warns (and a future version will refuse) for anything not on an
#: allowlist, and silently registering a broad allowlist would weaken the
#: checkpoint file as an attack surface — so we name exactly what we store.
CHECKPOINT_ALLOWED_MODULES: tuple[tuple[str, ...], ...] = (
    ("common.schemas", "PRAnalysis"),
    ("common.schemas", "ReviewComment"),
)


def db_path() -> str:
    """Return the SQLite file path (override with HITL_DB_PATH env var)."""
    return os.environ.get("HITL_DB_PATH", "hitl_audit.db")


async def _ensure_schema(conn: aiosqlite.Connection) -> None:
    async with conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'"
    ) as cur:
        if await cur.fetchone() is not None:
            return
    await conn.executescript(SCHEMA_FILE.read_text())
    await conn.commit()


#: One row per logical event. LangGraph re-executes a node from the top after an
#: `interrupt()` resumes, so a node's "before the interrupt" audit call runs
#: twice; this key plus INSERT OR IGNORE makes that write idempotent.
#:
#: `decision` participates in the key so that a *different* outcome (e.g. the
#: reviewer first approves, then rejects on a re-run) is still recorded, while an
#: identical repeat is dropped.
#:
#: `revision` participates so a *new Edit round* in the same thread is recorded:
#: without it, a second `human_approval`/`responded`/`edit` event would look like
#: a duplicate of the first and be dropped, losing a real human decision.
IDEMPOTENCY_COLUMNS = ("thread_id", "action", "status", "decision", "revision")

#: Columns that older databases may be missing, with the DDL to add them.
_MIGRATABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("status", "ALTER TABLE audit_events ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'"),
    ("revision", "ALTER TABLE audit_events ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"),
)


async def _ensure_idempotency_index(conn: aiosqlite.Connection) -> None:
    """Create the UNIQUE idempotency index, migrating pre-existing databases.

    Handles three generations of `audit_events`:

    1. the original schema (no ``status``, no ``revision``, no unique index);
    2. the schema with ``status`` but a 4-column index;
    3. the current 5-column index.

    A database created before an index existed can already contain duplicate rows
    (the very bug the index prevents). SQLite refuses to create a UNIQUE index over
    duplicates, so each duplicate group is first collapsed to its earliest row.

    Valid audit history is never deleted: only *exact* duplicates of the current
    key are collapsed, and the columns that distinguish a new Edit round
    (``revision``) are added before that comparison.
    """
    async with conn.execute("PRAGMA table_info(audit_events)") as cur:
        columns = {row[1] for row in await cur.fetchall()}

    for name, ddl in _MIGRATABLE_COLUMNS:
        if name not in columns:
            await conn.execute(ddl)
            await conn.commit()
            logger.info("audit_events: added `%s` column (schema migration)", name)

    if "status" not in columns:
        # Generation 1: `decision` played the role of today's `status`.
        await conn.execute("UPDATE audit_events SET status = decision WHERE status = 'pending'")
        await conn.commit()

    async with conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' "
        "AND name='audit_events_idempotency_idx'"
    ) as cur:
        row = await cur.fetchone()

    cols = ", ".join(IDEMPOTENCY_COLUMNS)
    if row is not None:
        existing = (row["sql"] or "") if isinstance(row, aiosqlite.Row) else row[0] or ""
        if all(column in existing for column in IDEMPOTENCY_COLUMNS):
            return  # already current
        # Wider key than before (e.g. missing `revision`) → rebuild it.
        await conn.execute("DROP INDEX IF EXISTS audit_events_idempotency_idx")
        await conn.commit()
        logger.info("audit_events: rebuilding the idempotency index on (%s)", cols)

    await conn.execute(
        f"""
        DELETE FROM audit_events
         WHERE id NOT IN (
               SELECT MIN(id) FROM audit_events GROUP BY {cols}
         )
        """
    )
    await conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS audit_events_idempotency_idx "
        f"ON audit_events ({cols})"
    )
    await conn.commit()
    logger.info("audit_events: idempotency index ready on (%s)", cols)


async def _prepare(conn: aiosqlite.Connection) -> None:
    await _ensure_schema(conn)
    await _ensure_idempotency_index(conn)


@asynccontextmanager
async def db_conn() -> AsyncIterator[aiosqlite.Connection]:
    """Open an aiosqlite connection, applying schema + migration if needed."""
    conn = await aiosqlite.connect(db_path())
    conn.row_factory = aiosqlite.Row
    try:
        await _prepare(conn)
        yield conn
    finally:
        await conn.close()


async def write_audit_event(
    *,
    thread_id: str,
    pr_url: str,
    entry: AuditEntry,
) -> bool:
    """Append one structured audit row — idempotent per idempotency key.

    `thread_id` and `pr_url` are session-context columns (used for grouping
    and filtering); all other fields come from the `AuditEntry` so they map
    1-to-1 with first-class SQL columns.

    The second and later calls with an *identical* key — (thread_id, action,
    status, decision, revision) — are ignored (``INSERT OR IGNORE``), which keeps
    the audit trail free of duplicates when a node containing ``interrupt()`` is
    re-executed after the graph resumes. A genuinely new Edit round carries a new
    ``revision``, so it is recorded instead of being mistaken for a duplicate.

    Returns True when the row was written, False when it was an ignored duplicate.
    """
    async with db_conn() as conn:
        cur = await conn.execute(
            """
            INSERT OR IGNORE INTO audit_events (
                timestamp, thread_id, pr_url,
                agent_id, action, confidence, risk_level,
                reviewer_id, decision, status, revision, reason, execution_time_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.timestamp.isoformat(), thread_id, pr_url,
                entry.agent_id, entry.action, entry.confidence, entry.risk_level,
                entry.reviewer_id, entry.decision, entry.status, entry.revision,
                entry.reason, entry.execution_time_ms,
            ),
        )
        written = cur.rowcount > 0
        await cur.close()
        await conn.commit()
    return written


async def replay_events(thread_id: str) -> list[dict[str, Any]]:
    """Return every event for a thread, ordered by time. Used by audit/replay.py."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT id, timestamp, agent_id, action, confidence, risk_level,
                   reviewer_id, decision, status, revision, reason, execution_time_ms
              FROM audit_events
             WHERE thread_id = ?
             ORDER BY id
            """,
            (thread_id,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


#: Worst-first ordering for risk levels. `MAX(risk_level)` over TEXT would
#: order alphabetically ("med" > "low" > "high"), which is wrong; this CASE
#: expression gives high > med > low.
RISK_RANK_SQL = """CASE MAX(
        CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END
    ) WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END"""


async def list_threads(limit: int = 25) -> list[dict[str, Any]]:
    """Recent review sessions with their worst risk and last event.

    Data-level helper shared by `audit.replay --list` and the Streamlit sidebar.
    """
    async with db_conn() as conn:
        async with conn.execute(
            f"""
            SELECT thread_id,
                   MAX(pr_url)    AS pr_url,
                   MIN(timestamp) AS started,
                   MAX(timestamp) AS last_event,
                   {RISK_RANK_SQL} AS worst_risk,
                   COUNT(*)       AS events,
                   (SELECT decision FROM audit_events e2
                     WHERE e2.thread_id = e.thread_id
                     ORDER BY e2.id DESC LIMIT 1) AS last_decision,
                   (SELECT action FROM audit_events e3
                     WHERE e3.thread_id = e.thread_id
                     ORDER BY e3.id DESC LIMIT 1) AS last_action
              FROM audit_events e
             GROUP BY thread_id
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


def checkpoint_serde():
    """Serializer for the LangGraph checkpointer, with an explicit allowlist."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_MODULES)


def memory_checkpointer():
    """In-process checkpointer using the same explicit allowlist.

    Exercise 2's skeleton asks for ``MemorySaver``; this is that, minus the
    permissive default serializer (which emits a warning today and will refuse
    unserializable types in a future LangGraph release).
    """
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver(serde=checkpoint_serde())


@asynccontextmanager
async def open_checkpointer(
    conn_string: str | None = None,
) -> AsyncIterator["Any"]:
    """Open a durable ``AsyncSqliteSaver`` over the audit database.

    Used by exercise 4 and by the Streamlit app, so both get the same explicit
    serialization allowlist. The connection is closed on exit, which is what
    keeps Streamlit's reruns from leaking handles or holding SQLite's lock.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    path = conn_string or db_path()
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        saver.serde = checkpoint_serde()
        await saver.setup()
        yield saver


# The allowlist must cover everything the graph stores in a checkpoint.
_STORED_TYPES = (PRAnalysis, ReviewComment)
assert all(  # defensive: keep the tuple above honest
    (t.__module__, t.__name__) in CHECKPOINT_ALLOWED_MODULES for t in _STORED_TYPES
)
