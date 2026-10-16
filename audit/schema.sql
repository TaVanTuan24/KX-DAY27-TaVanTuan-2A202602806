-- Audit trail for the HITL PR-review agent (SQLite version).
--
-- LangGraph's SqliteSaver already persists checkpoints (and lets us resume
-- or `time-travel` to any earlier state) in its own tables. Those tables are
-- great for *resuming* a paused graph but they're a blob-of-state per step,
-- not a queryable structured log. This table is the structured log.
--
-- One row per meaningful decision/event in a review session. Fields are
-- first-class columns so auditors can query directly:
--
--   SELECT AVG(confidence) FROM audit_events WHERE decision = 'approve';
--   SELECT * FROM audit_events WHERE risk_level = 'high' AND decision = 'auto';
--
-- `status` is the lifecycle phase of the event (pending / waiting / responded /
-- done / failed) and, together with thread_id + action + decision + revision,
-- forms the UNIQUE idempotency key: LangGraph re-executes a node from the top
-- after an interrupt() resumes, and INSERT OR IGNORE on this key keeps the trail
-- free of duplicate rows.
--
-- `revision` is which review revision the event belongs to. Without it, a second
-- Edit round in the SAME thread (same action/status/decision as the first) would
-- be silently dropped as a "duplicate" — `revision` is what separates a replay
-- from a genuinely new round.

CREATE TABLE IF NOT EXISTS audit_events (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT     NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- Session context
    thread_id         TEXT     NOT NULL,
    pr_url            TEXT     NOT NULL,

    -- AuditEntry fields (common/schemas.py)
    agent_id          TEXT     NOT NULL,
    action            TEXT     NOT NULL,
    confidence        REAL     NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    risk_level        TEXT     NOT NULL CHECK (risk_level IN ('low', 'med', 'high')),
    reviewer_id       TEXT,
    decision          TEXT     NOT NULL,
    status            TEXT     NOT NULL DEFAULT 'pending',
    revision          INTEGER  NOT NULL DEFAULT 0,
    reason            TEXT,
    execution_time_ms INTEGER  NOT NULL CHECK (execution_time_ms >= 0)
);

CREATE INDEX IF NOT EXISTS audit_events_thread_idx   ON audit_events (thread_id, id);
CREATE INDEX IF NOT EXISTS audit_events_pr_idx       ON audit_events (pr_url, timestamp DESC);
CREATE INDEX IF NOT EXISTS audit_events_decision_idx ON audit_events (decision);
CREATE INDEX IF NOT EXISTS audit_events_risk_idx     ON audit_events (risk_level);

-- Idempotency key. Also (re)created by common/db.py:_ensure_idempotency_index for
-- databases created before the `revision` column existed.
CREATE UNIQUE INDEX IF NOT EXISTS audit_events_idempotency_idx
    ON audit_events (thread_id, action, status, decision, revision);
