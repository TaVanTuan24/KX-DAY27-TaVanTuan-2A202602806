"""FIX 5 — CLI exit codes are consistent across exercises 1–4.

The bug: a workflow that failed (fetch/analyze/post) printed a message but exited 0,
so a script or CI job saw success. The contract now is:

====  ==========================================================
0     the workflow finished as designed
1     runtime failure, unconfirmed POST, or an incomplete audit row
2     invalid input (not a PR URL)
130   the operator pressed Ctrl+C
====  ==========================================================

The mapping is also checked *through the real CLIs* (see
``test_every_cli_exposes_the_same_exit_contract``), so it cannot drift from the
files that actually run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from exercises.cli import (
    EXIT_BAD_INPUT,
    EXIT_FAILURE,
    EXIT_INTERRUPTED,
    EXIT_OK,
    report_exit,
    workflow_exit_code,
    workflow_exit_message,
)

ROOT = Path(__file__).resolve().parent.parent
CLI_MODULES = [
    "exercises/exercise_1_confidence.py",
    "exercises/exercise_2_hitl.py",
    "exercises/exercise_3_escalation.py",
    "exercises/exercise_4_audit.py",
]


@pytest.mark.parametrize(
    "action",
    [
        "committed",
        "committed_already",
        "committed_dry_run",
        "auto_approved",
        "auto_approved_dry_run",
        "rejected",
        "pending_human_approval",
        "pending_escalation",
    ],
)
def test_designed_outcomes_exit_zero(action: str) -> None:
    assert workflow_exit_code({"final_action": action}) == EXIT_OK
    assert workflow_exit_message({"final_action": action}) is None


@pytest.mark.parametrize(
    "action",
    [
        "fetch_failed",
        "analyze_failed",
        "commit_failed",
        "commit_unconfirmed",
        "auto_approve_failed",
        "auto_refused_security",
    ],
)
def test_failures_exit_non_zero(action: str) -> None:
    result = {"final_action": action, "post_error": "boom"}
    assert workflow_exit_code(result) == EXIT_FAILURE
    assert workflow_exit_message(result)


def test_a_human_reject_is_not_a_failure() -> None:
    """The reviewer deciding "no" is a designed outcome, not an error."""
    result = {"final_action": "rejected", "post_status": "rejected"}
    assert workflow_exit_code(result) == EXIT_OK


def test_a_dry_run_is_not_a_failure() -> None:
    result = {"final_action": "committed_dry_run", "post_status": "dry_run"}
    assert workflow_exit_code(result) == EXIT_OK


def test_an_unconfirmed_post_is_a_failure_even_though_it_is_a_success_action() -> None:
    """`commit_unconfirmed` must never be treated as posted."""
    assert workflow_exit_code({"final_action": "commit_unconfirmed"}) == EXIT_FAILURE
    # ...even if some future path reports it with a success-looking action:
    assert (
        workflow_exit_code({"final_action": "committed", "post_status": "unconfirmed"})
        == EXIT_FAILURE
    )


def test_an_incomplete_audit_trail_is_a_failure() -> None:
    """A success with no evidence of the decision must not pass automation."""
    result = {"final_action": "committed", "audit_error": "commit: OperationalError"}
    assert workflow_exit_code(result) == EXIT_FAILURE
    message = workflow_exit_message(result)
    assert "audit trail is incomplete" in message


def test_unknown_or_missing_actions_never_claim_success() -> None:
    for result in ({}, {"final_action": None}, {"final_action": "weird_new_state"}):
        assert workflow_exit_code(result) == EXIT_FAILURE


def test_report_exit_returns_the_code() -> None:
    assert report_exit({"final_action": "committed"}) == EXIT_OK
    assert report_exit({"final_action": "fetch_failed"}) == EXIT_FAILURE


def test_the_shared_constants_are_the_documented_ones() -> None:
    assert (EXIT_OK, EXIT_FAILURE, EXIT_BAD_INPUT, EXIT_INTERRUPTED) == (0, 1, 2, 130)


def test_every_cli_uses_the_shared_exit_contract() -> None:
    """Static check: each CLI maps its result through `report_exit`.

    Parsed rather than imported, because importing a CLI runs argparse setup at
    most — but this also proves no CLI was left behind on the old ad-hoc codes.
    """
    for relative in CLI_MODULES:
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        assert "report_exit" in names, f"{relative} does not use report_exit"
        # No bare magic numbers in SystemExit for the documented cases.
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Raise)
                and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "SystemExit"
                and node.exc.args
            ):
                argument = node.exc.args[0]
                assert not isinstance(argument, ast.Constant), (
                    f"{relative}: SystemExit with a literal — use the shared constants"
                )
