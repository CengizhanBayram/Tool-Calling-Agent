"""Unit tests for ToolExecutor — dependency resolution, retry, error isolation."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.models import (
    AccountStatus,
    ExecutionPlan,
    FraudReason,
    PlanStep,
    ToolCall,
    ToolStatus,
    Transaction,
    TransactionStatus,
    UserDetails,
)
from src.tool_executor import ToolExecutor
from src.tools import InvalidParameterError, TransactionNotFoundError, UserNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user_details_result(user_id: str = "USR-001") -> UserDetails:
    return UserDetails(
        user_id=user_id,
        email="test@example.com",
        full_name="Test User",
        account_status=AccountStatus.ACTIVE,
        account_type="premium",
        created_at="2022-01-01",
        risk_score=10,
    )


def _transaction_list(include_failed: bool = True):
    txns = [
        Transaction(
            transaction_id="TXN-10042",
            user_id="USR-001",
            amount=299.0,
            currency="TRY",
            status=TransactionStatus.SUCCESS,
            type="subscription",
            merchant="Netflix",
            created_at="2024-11-13T10:05:00",
            failure_code=None,
        )
    ]
    if include_failed:
        txns.insert(0, Transaction(
            transaction_id="TXN-10041",
            user_id="USR-001",
            amount=4500.0,
            currency="TRY",
            status=TransactionStatus.FAILED,
            type="payment",
            merchant="TechStore",
            created_at="2024-11-14T22:31:00",
            failure_code="FRAUD_SUSPECTED",
        ))
    return txns


def _fraud_reason() -> FraudReason:
    return FraudReason(
        transaction_id="TXN-10041",
        failure_code="FRAUD_SUSPECTED",
        reason_tr="Şüpheli işlem.",
        reason_en="Suspicious transaction.",
        risk_signals=["unusual_location"],
        can_dispute=True,
        resolution_steps=["Müşteri hizmetlerini arayın."],
        detected_at="2024-11-14T22:31:05",
        reviewed_by="AutoFraudEngine",
    )


# ---------------------------------------------------------------------------
# Single step execution
# ---------------------------------------------------------------------------

class TestSingleStepExecution:
    def test_successful_single_step(self):
        plan = ExecutionPlan(steps=[
            PlanStep("step_1", "get_user_details", {"email": "ali@sirket.com"}, [])
        ])
        executor = ToolExecutor()

        with patch.dict("src.tool_executor.TOOL_FUNCTION_MAP", {
            "get_user_details": lambda email: _user_details_result()
        }):
            results = executor.execute_plan(plan)

        assert "step_1" in results
        assert results["step_1"].status == ToolStatus.SUCCESS
        assert results["step_1"].result.user_id == "USR-001"

    def test_domain_error_captured_as_error_call(self):
        plan = ExecutionPlan(steps=[
            PlanStep("step_1", "get_user_details", {"email": "missing@email.com"}, [])
        ])
        executor = ToolExecutor()

        with patch.dict("src.tool_executor.TOOL_FUNCTION_MAP", {
            "get_user_details": lambda email: (_ for _ in ()).throw(
                UserNotFoundError("Email not found")
            )
        }):
            results = executor.execute_plan(plan)

        assert results["step_1"].status == ToolStatus.ERROR
        assert "Email not found" in results["step_1"].error

    def test_empty_plan_returns_empty_dict(self):
        plan = ExecutionPlan(steps=[])
        executor = ToolExecutor()
        assert executor.execute_plan(plan) == {}

    def test_user_input_plan_returns_empty_dict(self):
        plan = ExecutionPlan(steps=[], requires_user_input=True)
        executor = ToolExecutor()
        assert executor.execute_plan(plan) == {}


# ---------------------------------------------------------------------------
# Dependency resolution and $ref
# ---------------------------------------------------------------------------

class TestDependencyResolution:
    def test_ref_resolves_user_id_from_previous_step(self):
        plan = ExecutionPlan(steps=[
            PlanStep("step_1", "get_user_details", {"email": "ali@sirket.com"}, []),
            PlanStep(
                "step_2",
                "get_recent_transactions",
                {"user_id": "$ref:step_1.user_id", "limit": 5},
                ["step_1"],
            ),
        ])
        executor = ToolExecutor()

        call_log = []

        def fake_get_user_details(email):
            call_log.append(("get_user_details", email))
            return _user_details_result("USR-001")

        def fake_get_transactions(user_id, limit=10):
            call_log.append(("get_recent_transactions", user_id))
            assert user_id == "USR-001"
            return _transaction_list()

        with patch.dict("src.tool_executor.TOOL_FUNCTION_MAP", {
            "get_user_details": fake_get_user_details,
            "get_recent_transactions": fake_get_transactions,
        }):
            results = executor.execute_plan(plan)

        assert results["step_1"].status == ToolStatus.SUCCESS
        assert results["step_2"].status == ToolStatus.SUCCESS
        assert ("get_recent_transactions", "USR-001") in call_log

    def test_first_failed_ref_extracts_failed_transaction(self):
        executor = ToolExecutor()
        txns = _transaction_list(include_failed=True)
        field_val = executor._extract_field(txns, "first_failed")
        assert field_val == "TXN-10041"

    def test_first_failed_raises_when_no_failed_transactions(self):
        executor = ToolExecutor()
        txns = _transaction_list(include_failed=False)
        with pytest.raises(TransactionNotFoundError):
            executor._extract_field(txns, "first_failed")

    def test_dependent_step_skipped_when_upstream_fails(self):
        plan = ExecutionPlan(steps=[
            PlanStep("step_1", "get_user_details", {"email": "bad@email.com"}, []),
            PlanStep("step_2", "get_recent_transactions", {"user_id": "$ref:step_1.user_id", "limit": 5}, ["step_1"]),
            PlanStep("step_3", "check_fraud_reason", {"transaction_id": "$ref:step_2.first_failed"}, ["step_2"]),
        ])
        executor = ToolExecutor()

        with patch.dict("src.tool_executor.TOOL_FUNCTION_MAP", {
            "get_user_details": lambda email: (_ for _ in ()).throw(
                UserNotFoundError("not found")
            ),
            "get_recent_transactions": lambda **kw: _transaction_list(),
            "check_fraud_reason": lambda transaction_id: _fraud_reason(),
        }):
            results = executor.execute_plan(plan)

        assert results["step_1"].status == ToolStatus.ERROR
        assert results["step_2"].status == ToolStatus.SKIPPED
        assert results["step_3"].status == ToolStatus.SKIPPED


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    def test_transient_error_retried(self):
        """ConnectionError should be retried; third attempt succeeds."""
        attempt_counter = {"n": 0}

        def flaky_tool(email):
            attempt_counter["n"] += 1
            if attempt_counter["n"] < 3:
                raise ConnectionError("timeout")
            return _user_details_result()

        plan = ExecutionPlan(steps=[
            PlanStep("step_1", "get_user_details", {"email": "ali@sirket.com"}, [])
        ])
        executor = ToolExecutor()

        with patch.dict("src.tool_executor.TOOL_FUNCTION_MAP", {"get_user_details": flaky_tool}):
            with patch("src.tool_executor.time.sleep"):  # skip actual sleep
                results = executor.execute_plan(plan)

        assert results["step_1"].status == ToolStatus.SUCCESS
        assert results["step_1"].attempt_count == 3

    def test_domain_error_not_retried(self):
        """UserNotFoundError should NOT be retried — return immediately as ERROR."""
        attempt_counter = {"n": 0}

        def domain_error_tool(email):
            attempt_counter["n"] += 1
            raise UserNotFoundError("not found")

        plan = ExecutionPlan(steps=[
            PlanStep("step_1", "get_user_details", {"email": "x@x.com"}, [])
        ])
        executor = ToolExecutor()

        with patch.dict("src.tool_executor.TOOL_FUNCTION_MAP", {"get_user_details": domain_error_tool}):
            results = executor.execute_plan(plan)

        assert results["step_1"].status == ToolStatus.ERROR
        assert attempt_counter["n"] == 1  # only one attempt
