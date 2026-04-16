"""Execution plan builder — constructs a dependency-aware DAG of tool steps.

The planner analyses what information is already known and what the user's
intent requires, then produces the minimal ordered list of tool calls needed
to fulfil the request.

Dependency rules
----------------
* ``get_recent_transactions`` depends on ``get_user_details`` unless
  ``user_id`` is already known from conversation context.
* ``check_fraud_reason`` depends on ``get_recent_transactions`` (to find the
  first failed transaction) unless ``transaction_id`` is already known.
* If ``transaction_id`` is known upfront, the entire chain collapses to a
  single ``check_fraud_reason`` step.
* Independent steps (none in the current three-tool set) carry
  ``can_run_parallel=True`` so the executor may run them concurrently.
"""

from typing import Any, Dict, Optional

from src.logger import log
from src.models import ExecutionPlan, IntentClassification, PlanStep


class ToolPlanner:
    """Builds an :class:`~src.models.ExecutionPlan` from a classification result.

    The planner is pure Python — it makes no external calls.
    """

    def build_plan(
        self,
        classification: IntentClassification,
        conversation_context: Dict[str, Any],
    ) -> ExecutionPlan:
        """Produce the execution plan for the current turn.

        Args:
            classification: Result from :class:`~src.intent_classifier.IntentClassifier`.
            conversation_context: Params already known from prior turns
                (keys: ``email``, ``user_id``, ``transaction_id``, etc.).

        Returns:
            :class:`~src.models.ExecutionPlan` with ordered steps, or a plan
            that signals the agent to ask the user for missing information.
        """
        if not classification.can_proceed:
            return ExecutionPlan(
                steps=[],
                requires_user_input=True,
                missing_params=classification.missing_required_params,
                clarification_message=classification.clarification_needed,
            )

        # Merge conversation context with params extracted from the current message
        known: Dict[str, Optional[str]] = {
            **conversation_context,
            **{k: v for k, v in classification.provided_params.items() if v is not None},
        }

        intent = classification.intent
        steps = []
        counter = 0

        # ------------------------------------------------------------------
        # Fast path: transaction_id is already known → single step
        # ------------------------------------------------------------------
        if intent == "fraud_inquiry" and known.get("transaction_id"):
            counter += 1
            steps.append(
                PlanStep(
                    step_id=f"step_{counter}",
                    tool_name="check_fraud_reason",
                    parameters={"transaction_id": known["transaction_id"]},
                    depends_on=[],
                    can_run_parallel=False,
                )
            )
            log.debug(f"Plan: fast-path fraud lookup for {known['transaction_id']}")
            return ExecutionPlan(steps=steps)

        # ------------------------------------------------------------------
        # Standard path: start from email if user_id not known
        # ------------------------------------------------------------------
        needs_user_lookup = (
            intent in ("fraud_inquiry", "transaction_history", "account_info", "general")
            and not known.get("user_id")
            and known.get("email")
        )
        needs_transactions = intent in ("fraud_inquiry", "transaction_history", "general")
        needs_fraud = intent == "fraud_inquiry"

        user_step_id: Optional[str] = None
        txn_step_id: Optional[str] = None

        # Step A: get_user_details
        if needs_user_lookup:
            counter += 1
            user_step_id = f"step_{counter}"
            steps.append(
                PlanStep(
                    step_id=user_step_id,
                    tool_name="get_user_details",
                    parameters={"email": known["email"]},
                    depends_on=[],
                    can_run_parallel=False,
                )
            )

        # For account_info intent, we only need user details — no transactions
        if intent == "account_info" and not needs_transactions:
            log.debug(f"Plan: account_info only, {len(steps)} step(s)")
            return ExecutionPlan(steps=steps)

        # Step B: get_recent_transactions
        if needs_transactions:
            counter += 1
            txn_step_id = f"step_{counter}"
            if known.get("user_id"):
                user_id_param: Any = known["user_id"]
                deps = []
            else:
                user_id_param = f"$ref:{user_step_id}.user_id"
                deps = [user_step_id] if user_step_id else []

            steps.append(
                PlanStep(
                    step_id=txn_step_id,
                    tool_name="get_recent_transactions",
                    parameters={"user_id": user_id_param, "limit": 5},
                    depends_on=deps,
                    can_run_parallel=False,
                )
            )

        # Step C: check_fraud_reason
        if needs_fraud:
            counter += 1
            fraud_step_id = f"step_{counter}"
            if known.get("transaction_id"):
                txn_id_param: Any = known["transaction_id"]
                deps = []
            else:
                txn_id_param = f"$ref:{txn_step_id}.first_failed"
                deps = [txn_step_id] if txn_step_id else []

            steps.append(
                PlanStep(
                    step_id=fraud_step_id,
                    tool_name="check_fraud_reason",
                    parameters={"transaction_id": txn_id_param},
                    depends_on=deps,
                    can_run_parallel=False,
                )
            )

        # "other" intent — no steps needed; synthesizer will handle it
        if intent == "other" or not steps:
            log.debug("Plan: no tool steps required (intent=other or nothing to do)")
            return ExecutionPlan(steps=[])

        log.debug(f"Plan built: {[s.tool_name for s in steps]}")
        return ExecutionPlan(steps=steps)
