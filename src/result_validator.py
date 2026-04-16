"""Validates tool outputs against expected schemas before sending to Claude.

Catches unexpected formats, missing required fields, and empty results so
the synthesiser receives only well-formed data.
"""

from typing import Any, Dict, List

from src.logger import log
from src.models import (
    FraudReason,
    ToolCall,
    ToolStatus,
    Transaction,
    UserDetails,
    ValidationResult,
)


class ResultValidator:
    """Validates the output of each successfully-executed tool call.

    Only ``ToolStatus.SUCCESS`` calls are validated; error / skipped calls
    pass through without validation.
    """

    def validate_user_details(self, result: Any) -> ValidationResult:
        """Validate a :class:`~src.models.UserDetails` result.

        Args:
            result: Value returned by ``get_user_details``.

        Returns:
            :class:`~src.models.ValidationResult` indicating whether the
            result is well-formed.
        """
        if not isinstance(result, UserDetails):
            return ValidationResult(
                valid=False,
                error=f"Expected UserDetails, got {type(result).__name__}",
            )
        if not result.user_id or not result.email:
            return ValidationResult(
                valid=False,
                error="UserDetails is missing required fields (user_id or email).",
            )
        if not result.account_status:
            return ValidationResult(
                valid=False,
                error="UserDetails is missing account_status.",
            )
        return ValidationResult(valid=True)

    def validate_transactions(self, result: Any) -> ValidationResult:
        """Validate a list of :class:`~src.models.Transaction` objects.

        Args:
            result: Value returned by ``get_recent_transactions``.

        Returns:
            :class:`~src.models.ValidationResult`. An empty list is valid
            (user may simply have no transactions) but triggers a warning.
        """
        if not isinstance(result, list):
            return ValidationResult(
                valid=False,
                error=f"Expected list[Transaction], got {type(result).__name__}",
            )
        for item in result:
            if not isinstance(item, Transaction):
                return ValidationResult(
                    valid=False,
                    error=f"List contains non-Transaction item: {type(item).__name__}",
                )
        warning = "Kullanıcının hiç işlemi bulunamadı." if not result else None
        return ValidationResult(valid=True, warning=warning)

    def validate_fraud_reason(self, result: Any) -> ValidationResult:
        """Validate a :class:`~src.models.FraudReason` result.

        Args:
            result: Value returned by ``check_fraud_reason``.

        Returns:
            :class:`~src.models.ValidationResult`.
        """
        if not isinstance(result, FraudReason):
            return ValidationResult(
                valid=False,
                error=f"Expected FraudReason, got {type(result).__name__}",
            )
        if not result.reason_tr:
            return ValidationResult(
                valid=False,
                error="FraudReason is missing the Turkish explanation (reason_tr).",
            )
        if not result.resolution_steps:
            return ValidationResult(
                valid=True,
                warning="FraudReason has no resolution steps listed.",
            )
        return ValidationResult(valid=True)

    def validate_all(
        self, tool_calls: Dict[str, ToolCall]
    ) -> Dict[str, ValidationResult]:
        """Validate all successful tool calls in a results dict.

        Args:
            tool_calls: Mapping of step_id → :class:`~src.models.ToolCall`
                as returned by :meth:`~src.tool_executor.ToolExecutor.execute_plan`.

        Returns:
            Mapping of step_id → :class:`~src.models.ValidationResult` for
            every step that was in ``SUCCESS`` status.
        """
        validators = {
            "get_user_details": self.validate_user_details,
            "get_recent_transactions": self.validate_transactions,
            "check_fraud_reason": self.validate_fraud_reason,
        }

        results: Dict[str, ValidationResult] = {}
        for step_id, call in tool_calls.items():
            if call.status != ToolStatus.SUCCESS:
                continue
            validator = validators.get(call.tool_name)
            if validator is None:
                continue
            vr = validator(call.result)
            if not vr.valid:
                log.warning(
                    f"Validation FAILED for {call.tool_name} (step {step_id}): {vr.error}"
                )
            elif vr.warning:
                log.warning(
                    f"Validation WARNING for {call.tool_name} (step {step_id}): {vr.warning}"
                )
            results[step_id] = vr

        return results
