"""Tool definitions in Anthropic's tool_use format and Python function map.

Import ``TOOL_DEFINITIONS`` when constructing the Anthropic API request.
Import ``TOOL_FUNCTION_MAP`` when dispatching tool calls to Python functions.
"""

from typing import Any, Callable, Dict, List

from src.tools import check_fraud_reason, get_recent_transactions, get_user_details


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "get_user_details",
        "description": (
            "Look up a user account by email address. "
            "Returns user_id, account_status, account_type, and other account metadata. "
            "MUST be called before get_recent_transactions if only email is known. "
            "Raises an error if the email is not registered in the system."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The user's email address. Example: ali@sirket.com",
                }
            },
            "required": ["email"],
        },
    },
    {
        "name": "get_recent_transactions",
        "description": (
            "Retrieve the most recent transactions for a user by their user_id. "
            "Returns transaction list sorted newest-first. "
            "Use limit=5 for recent activity checks, limit=20 for broader history. "
            "Requires user_id (obtained from get_user_details), NOT email."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": (
                        "Internal user ID in format USR-XXX. "
                        "Obtain this from get_user_details first."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of transactions to return. Default: 10. Range: 1-50.",
                    "default": 10,
                },
            },
            "required": ["user_id"],
        },
    },
    {
        "name": "check_fraud_reason",
        "description": (
            "Get the detailed rejection/fraud reason for a failed transaction. "
            "Only call this for transactions where status is 'failed'. "
            "Returns human-readable explanation, risk signals, and resolution steps. "
            "Raises an error if no fraud record exists (e.g. transaction was successful)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "string",
                    "description": (
                        "Transaction ID in format TXN-XXXXX. "
                        "Obtain this from get_recent_transactions."
                    ),
                }
            },
            "required": ["transaction_id"],
        },
    },
]

# Maps tool name → callable Python function for the executor.
TOOL_FUNCTION_MAP: Dict[str, Callable] = {
    "get_user_details": get_user_details,
    "get_recent_transactions": get_recent_transactions,
    "check_fraud_reason": check_fraud_reason,
}
