"""All dataclasses and schemas used across the agent pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class AccountStatus(str, Enum):
    """Possible states of a user account."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class TransactionStatus(str, Enum):
    """Possible states of a transaction."""

    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    REFUNDED = "refunded"


# ---------------------------------------------------------------------------
# Domain dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UserDetails:
    """Represents a single user account record."""

    user_id: str
    email: str
    full_name: str
    account_status: AccountStatus
    account_type: str
    created_at: str
    risk_score: int
    phone: str = ""


@dataclass
class Transaction:
    """Represents a single financial transaction record."""

    transaction_id: str
    user_id: str
    amount: float
    currency: str
    status: TransactionStatus
    type: str
    merchant: str
    created_at: str
    failure_code: Optional[str]


@dataclass
class FraudReason:
    """Detailed fraud/rejection analysis for a failed transaction."""

    transaction_id: str
    failure_code: str
    reason_tr: str
    reason_en: str
    risk_signals: List[str]
    can_dispute: bool
    resolution_steps: List[str]
    detected_at: str
    reviewed_by: str


# ---------------------------------------------------------------------------
# Tool execution models
# ---------------------------------------------------------------------------

class ToolStatus(str, Enum):
    """Execution outcome of a single tool call."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ToolCall:
    """Record of a single tool invocation including its result."""

    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    status: ToolStatus = ToolStatus.SUCCESS
    duration_ms: float = 0.0
    attempt_count: int = 1


@dataclass
class PlanStep:
    """A single step in the execution plan.

    Parameters may contain ``$ref:step_id.field`` placeholders that the
    executor resolves at runtime using the output of a previous step.
    """

    step_id: str
    tool_name: str
    parameters: Dict[str, Any]
    depends_on: List[str]
    can_run_parallel: bool = False


@dataclass
class ExecutionPlan:
    """The full plan produced by ToolPlanner."""

    steps: List[PlanStep]
    requires_user_input: bool = False
    missing_params: List[str] = field(default_factory=list)
    clarification_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Intent classification models
# ---------------------------------------------------------------------------

@dataclass
class IntentClassification:
    """Result of classifying a user message via the IntentClassifier."""

    intent: str  # "fraud_inquiry" | "transaction_history" | "account_info" | "general" | "other"
    provided_params: Dict[str, Optional[str]]
    missing_required_params: List[str]
    can_proceed: bool
    clarification_needed: Optional[str]
    reasoning: str


# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a tool's output against expected schema."""

    valid: bool
    error: Optional[str] = None
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent-level models
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Final response returned by the Agent to the caller."""

    answer: str
    tool_calls: List[ToolCall]
    iterations: int
    total_latency_ms: float
    requires_followup: bool = False
    followup_question: Optional[str] = None
