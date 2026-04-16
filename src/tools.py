"""The three core tool functions callable by the agent executor.

Each function validates its inputs, queries the database, and returns a
typed dataclass on success.  Domain errors are raised as specific exception
types — never returned as ``None`` and never silently swallowed.
"""

import re
from typing import List

from src.database import db
from src.models import AccountStatus, FraudReason, Transaction, TransactionStatus, UserDetails


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------

class UserNotFoundError(Exception):
    """Raised when no user matches the given email or user_id."""


class UserSuspendedError(Exception):
    """Raised when the user's account is suspended and the operation is restricted."""


class TransactionNotFoundError(Exception):
    """Raised when no transaction matches the given transaction_id."""


class FraudRecordNotFoundError(Exception):
    """Raised when no fraud record exists for the given transaction."""


class InvalidParameterError(Exception):
    """Raised when a parameter fails format or value validation."""


# ---------------------------------------------------------------------------
# Tool 1 — get_user_details
# ---------------------------------------------------------------------------

def get_user_details(email: str) -> UserDetails:
    """Look up a user by email address and return their account details.

    Args:
        email: The user's email address. Must be a valid email format.

    Returns:
        :class:`~src.models.UserDetails` dataclass with user_id,
        account_status, account_type, and other account metadata.

    Raises:
        InvalidParameterError: If the email format is invalid.
        UserNotFoundError: If no user with this email exists in the system.
    """
    if not email or not isinstance(email, str):
        raise InvalidParameterError("E-posta adresi boş olamaz.")

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
        raise InvalidParameterError(
            f"'{email}' geçerli bir e-posta adresi değil. "
            f"Lütfen 'ornek@alan.com' formatında bir adres girin."
        )

    user_dict = db.find_user_by_email(email)
    if user_dict is None:
        raise UserNotFoundError(
            f"'{email}' e-posta adresine kayıtlı bir hesap bulunamadı. "
            f"E-posta adresini doğru yazdığınızdan emin olun."
        )

    return UserDetails(
        user_id=user_dict["user_id"],
        email=user_dict["email"],
        full_name=user_dict["full_name"],
        account_status=AccountStatus(user_dict["account_status"]),
        account_type=user_dict["account_type"],
        created_at=user_dict["created_at"],
        risk_score=int(user_dict["risk_score"]),
        phone=user_dict.get("phone", ""),
    )


# ---------------------------------------------------------------------------
# Tool 2 — get_recent_transactions
# ---------------------------------------------------------------------------

def get_recent_transactions(user_id: str, limit: int = 10) -> List[Transaction]:
    """Retrieve the most recent transactions for a given user.

    Args:
        user_id: The internal user ID (format: ``USR-XXX``).
        limit: Maximum number of transactions to return. Clamped to ``[1, 50]``.

    Returns:
        List of :class:`~src.models.Transaction` objects sorted by date
        descending (newest first). An empty list means the user has no
        transactions — this is NOT an error.

    Raises:
        InvalidParameterError: If user_id format is invalid.
        UserNotFoundError: If user_id does not exist in the system.
    """
    if not user_id or not isinstance(user_id, str) or not user_id.startswith("USR-"):
        raise InvalidParameterError(
            f"Geçersiz kullanıcı ID formatı: '{user_id}'. Beklenen format: USR-XXX"
        )

    limit = max(1, min(50, int(limit)))

    if db.find_user_by_id(user_id) is None:
        raise UserNotFoundError(
            f"Kullanıcı bulunamadı: '{user_id}'. "
            f"Önce get_user_details ile e-posta üzerinden kullanıcı sorgulayın."
        )

    txn_dicts = db.get_transactions_by_user(user_id, limit=limit)

    return [
        Transaction(
            transaction_id=t["transaction_id"],
            user_id=t["user_id"],
            amount=float(t["amount"]),
            currency=t["currency"],
            status=TransactionStatus(t["status"]),
            type=t["type"],
            merchant=t["merchant"],
            created_at=t["created_at"],
            failure_code=t.get("failure_code"),
        )
        for t in txn_dicts
    ]


# ---------------------------------------------------------------------------
# Tool 3 — check_fraud_reason
# ---------------------------------------------------------------------------

def check_fraud_reason(transaction_id: str) -> FraudReason:
    """Retrieve the fraud / rejection reason for a failed transaction.

    Args:
        transaction_id: The transaction ID (format: ``TXN-XXXXX``).

    Returns:
        :class:`~src.models.FraudReason` dataclass with human-readable
        explanation, risk signals, and resolution steps.

    Raises:
        InvalidParameterError: If transaction_id format is invalid.
        TransactionNotFoundError: If the transaction doesn't exist.
        FraudRecordNotFoundError: If the transaction exists but has no fraud
            record (e.g. it was a successful transaction).
    """
    if not transaction_id or not isinstance(transaction_id, str) or not transaction_id.startswith("TXN-"):
        raise InvalidParameterError(
            f"Geçersiz işlem ID formatı: '{transaction_id}'. Beklenen format: TXN-XXXXX"
        )

    txn = db.find_transaction_by_id(transaction_id)
    if txn is None:
        raise TransactionNotFoundError(
            f"'{transaction_id}' numaralı işlem sistemde bulunamadı."
        )

    fraud_dict = db.get_fraud_reason(transaction_id)
    if fraud_dict is None:
        status = txn.get("status", "unknown")
        if status == "success":
            reason = "Bu işlem başarılı şekilde tamamlandı; red kaydı bulunmuyor."
        else:
            reason = "Red detayı henüz sistemde kayıtlı değil."
        raise FraudRecordNotFoundError(
            f"'{transaction_id}' numaralı işlem için red kaydı bulunamadı. "
            f"İşlem durumu: {status}. {reason}"
        )

    return FraudReason(
        transaction_id=fraud_dict["transaction_id"],
        failure_code=fraud_dict["failure_code"],
        reason_tr=fraud_dict["reason_tr"],
        reason_en=fraud_dict["reason_en"],
        risk_signals=list(fraud_dict["risk_signals"]),
        can_dispute=bool(fraud_dict["can_dispute"]),
        resolution_steps=list(fraud_dict["resolution_steps"]),
        detected_at=fraud_dict["detected_at"],
        reviewed_by=fraud_dict["reviewed_by"],
    )
