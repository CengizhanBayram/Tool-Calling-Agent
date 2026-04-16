"""Unit tests for the three core tool functions in src/tools.py."""

import pytest

from src.tools import (
    FraudRecordNotFoundError,
    InvalidParameterError,
    TransactionNotFoundError,
    UserNotFoundError,
    check_fraud_reason,
    get_recent_transactions,
    get_user_details,
)
from src.models import AccountStatus, FraudReason, Transaction, TransactionStatus, UserDetails


# ---------------------------------------------------------------------------
# get_user_details
# ---------------------------------------------------------------------------

class TestGetUserDetails:
    def test_valid_email_returns_user_details(self):
        result = get_user_details("ali@sirket.com")
        assert isinstance(result, UserDetails)
        assert result.user_id == "USR-001"
        assert result.full_name == "Ali Yılmaz"
        assert result.account_status == AccountStatus.ACTIVE

    def test_case_insensitive_email_lookup(self):
        result = get_user_details("ALI@SIRKET.COM")
        assert result.user_id == "USR-001"

    def test_email_with_leading_trailing_spaces(self):
        result = get_user_details("  ali@sirket.com  ")
        assert result.user_id == "USR-001"

    def test_suspended_account_still_returned(self):
        result = get_user_details("ayse@ornek.com")
        assert result.account_status == AccountStatus.SUSPENDED

    def test_unknown_email_raises_user_not_found(self):
        with pytest.raises(UserNotFoundError, match="bilinmeyen@yok.com"):
            get_user_details("bilinmeyen@yok.com")

    def test_invalid_email_format_raises_invalid_param(self):
        with pytest.raises(InvalidParameterError):
            get_user_details("gecersiz-email")

    def test_empty_string_raises_invalid_param(self):
        with pytest.raises(InvalidParameterError):
            get_user_details("")

    def test_email_without_tld_raises_invalid_param(self):
        with pytest.raises(InvalidParameterError):
            get_user_details("user@domain")

    def test_enterprise_user_returned(self):
        result = get_user_details("mehmet@kurumsal.com")
        assert result.account_type == "enterprise"
        assert result.user_id == "USR-003"


# ---------------------------------------------------------------------------
# get_recent_transactions
# ---------------------------------------------------------------------------

class TestGetRecentTransactions:
    def test_valid_user_returns_transactions(self):
        result = get_recent_transactions("USR-001")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(t, Transaction) for t in result)

    def test_transactions_sorted_newest_first(self):
        result = get_recent_transactions("USR-001", limit=10)
        dates = [t.created_at for t in result]
        assert dates == sorted(dates, reverse=True)

    def test_limit_is_respected(self):
        result = get_recent_transactions("USR-001", limit=1)
        assert len(result) == 1

    def test_limit_is_clamped_to_50(self):
        # Should not raise; limit gets clamped
        result = get_recent_transactions("USR-001", limit=999)
        assert len(result) <= 50

    def test_limit_minimum_is_1(self):
        result = get_recent_transactions("USR-001", limit=0)
        assert len(result) >= 1

    def test_invalid_user_id_format_raises(self):
        with pytest.raises(InvalidParameterError):
            get_recent_transactions("12345")

    def test_nonexistent_user_id_raises(self):
        with pytest.raises(UserNotFoundError):
            get_recent_transactions("USR-999")

    def test_failed_transactions_have_failure_code(self):
        result = get_recent_transactions("USR-001", limit=10)
        failed = [t for t in result if t.status == TransactionStatus.FAILED]
        for t in failed:
            assert t.failure_code is not None

    def test_successful_transactions_have_no_failure_code(self):
        result = get_recent_transactions("USR-001", limit=10)
        successful = [t for t in result if t.status == TransactionStatus.SUCCESS]
        for t in successful:
            assert t.failure_code is None


# ---------------------------------------------------------------------------
# check_fraud_reason
# ---------------------------------------------------------------------------

class TestCheckFraudReason:
    def test_valid_failed_txn_returns_fraud_reason(self):
        result = check_fraud_reason("TXN-10041")
        assert isinstance(result, FraudReason)
        assert result.failure_code == "FRAUD_SUSPECTED"
        assert result.reason_tr  # non-empty Turkish text
        assert isinstance(result.risk_signals, list)
        assert len(result.resolution_steps) >= 1

    def test_fraud_reason_has_all_required_fields(self):
        result = check_fraud_reason("TXN-10041")
        assert result.transaction_id == "TXN-10041"
        assert result.reason_en
        assert result.detected_at
        assert result.reviewed_by

    def test_can_dispute_is_bool(self):
        result = check_fraud_reason("TXN-10041")
        assert isinstance(result.can_dispute, bool)

    def test_successful_txn_raises_fraud_record_not_found(self):
        with pytest.raises(FraudRecordNotFoundError, match="başarılı"):
            check_fraud_reason("TXN-10042")  # TXN-10042 is a success

    def test_nonexistent_txn_id_raises_transaction_not_found(self):
        with pytest.raises(TransactionNotFoundError):
            check_fraud_reason("TXN-99999")

    def test_invalid_txn_id_format_raises_invalid_param(self):
        with pytest.raises(InvalidParameterError):
            check_fraud_reason("10041")

    def test_insufficient_funds_reason(self):
        result = check_fraud_reason("TXN-10056")
        assert result.failure_code == "INSUFFICIENT_FUNDS"

    def test_blocked_merchant_reason(self):
        result = check_fraud_reason("TXN-10055")
        assert result.failure_code == "BLOCKED_MERCHANT"

    def test_velocity_limit_exceeded(self):
        result = check_fraud_reason("TXN-10047")
        assert result.failure_code == "VELOCITY_LIMIT_EXCEEDED"

    def test_card_expired(self):
        result = check_fraud_reason("TXN-10049")
        assert result.failure_code == "CARD_EXPIRED"

    def test_account_suspended_reason(self):
        result = check_fraud_reason("TXN-10044")
        assert result.failure_code == "ACCOUNT_SUSPENDED"
