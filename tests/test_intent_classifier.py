"""Unit tests for IntentClassifier — uses mocked Gemini responses."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.intent_classifier import IntentClassifier
from src.models import IntentClassification


def _make_gemini_response(data: dict) -> Any:
    """Build a mock Gemini response object returning JSON *data*."""
    response = MagicMock()
    response.text = json.dumps(data)
    return response


def _mock_model(response_data: dict):
    """Return a mock GenerativeModel whose generate_content returns *response_data*."""
    model = MagicMock()
    model.generate_content.return_value = _make_gemini_response(response_data)
    return model


@pytest.fixture()
def classifier():
    """Return an IntentClassifier with a mocked Gemini GenerativeModel."""
    with patch("src.intent_classifier.genai.configure"), \
         patch("src.intent_classifier.genai.GenerativeModel") as mock_model_cls:
        clf = IntentClassifier.__new__(IntentClassifier)
        clf._model_name = "gemini-2.5-flash-preview-04-17"
        clf._max_tokens = 400
        yield clf, mock_model_cls


class TestClassifyFraudInquiry:
    def test_fraud_inquiry_with_email(self, classifier):
        clf, mock_model_cls = classifier
        mock_model_cls.return_value = _mock_model({
            "intent": "fraud_inquiry",
            "provided_params": {"email": "ali@sirket.com", "transaction_id": None, "user_id": None},
            "missing_required_params": [],
            "can_proceed": True,
            "clarification_needed": None,
            "reasoning": "User provided email and asked about rejection.",
        })
        result = clf.classify("ali@sirket.com ödeme neden reddedildi?", [])
        assert result.intent == "fraud_inquiry"
        assert result.can_proceed is True
        assert result.provided_params.get("email") == "ali@sirket.com"

    def test_fraud_inquiry_with_txn_id(self, classifier):
        clf, mock_model_cls = classifier
        mock_model_cls.return_value = _mock_model({
            "intent": "fraud_inquiry",
            "provided_params": {"email": None, "transaction_id": "TXN-10041", "user_id": None},
            "missing_required_params": [],
            "can_proceed": True,
            "clarification_needed": None,
            "reasoning": "Transaction ID provided directly.",
        })
        result = clf.classify("TXN-10041 neden reddedildi?", [])
        assert result.can_proceed is True
        assert result.provided_params.get("transaction_id") == "TXN-10041"

    def test_fraud_inquiry_no_identifiers_asks_for_email(self, classifier):
        clf, mock_model_cls = classifier
        mock_model_cls.return_value = _mock_model({
            "intent": "fraud_inquiry",
            "provided_params": {"email": None, "transaction_id": None, "user_id": None},
            "missing_required_params": ["email"],
            "can_proceed": False,
            "clarification_needed": "Hesabınıza erişmek için e-posta adresinizi paylaşır mısınız?",
            "reasoning": "No identifier provided.",
        })
        result = clf.classify("son ödeme neden reddedildi?", [])
        assert result.can_proceed is False
        assert "email" in result.missing_required_params
        assert result.clarification_needed is not None


class TestClassifyTransactionHistory:
    def test_transaction_history_without_email_asks(self, classifier):
        clf, mock_model_cls = classifier
        mock_model_cls.return_value = _mock_model({
            "intent": "transaction_history",
            "provided_params": {"email": None, "transaction_id": None, "user_id": None},
            "missing_required_params": ["email"],
            "can_proceed": False,
            "clarification_needed": "Son işlemlerinizi görmek için e-posta adresinizi paylaşır mısınız?",
            "reasoning": "No email given.",
        })
        result = clf.classify("son işlemlerimi görmek istiyorum", [])
        assert result.can_proceed is False
        assert result.intent == "transaction_history"

    def test_transaction_history_email_from_history(self, classifier):
        clf, mock_model_cls = classifier
        mock_model_cls.return_value = _mock_model({
            "intent": "transaction_history",
            "provided_params": {"email": None, "transaction_id": None, "user_id": None},
            "missing_required_params": ["email"],
            "can_proceed": False,
            "clarification_needed": "E-posta adresinizi paylaşır mısınız?",
            "reasoning": "No email in current message.",
        })
        history = [{"role": "user", "content": "ali@sirket.com hesabım hakkında bilgi almak istiyorum"}]
        result = clf.classify("son işlemlerimi göster", history)
        # Email from history should be merged → can_proceed becomes True
        assert result.can_proceed is True
        assert result.provided_params.get("email") == "ali@sirket.com"


class TestClassifyOtherIntent:
    def test_off_topic_message(self, classifier):
        clf, mock_model_cls = classifier
        mock_model_cls.return_value = _mock_model({
            "intent": "other",
            "provided_params": {"email": None, "transaction_id": None, "user_id": None},
            "missing_required_params": [],
            "can_proceed": True,
            "clarification_needed": None,
            "reasoning": "Completely unrelated question.",
        })
        result = clf.classify("hava nasıl?", [])
        assert result.intent == "other"
        assert result.can_proceed is True

    def test_empty_message_returns_safe_default(self, classifier):
        clf, _ = classifier
        result = clf.classify("", [])
        assert result.can_proceed is False
        assert result.clarification_needed is not None


class TestHistoryExtraction:
    def test_extracts_email_from_history(self):
        with patch("src.intent_classifier.genai.configure"):
            clf = IntentClassifier.__new__(IntentClassifier)
            clf._model_name = "test"
            clf._max_tokens = 400
        history = [
            {"role": "user", "content": "Merhaba, emre@tech.io ile kayıtlıyım."},
            {"role": "assistant", "content": "Merhaba, nasıl yardımcı olabilirim?"},
        ]
        params = clf._extract_params_from_history(history)
        assert params["email"] == "emre@tech.io"

    def test_extracts_user_id_from_history(self):
        with patch("src.intent_classifier.genai.configure"):
            clf = IntentClassifier.__new__(IntentClassifier)
            clf._model_name = "test"
            clf._max_tokens = 400
        history = [{"role": "assistant", "content": "Kullanıcı ID: USR-005 bulundu."}]
        params = clf._extract_params_from_history(history)
        assert params["user_id"] == "USR-005"

    def test_extracts_txn_id_from_history(self):
        with patch("src.intent_classifier.genai.configure"):
            clf = IntentClassifier.__new__(IntentClassifier)
            clf._model_name = "test"
            clf._max_tokens = 400
        history = [{"role": "user", "content": "TXN-10041 hakkında bilgi almak istiyorum."}]
        params = clf._extract_params_from_history(history)
        assert params["transaction_id"] == "TXN-10041"
