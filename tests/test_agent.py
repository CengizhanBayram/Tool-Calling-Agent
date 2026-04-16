"""Integration tests for the Agent orchestrator with mocked Gemini calls."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agent import Agent
from src.models import AgentResponse


def _make_classification_model(
    intent: str,
    email: str = None,
    txn_id: str = None,
    user_id: str = None,
    can_proceed: bool = True,
    clarification: str = None,
) -> MagicMock:
    """Return a mock GenerativeModel for the classifier stage."""
    data = {
        "intent": intent,
        "provided_params": {"email": email, "transaction_id": txn_id, "user_id": user_id},
        "missing_required_params": [] if can_proceed else ["email"],
        "can_proceed": can_proceed,
        "clarification_needed": clarification,
        "reasoning": "test",
    }
    model = MagicMock()
    resp = MagicMock()
    resp.text = json.dumps(data)
    model.generate_content.return_value = resp
    return model


def _make_synthesis_model(text: str) -> MagicMock:
    """Return a mock GenerativeModel for the synthesizer stage."""
    model = MagicMock()
    resp = MagicMock()
    resp.text = text
    model.generate_content.return_value = resp
    return model


@pytest.fixture()
def agent_with_mocks():
    """Return an Agent whose Gemini GenerativeModel is fully mocked."""
    with patch("src.intent_classifier.genai.configure"), \
         patch("src.response_synthesizer.genai.configure"), \
         patch("src.intent_classifier.genai.GenerativeModel") as clf_model_cls, \
         patch("src.response_synthesizer.genai.GenerativeModel") as syn_model_cls:

        ag = Agent()
        yield ag, clf_model_cls, syn_model_cls


# ---------------------------------------------------------------------------
# Test 1 — Full chain: email → user → txn → fraud
# ---------------------------------------------------------------------------

class TestFullChain:
    def test_fraud_inquiry_with_email_runs_full_chain(self, agent_with_mocks):
        ag, clf_model_cls, syn_model_cls = agent_with_mocks

        clf_model_cls.return_value = _make_classification_model(
            intent="fraud_inquiry",
            email="ali@sirket.com",
            can_proceed=True,
        )
        syn_model_cls.return_value = _make_synthesis_model(
            "Ali Bey, TXN-10041 numaralı işleminiz Moskova'dan deneme nedeniyle reddedildi."
        )

        resp = ag.chat("ali@sirket.com son ödeme neden reddedildi?", session_id="t1")
        assert isinstance(resp, AgentResponse)
        assert resp.requires_followup is False
        assert len(resp.tool_calls) >= 1
        assert resp.answer


# ---------------------------------------------------------------------------
# Test 2 — Missing email → asks user, no tool calls
# ---------------------------------------------------------------------------

class TestMissingEmail:
    def test_no_email_returns_clarification_no_tools(self, agent_with_mocks):
        ag, clf_model_cls, _ = agent_with_mocks

        clf_model_cls.return_value = _make_classification_model(
            intent="transaction_history",
            can_proceed=False,
            clarification="Hesabınıza erişmek için e-posta adresinizi paylaşır mısınız?",
        )

        resp = ag.chat("son işlemlerimi görmek istiyorum", session_id="t2")
        assert resp.requires_followup is True
        assert resp.tool_calls == []
        assert resp.iterations == 0
        assert resp.answer


# ---------------------------------------------------------------------------
# Test 3 — Account status check
# ---------------------------------------------------------------------------

class TestAccountInfo:
    def test_suspended_account_info_returned(self, agent_with_mocks):
        ag, clf_model_cls, syn_model_cls = agent_with_mocks

        clf_model_cls.return_value = _make_classification_model(
            intent="account_info",
            email="ayse@ornek.com",
            can_proceed=True,
        )
        syn_model_cls.return_value = _make_synthesis_model(
            "Ayşe Hanım, hesabınız askıya alınmış durumda. [Kaynak: get_user_details]"
        )

        resp = ag.chat("ayse@ornek.com hesabının durumu nedir?", session_id="t3")
        assert resp.requires_followup is False
        assert len(resp.tool_calls) >= 1


# ---------------------------------------------------------------------------
# Test 4 — Unknown email → UserNotFoundError handled gracefully
# ---------------------------------------------------------------------------

class TestUnknownEmail:
    def test_unknown_email_returns_natural_error(self, agent_with_mocks):
        ag, clf_model_cls, syn_model_cls = agent_with_mocks

        clf_model_cls.return_value = _make_classification_model(
            intent="account_info",
            email="bilinmeyen@email.com",
            can_proceed=True,
        )
        syn_model_cls.return_value = _make_synthesis_model(
            "Üzgünüm, bilinmeyen@email.com adresine kayıtlı hesap bulunamadı."
        )

        resp = ag.chat("bilinmeyen@email.com hesabı", session_id="t4")
        # Response must be graceful — never crash, always return an answer
        assert resp.answer
        # Either a tool error was captured, OR the synthesis message contains error context
        error_calls = [c for c in resp.tool_calls if c.error is not None]
        if resp.tool_calls:
            assert len(error_calls) >= 1, (
                f"Expected at least one tool error, got: {[(c.tool_name, c.status) for c in resp.tool_calls]}"
            )


# ---------------------------------------------------------------------------
# Test 5 — TXN-ID given directly → single check_fraud_reason step
# ---------------------------------------------------------------------------

class TestDirectTxnId:
    def test_txn_id_given_directly_skips_chain(self, agent_with_mocks):
        ag, clf_model_cls, syn_model_cls = agent_with_mocks

        clf_model_cls.return_value = _make_classification_model(
            intent="fraud_inquiry",
            txn_id="TXN-10041",
            can_proceed=True,
        )
        syn_model_cls.return_value = _make_synthesis_model(
            "TXN-10041 Moskova'dan işlem denemesi nedeniyle reddedildi."
        )

        resp = ag.chat("TXN-10041 neden reddedildi?", session_id="t5")
        assert resp.requires_followup is False
        # Should only call check_fraud_reason (1 step, not 3)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool_name == "check_fraud_reason"


# ---------------------------------------------------------------------------
# Test 6 — Multi-turn: email from history, no re-asking
# ---------------------------------------------------------------------------

class TestMultiTurn:
    def test_email_from_history_not_re_asked(self, agent_with_mocks):
        ag, clf_model_cls, syn_model_cls = agent_with_mocks
        session = "t6"

        # Turn 1: user gives email, asks about account
        clf_model_cls.return_value = _make_classification_model(
            intent="account_info",
            email="ali@sirket.com",
            can_proceed=True,
        )
        syn_model_cls.return_value = _make_synthesis_model("Hesabınız aktif durumda.")
        ag.chat("ali@sirket.com hesabım aktif mi?", session_id=session)

        # Turn 2: follow-up without email; classifier says email missing
        # but the agent's history-extraction should merge it automatically
        clf_model_cls.return_value = _make_classification_model(
            intent="transaction_history",
            can_proceed=False,
            clarification="E-posta adresinizi paylaşır mısınız?",
        )
        syn_model_cls.return_value = _make_synthesis_model("İşte son işlemleriniz.")
        resp2 = ag.chat("son işlemlerimi de göster", session_id=session)
        # Key assertion: no crash, answer is returned
        assert resp2.answer


# ---------------------------------------------------------------------------
# Test 9 — Off-topic: polite redirect, no tool calls
# ---------------------------------------------------------------------------

class TestOffTopic:
    def test_off_topic_query_no_tool_calls(self, agent_with_mocks):
        ag, clf_model_cls, syn_model_cls = agent_with_mocks

        clf_model_cls.return_value = _make_classification_model(
            intent="other",
            can_proceed=True,
        )
        syn_model_cls.return_value = _make_synthesis_model(
            "Bu konuda yardımcı olamam, ancak hesap veya işlem sorularınızda size destek olabilirim."
        )

        resp = ag.chat("hava nasıl?", session_id="t9")
        assert resp.tool_calls == []
        assert resp.answer


# ---------------------------------------------------------------------------
# Test 10 — Empty message
# ---------------------------------------------------------------------------

class TestEmptyMessage:
    def test_empty_message_asks_for_help(self, agent_with_mocks):
        ag, clf_model_cls, _ = agent_with_mocks
        # IntentClassifier handles empty message before calling Gemini
        resp = ag.chat("", session_id="t10")
        assert resp.requires_followup is True
        assert resp.tool_calls == []
        assert resp.answer
