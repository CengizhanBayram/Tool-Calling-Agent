"""Intent classifier — detects user intent and identifies missing parameters.

This module prevents the agent from hallucinating parameters by explicitly
asking the user for any required information that is absent from the current
message AND from the conversation history.
"""

import json
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from src.config import Settings, settings
from src.logger import log
from src.models import IntentClassification


CLASSIFICATION_SYSTEM_PROMPT = """You are an intent analysis assistant for a Turkish customer support agent.

Analyze the user message and conversation history context, then identify:
1. What does the user want?
2. What information is explicitly provided in the message or history?
3. What required information is MISSING?

Available tools and their required parameters:
- get_user_details: requires email address
- get_recent_transactions: requires user_id (can be derived from email via get_user_details)
- check_fraud_reason: requires transaction_id (can be derived from transaction list OR given directly)

INTENT CATEGORIES:
- "fraud_inquiry": user asking why a payment/transaction was rejected or blocked
- "transaction_history": user asking to see recent transactions or payment history
- "account_info": user asking about their account status, type, or details
- "general": user asking a general question that may need one of the above tools
- "other": completely unrelated question (weather, jokes, etc.) - no tools needed

CRITICAL RULES:
1. If intent is fraud_inquiry, transaction_history, or account_info:
   - can_proceed = true IF (email OR user_id) is provided OR transaction_id for fraud_inquiry
   - can_proceed = false IF none of the above identifiers are present
2. If intent is "other" or "general": can_proceed = true (no identifiers needed)
3. NEVER assume or fabricate an email address, user_id, or transaction_id.
4. Turkish phrases like "hesabım", "işlemim", "ödeme yaptım", "son işlemlerim" imply
   the user's OWN account — email is required unless already in history.
5. If transaction_id is explicitly given (e.g. "TXN-10041"), mark it as provided and
   can_proceed = true for fraud_inquiry WITHOUT needing email.
6. If email is in the conversation history context, treat it as provided.

If can_proceed is false, write a polite Turkish question in clarification_needed.
Example: "Hesabınıza erişmek için lütfen kayıtlı e-posta adresinizi paylaşır mısınız?"

Respond ONLY with valid JSON (no markdown fences, no extra text):
{
  "intent": "fraud_inquiry|transaction_history|account_info|general|other",
  "provided_params": {
    "email": "value or null",
    "transaction_id": "value or null",
    "user_id": "value or null"
  },
  "missing_required_params": ["email"],
  "can_proceed": true,
  "clarification_needed": "Turkish question string or null",
  "reasoning": "brief explanation in English"
}"""


class IntentClassifier:
    """Classifies user messages and detects missing required parameters.

    Uses a Gemini API call with a structured prompt to reliably parse
    intent and extract or identify missing parameters before any tools run.

    Args:
        cfg: Application settings. Defaults to the module-level singleton.
    """

    def __init__(self, cfg: Settings = settings) -> None:
        genai.configure(api_key=cfg.google_api_key)
        self._model_name = cfg.gemini_model
        self._max_tokens = 400

    def classify(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
    ) -> IntentClassification:
        """Classify the user message and identify missing parameters.

        Parameters already provided in *conversation_history* are treated as
        available so the agent does not re-ask for information the user
        already supplied in an earlier turn.

        Args:
            message: The current user message text.
            conversation_history: List of ``{"role": ..., "content": ...}``
                dicts from the current session.

        Returns:
            :class:`~src.models.IntentClassification` with intent, params,
            and whether the agent can proceed or must ask for clarification.
        """
        if not message or not message.strip():
            return IntentClassification(
                intent="other",
                provided_params={"email": None, "transaction_id": None, "user_id": None},
                missing_required_params=[],
                can_proceed=False,
                clarification_needed="Merhaba! Size nasıl yardımcı olabilirim? Hesabınız veya işlemlerinizle ilgili bir sorunuz var mı?",
                reasoning="Empty message received.",
            )

        history_params = self._extract_params_from_history(conversation_history)
        log.debug(f"History params extracted: {history_params}")

        context_note = f"Conversation history extracted params: {json.dumps(history_params)}"
        user_content = f"{context_note}\n\nCurrent user message: {message}"

        try:
            model = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=CLASSIFICATION_SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    max_output_tokens=self._max_tokens,
                ),
            )
            response = model.generate_content(user_content)
            raw = response.text.strip()
            # Strip possible markdown fences just in case
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-z]*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            raw = raw.strip()
            data = self._parse_json_robust(raw)
        except (json.JSONDecodeError, Exception) as exc:
            log.error(f"Intent classification failed: {exc}. Defaulting to safe fallback.")
            return self._fallback_classification(message, history_params)

        # Merge history-extracted params into provided_params
        provided = data.get("provided_params", {})
        missing = list(data.get("missing_required_params", []))

        for key, val in history_params.items():
            if val and not provided.get(key):
                provided[key] = val
                if key in missing:
                    missing.remove(key)

        # Re-evaluate can_proceed with merged params
        intent = data.get("intent", "other")
        can_proceed = data.get("can_proceed", False)

        if intent in ("fraud_inquiry",):
            has_id = (
                bool(provided.get("email"))
                or bool(provided.get("user_id"))
                or bool(provided.get("transaction_id"))
            )
            can_proceed = has_id
        elif intent in ("transaction_history", "account_info", "general"):
            has_id = bool(provided.get("email")) or bool(provided.get("user_id"))
            can_proceed = has_id
        elif intent == "other":
            can_proceed = True

        clarification = data.get("clarification_needed") if not can_proceed else None

        return IntentClassification(
            intent=intent,
            provided_params=provided,
            missing_required_params=missing if not can_proceed else [],
            can_proceed=can_proceed,
            clarification_needed=clarification,
            reasoning=data.get("reasoning", ""),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_params_from_history(
        self, history: List[Dict[str, Any]]
    ) -> Dict[str, Optional[str]]:
        """Scan previous messages for email, user_id, and transaction_id.

        Args:
            history: Conversation history as list of role/content dicts.

        Returns:
            Dict with keys ``email``, ``user_id``, ``transaction_id`` —
            each either a string value or ``None``.
        """
        params: Dict[str, Optional[str]] = {
            "email": None,
            "user_id": None,
            "transaction_id": None,
        }
        for msg in history:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if not params["email"]:
                m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", content)
                if m:
                    params["email"] = m.group()
            if not params["user_id"]:
                m = re.search(r"USR-\d+", content)
                if m:
                    params["user_id"] = m.group()
            if not params["transaction_id"]:
                m = re.search(r"TXN-\d+", content)
                if m:
                    params["transaction_id"] = m.group()
        return params

    @staticmethod
    def _parse_json_robust(raw: str) -> dict:
        """Parse JSON, falling back to ast.literal_eval for Python-dict strings.

        Args:
            raw: Raw string from the LLM response.

        Returns:
            Parsed dict.

        Raises:
            ValueError: If neither parser succeeds.
        """
        import ast
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Some models return Python-style dicts with single quotes / True/False/None
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
        raise ValueError(f"Cannot parse LLM output as JSON: {raw[:200]}")

    def _fallback_classification(
        self,
        message: str,
        history_params: Dict[str, Optional[str]],
    ) -> IntentClassification:
        """Produce a safe classification when the Gemini call fails.

        Applies simple regex heuristics so the agent keeps running.

        Args:
            message: Original user message.
            history_params: Params already extracted from history.

        Returns:
            Best-effort :class:`~src.models.IntentClassification`.
        """
        msg_lower = message.lower()
        fraud_keywords = ["red", "reddedildi", "neden", "fraud", "blok", "engel", "failed"]
        history_keywords = ["işlem", "transfer", "ödeme", "payment", "transaction"]
        # Include Turkish morphological variants: hesap/hesabı/hesabım/hesabın
        account_keywords = ["hesap", "hesab", "account", "durum", "status", "bilgi"]

        if any(k in msg_lower for k in fraud_keywords):
            intent = "fraud_inquiry"
        elif any(k in msg_lower for k in history_keywords):
            intent = "transaction_history"
        elif any(k in msg_lower for k in account_keywords):
            intent = "account_info"
        else:
            intent = "other"

        email_match = re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", message)
        txn_match = re.search(r"TXN-\d+", message)

        provided = {
            "email": history_params.get("email") or (email_match.group() if email_match else None),
            "transaction_id": history_params.get("transaction_id") or (txn_match.group() if txn_match else None),
            "user_id": history_params.get("user_id"),
        }

        can_proceed = (
            intent == "other"
            or bool(provided.get("email"))
            or bool(provided.get("user_id"))
            or (intent == "fraud_inquiry" and bool(provided.get("transaction_id")))
        )

        clarification = (
            "Hesabınıza erişmek için lütfen kayıtlı e-posta adresinizi paylaşır mısınız?"
            if not can_proceed
            else None
        )

        return IntentClassification(
            intent=intent,
            provided_params=provided,
            missing_required_params=[] if can_proceed else ["email"],
            can_proceed=can_proceed,
            clarification_needed=clarification,
            reasoning="Fallback heuristic classification (Gemini call failed).",
        )
