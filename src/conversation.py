"""Multi-turn conversation memory manager.

Maintains per-session message history and a cache of parameters (email,
user_id, transaction_id) that were successfully resolved in previous turns so
the agent does not re-ask for information the user already provided.
"""

import threading
from typing import Any, Dict, List, Optional

from src.logger import log
from src.models import ToolCall, ToolStatus, TransactionStatus

# Maximum number of messages kept per session (older messages are dropped)
_MAX_HISTORY_LENGTH = 20


class ConversationManager:
    """Thread-safe, per-session conversation memory.

    Stores:
    * Full message history (up to ``_MAX_HISTORY_LENGTH`` messages).
    * A parameter cache (email, user_id, last_failed_txn) built from
      successful tool calls so subsequent turns can skip re-fetching them.
    """

    def __init__(self) -> None:
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self._params: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # History API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return a copy of the message history for *session_id*.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of ``{"role": str, "content": str}`` dicts, oldest first.
        """
        with self._lock:
            return list(self._histories.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history.

        Older messages beyond ``_MAX_HISTORY_LENGTH`` are silently dropped.

        Args:
            session_id: Unique session identifier.
            role: ``"user"`` or ``"assistant"``.
            content: Message text.
        """
        with self._lock:
            history = self._histories.setdefault(session_id, [])
            history.append({"role": role, "content": content})
            # Keep only the most recent N messages
            if len(history) > _MAX_HISTORY_LENGTH:
                self._histories[session_id] = history[-_MAX_HISTORY_LENGTH:]

    # ------------------------------------------------------------------
    # Parameter cache API
    # ------------------------------------------------------------------

    def get_extracted_params(self, session_id: str) -> Dict[str, Any]:
        """Return a copy of the resolved parameter cache for *session_id*.

        Args:
            session_id: Unique session identifier.

        Returns:
            Dict with keys ``email``, ``user_id``, ``transaction_id`` (and
            others) populated from previous successful tool calls.
        """
        with self._lock:
            return dict(self._params.get(session_id, {}))

    def update_extracted_params(
        self, session_id: str, tool_results: Dict[str, ToolCall]
    ) -> None:
        """Persist identifiers discovered via successful tool calls.

        Called by the agent after each turn so future turns skip tool calls
        for data that is already known.

        Args:
            session_id: Unique session identifier.
            tool_results: Mapping of step_id → :class:`~src.models.ToolCall`.
        """
        with self._lock:
            cache = self._params.setdefault(session_id, {})
            for call in tool_results.values():
                if call.status != ToolStatus.SUCCESS or call.result is None:
                    continue
                if call.tool_name == "get_user_details":
                    cache["user_id"] = call.result.user_id
                    cache["email"] = call.result.email
                    log.debug(
                        f"[Session {session_id}] Cached user_id={call.result.user_id}"
                    )
                elif call.tool_name == "get_recent_transactions":
                    txns = call.result
                    if isinstance(txns, list):
                        failed = [
                            t for t in txns if t.status == TransactionStatus.FAILED
                        ]
                        if failed:
                            cache["transaction_id"] = failed[0].transaction_id
                            log.debug(
                                f"[Session {session_id}] Cached last_failed_txn="
                                f"{failed[0].transaction_id}"
                            )
                elif call.tool_name == "check_fraud_reason":
                    cache["last_checked_txn"] = call.result.transaction_id

    def clear_session(self, session_id: str) -> None:
        """Remove all history and cached params for *session_id*.

        Args:
            session_id: Unique session identifier.
        """
        with self._lock:
            self._histories.pop(session_id, None)
            self._params.pop(session_id, None)
            log.debug(f"[Session {session_id}] Cleared.")

    def list_sessions(self) -> List[str]:
        """Return all active session IDs.

        Returns:
            Sorted list of session ID strings.
        """
        with self._lock:
            return sorted(self._histories.keys())

    def session_summary(self, session_id: str) -> Dict[str, Any]:
        """Return a summary dict for display in the Streamlit sidebar.

        Args:
            session_id: Unique session identifier.

        Returns:
            Dict with message_count, cached params, etc.
        """
        with self._lock:
            history = self._histories.get(session_id, [])
            params = self._params.get(session_id, {})
        return {
            "session_id": session_id,
            "message_count": len(history),
            "cached_email": params.get("email"),
            "cached_user_id": params.get("user_id"),
            "cached_transaction_id": params.get("transaction_id"),
        }
