"""Master orchestrator — wires all pipeline stages together.

Pipeline stages (executed per ``chat()`` call)
-----------------------------------------------
1. IntentClassifier  — detect intent + missing params
2. ToolPlanner       — build dependency-aware execution plan
3. ToolExecutor      — run plan (with retry + error isolation)
4. ResultValidator   — validate tool outputs
5. ResponseSynthesizer — generate natural Turkish answer via Claude
"""

import time
from typing import Any, Dict, Optional

from src.config import Settings, settings
from src.conversation import ConversationManager
from src.intent_classifier import IntentClassifier
from src.logger import log
from src.models import AgentResponse, ToolCall
from src.response_synthesizer import ResponseSynthesizer
from src.result_validator import ResultValidator
from src.tool_executor import ToolExecutor
from src.tool_planner import ToolPlanner


class Agent:
    """Stateless-per-call orchestrator for the multi-stage tool pipeline.

    Each session's conversational memory is stored in :class:`ConversationManager`.
    Multiple concurrent sessions are fully isolated.

    Args:
        cfg: Application settings. Defaults to the module-level singleton.
    """

    def __init__(self, cfg: Settings = settings) -> None:
        self._cfg = cfg
        self._classifier = IntentClassifier(cfg)
        self._planner = ToolPlanner()
        self._executor = ToolExecutor(cfg)
        self._validator = ResultValidator()
        self._synthesizer = ResponseSynthesizer(cfg)
        self._conversation = ConversationManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, session_id: str = "default") -> AgentResponse:
        """Process a single user message and return the agent's response.

        Args:
            user_message: Raw text entered by the user.
            session_id: Identifier for the current conversation session.
                Different sessions do not share state.

        Returns:
            :class:`~src.models.AgentResponse` with the answer text,
            a trace of all tool calls made, and latency information.
        """
        t_start = time.perf_counter()
        log.info(f"[Session {session_id}] User: {user_message[:120]}")

        history = self._conversation.get_history(session_id)
        context = self._conversation.get_extracted_params(session_id)

        # ----------------------------------------------------------------
        # Stage 1 — Intent classification + missing-param detection
        # ----------------------------------------------------------------
        classification = self._classifier.classify(user_message, history)
        log.info(
            f"[Session {session_id}] Intent={classification.intent} "
            f"can_proceed={classification.can_proceed} "
            f"provided={classification.provided_params}"
        )

        # ----------------------------------------------------------------
        # Stage 2a — Missing params: ask user immediately, no tool calls
        # ----------------------------------------------------------------
        if not classification.can_proceed:
            clarification = (
                classification.clarification_needed
                or "Bu konuda size yardımcı olabilmem için biraz daha bilgiye ihtiyacım var. "
                   "Lütfen sorununuzu daha ayrıntılı açıklar mısınız?"
            )
            self._conversation.add_message(session_id, "user", user_message)
            self._conversation.add_message(session_id, "assistant", clarification)
            latency = (time.perf_counter() - t_start) * 1000
            log.info(f"[Session {session_id}] Asking for clarification (no tools run).")
            return AgentResponse(
                answer=clarification,
                tool_calls=[],
                iterations=0,
                total_latency_ms=latency,
                requires_followup=True,
                followup_question=clarification,
            )

        # ----------------------------------------------------------------
        # Stage 2b — Build execution plan
        # ----------------------------------------------------------------
        plan = self._planner.build_plan(classification, context)
        log.info(
            f"[Session {session_id}] Plan: {[s.tool_name for s in plan.steps]} "
            f"requires_input={plan.requires_user_input}"
        )

        # ----------------------------------------------------------------
        # Stage 3 — Execute plan
        # ----------------------------------------------------------------
        tool_results: Dict[str, ToolCall] = self._executor.execute_plan(plan)

        # ----------------------------------------------------------------
        # Stage 4 — Validate outputs
        # ----------------------------------------------------------------
        validations = self._validator.validate_all(tool_results)

        # ----------------------------------------------------------------
        # Stage 5 — Synthesize final answer
        # ----------------------------------------------------------------
        answer = self._synthesizer.synthesize(user_message, tool_results, validations)

        # ----------------------------------------------------------------
        # Update conversation memory
        # ----------------------------------------------------------------
        self._conversation.add_message(session_id, "user", user_message)
        self._conversation.add_message(session_id, "assistant", answer)
        self._conversation.update_extracted_params(session_id, tool_results)

        latency = (time.perf_counter() - t_start) * 1000
        tool_calls_list = list(tool_results.values())

        log.info(
            f"[Session {session_id}] Done — {len(tool_calls_list)} tool calls, "
            f"{latency:.0f}ms total."
        )

        return AgentResponse(
            answer=answer,
            tool_calls=tool_calls_list,
            iterations=len(tool_calls_list),
            total_latency_ms=latency,
            requires_followup=False,
        )

    # ------------------------------------------------------------------
    # Session management helpers (used by Streamlit UI / CLI)
    # ------------------------------------------------------------------

    def new_session(self, session_id: str) -> None:
        """Clear all memory for *session_id* to start fresh.

        Args:
            session_id: Session to reset.
        """
        self._conversation.clear_session(session_id)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Return display-friendly session metadata.

        Args:
            session_id: Session to summarise.

        Returns:
            Dict suitable for sidebar display in the Streamlit UI.
        """
        return self._conversation.session_summary(session_id)

    def list_sessions(self):
        """Return all active session IDs."""
        return self._conversation.list_sessions()
