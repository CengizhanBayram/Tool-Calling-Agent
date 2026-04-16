"""Parallel tool executor with dependency resolution and retry logic.

Executes an :class:`~src.models.ExecutionPlan` by:

* Resolving ``$ref:step_id.field`` parameter placeholders at runtime.
* Running independent steps concurrently via :class:`~concurrent.futures.ThreadPoolExecutor`.
* Retrying transient errors (network / timeout) with exponential back-off.
* Isolating errors — one failing tool does not crash others.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from src.config import Settings, settings
from src.logger import log, log_tool_call
from src.models import (
    ExecutionPlan,
    PlanStep,
    ToolCall,
    ToolStatus,
    TransactionStatus,
)
from src.tool_registry import TOOL_FUNCTION_MAP
from src.tools import (
    FraudRecordNotFoundError,
    InvalidParameterError,
    TransactionNotFoundError,
    UserNotFoundError,
)


class ToolExecutor:
    """Executes a tool execution plan respecting dependencies and retrying on failure.

    Args:
        cfg: Application settings. Defaults to the module-level singleton.
    """

    # Exceptions that are transient and safe to retry
    _TRANSIENT = (ConnectionError, TimeoutError, OSError)

    # Domain exceptions — do NOT retry; return as structured errors
    _DOMAIN_ERRORS = (
        UserNotFoundError,
        TransactionNotFoundError,
        FraudRecordNotFoundError,
        InvalidParameterError,
    )

    def __init__(self, cfg: Settings = settings) -> None:
        self._timeout = cfg.tool_timeout_seconds
        self._max_retries = cfg.tool_max_retries
        self._retry_delay = cfg.tool_retry_delay_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, ToolCall]:
        """Execute all steps in the plan respecting their dependency graph.

        Args:
            plan: The :class:`~src.models.ExecutionPlan` to execute.

        Returns:
            Dict mapping ``step_id`` → :class:`~src.models.ToolCall` result
            for every step that was attempted.  Steps whose dependencies
            failed are skipped and recorded with ``ToolStatus.SKIPPED``.
        """
        if plan.requires_user_input:
            return {}

        completed: Dict[str, ToolCall] = {}
        pending = list(plan.steps)
        failed_step_ids: set = set()

        while pending:
            ready = [
                step
                for step in pending
                if all(dep in completed for dep in step.depends_on)
                and not any(dep in failed_step_ids for dep in step.depends_on)
            ]

            # Steps whose upstream dependency failed → skip them
            unrunnable = [
                step
                for step in pending
                if any(dep in failed_step_ids for dep in step.depends_on)
            ]
            for step in unrunnable:
                completed[step.step_id] = ToolCall(
                    tool_name=step.tool_name,
                    parameters=step.parameters,
                    error="Bağımlı adım başarısız olduğu için bu adım atlandı.",
                    status=ToolStatus.SKIPPED,
                )
                # Propagate: skipped steps also block their dependents
                failed_step_ids.add(step.step_id)
                pending.remove(step)
                log.warning(
                    f"[SKIPPED] {step.tool_name} (step {step.step_id}) — "
                    f"upstream dependency failed."
                )

            if not ready and not unrunnable:
                if pending:
                    log.error(
                        "No ready steps but %d pending remain — possible cycle.",
                        len(pending),
                    )
                break

            parallel_ready = [s for s in ready if s.can_run_parallel]
            serial_ready = [s for s in ready if not s.can_run_parallel]

            # Run parallel-flagged steps concurrently
            if len(parallel_ready) > 1:
                with ThreadPoolExecutor(max_workers=len(parallel_ready)) as pool:
                    future_map = {
                        pool.submit(
                            self._execute_with_retry,
                            step.tool_name,
                            self._resolve_parameters(step, completed),
                            step.step_id,
                        ): step
                        for step in parallel_ready
                    }
                    for future in as_completed(future_map):
                        step = future_map[future]
                        result = future.result()
                        completed[step.step_id] = result
                        if result.status == ToolStatus.ERROR:
                            failed_step_ids.add(step.step_id)
                        pending.remove(step)
            elif parallel_ready:
                serial_ready.extend(parallel_ready)

            # Run serial steps one at a time
            for step in serial_ready:
                if step not in pending:
                    continue
                try:
                    resolved = self._resolve_parameters(step, completed)
                except Exception as exc:
                    log.error(f"Parameter resolution failed for {step.step_id}: {exc}")
                    result = self._error_call(step.tool_name, step.parameters, str(exc), 0)
                    completed[step.step_id] = result
                    failed_step_ids.add(step.step_id)
                    pending.remove(step)
                    continue

                result = self._execute_with_retry(step.tool_name, resolved, step.step_id)
                completed[step.step_id] = result
                if result.status == ToolStatus.ERROR:
                    failed_step_ids.add(step.step_id)
                pending.remove(step)

        return completed

    # ------------------------------------------------------------------
    # Parameter resolution
    # ------------------------------------------------------------------

    def _resolve_parameters(
        self,
        step: PlanStep,
        completed: Dict[str, ToolCall],
    ) -> Dict[str, Any]:
        """Resolve ``$ref:step_id.field`` placeholders in step parameters.

        Supported patterns
        ------------------
        * ``$ref:step_1.user_id``       → ``completed["step_1"].result.user_id``
        * ``$ref:step_2.first_failed``  → transaction_id of the first failed
          transaction in the list returned by step_2.

        Args:
            step: The plan step whose parameters may contain refs.
            completed: Map of already-executed step results.

        Returns:
            Fully-resolved parameter dict ready to pass to the tool function.

        Raises:
            RuntimeError: If a referenced step has not completed or failed.
            ValueError: If the field cannot be extracted from the result.
        """
        resolved: Dict[str, Any] = {}
        for key, value in step.parameters.items():
            if isinstance(value, str) and value.startswith("$ref:"):
                ref_path = value[5:]  # strip "$ref:"
                parts = ref_path.split(".", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid $ref syntax: '{value}'")
                ref_step_id, ref_field = parts
                source = completed.get(ref_step_id)
                if source is None:
                    raise RuntimeError(
                        f"Cannot resolve '{value}': step '{ref_step_id}' has not run yet."
                    )
                if source.status == ToolStatus.ERROR:
                    raise RuntimeError(
                        f"Cannot resolve '{value}': step '{ref_step_id}' failed with: {source.error}"
                    )
                resolved[key] = self._extract_field(source.result, ref_field)
            else:
                resolved[key] = value
        return resolved

    @staticmethod
    def _extract_field(result: Any, field: str) -> Any:
        """Extract a named field (or computed property) from a tool result.

        Args:
            result: The tool's return value (dataclass, list, etc.).
            field: The field name or special computed name.

        Returns:
            The extracted value.

        Raises:
            TransactionNotFoundError: If ``first_failed`` requested but none exist.
            ValueError: If the field cannot be found on the result.
        """
        if field == "first_failed":
            if isinstance(result, list):
                failed = [t for t in result if t.status == TransactionStatus.FAILED]
                if not failed:
                    raise TransactionNotFoundError(
                        "Son işlemler arasında başarısız bir işlem bulunamadı. "
                        "İncelemek istediğiniz işlemin ID'sini (TXN-XXXXX) paylaşır mısınız?"
                    )
                return failed[0].transaction_id
            raise ValueError(f"Cannot extract 'first_failed' from non-list type: {type(result)}")

        if hasattr(result, field):
            return getattr(result, field)
        if isinstance(result, dict) and field in result:
            return result[field]
        raise ValueError(f"Cannot extract field '{field}' from {type(result).__name__}")

    # ------------------------------------------------------------------
    # Execution with retry
    # ------------------------------------------------------------------

    def _execute_with_retry(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        step_id: str,
    ) -> ToolCall:
        """Execute a tool function with exponential back-off on transient errors.

        Args:
            tool_name: Name of the tool to execute (key in TOOL_FUNCTION_MAP).
            parameters: Resolved parameter dict to pass to the function.
            step_id: Used for log context.

        Returns:
            A :class:`~src.models.ToolCall` recording the outcome.
        """
        tool_fn = TOOL_FUNCTION_MAP.get(tool_name)
        if tool_fn is None:
            return self._error_call(tool_name, parameters, f"Unknown tool: '{tool_name}'", 1)

        for attempt in range(1, self._max_retries + 1):
            t_start = time.perf_counter()
            try:
                result = tool_fn(**parameters)
                duration_ms = (time.perf_counter() - t_start) * 1000
                log_tool_call(
                    tool_name=tool_name,
                    parameters=parameters,
                    result_type=type(result).__name__,
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                    status="success",
                )
                return ToolCall(
                    tool_name=tool_name,
                    parameters=parameters,
                    result=result,
                    status=ToolStatus.SUCCESS,
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                )

            except self._TRANSIENT as exc:
                duration_ms = (time.perf_counter() - t_start) * 1000
                if attempt < self._max_retries:
                    wait = self._retry_delay * (2 ** (attempt - 1))
                    log.warning(
                        f"[RETRY {attempt}/{self._max_retries}] {tool_name} transient error: {exc}. "
                        f"Retrying in {wait:.2f}s…"
                    )
                    time.sleep(wait)
                else:
                    log_tool_call(
                        tool_name=tool_name,
                        parameters=parameters,
                        result_type="N/A",
                        duration_ms=duration_ms,
                        attempt_count=attempt,
                        status="error",
                        error=str(exc),
                    )
                    return self._error_call(tool_name, parameters, str(exc), attempt)

            except self._DOMAIN_ERRORS as exc:
                duration_ms = (time.perf_counter() - t_start) * 1000
                log_tool_call(
                    tool_name=tool_name,
                    parameters=parameters,
                    result_type="N/A",
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                    status="error",
                    error=str(exc),
                )
                return ToolCall(
                    tool_name=tool_name,
                    parameters=parameters,
                    error=str(exc),
                    status=ToolStatus.ERROR,
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                )

            except Exception as exc:
                duration_ms = (time.perf_counter() - t_start) * 1000
                msg = f"Beklenmedik hata ({type(exc).__name__}): {exc}"
                log.error(f"[UNEXPECTED] {tool_name} step {step_id}: {msg}")
                log_tool_call(
                    tool_name=tool_name,
                    parameters=parameters,
                    result_type="N/A",
                    duration_ms=duration_ms,
                    attempt_count=attempt,
                    status="error",
                    error=msg,
                )
                return self._error_call(tool_name, parameters, msg, attempt)

        # Should not be reached
        return self._error_call(tool_name, parameters, "Max retries exhausted.", self._max_retries)

    @staticmethod
    def _error_call(
        tool_name: str,
        params: Dict[str, Any],
        error_msg: str,
        attempts: int,
    ) -> ToolCall:
        """Build a failed :class:`~src.models.ToolCall` record."""
        return ToolCall(
            tool_name=tool_name,
            parameters=params,
            error=error_msg,
            status=ToolStatus.ERROR,
            attempt_count=attempts,
        )
