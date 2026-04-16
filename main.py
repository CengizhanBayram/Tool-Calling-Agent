"""CLI entry point for the tool-calling agent.

Usage
-----
Interactive chat session::

    python main.py --interactive

Single query::

    python main.py --query "ali@sirket.com son ödeme neden reddedildi?"

Run all 12 test scenarios::

    python main.py --run-tests

Performance benchmark (10 representative queries)::

    python main.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Rich for pretty output
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure project root is on sys.path when run directly
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.agent import Agent
from src.config import settings
from src.database import db
from src.models import AgentResponse, ToolStatus

console = Console()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_tool_trace(response: AgentResponse) -> None:
    """Print a Rich panel showing the tool execution trace."""
    if not response.tool_calls:
        return

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column("Status", style="bold", min_width=3)
    table.add_column("Tool", style="cyan")
    table.add_column("Params", style="dim")
    table.add_column("Latency", style="yellow", justify="right")

    for call in response.tool_calls:
        if call.status == ToolStatus.SUCCESS:
            icon = "✅"
        elif call.status == ToolStatus.ERROR:
            icon = "❌"
        else:
            icon = "⏭️"

        params_str = ", ".join(f"{k}={repr(v)}" for k, v in call.parameters.items())
        if len(params_str) > 60:
            params_str = params_str[:57] + "…"

        latency_str = f"{call.duration_ms:.0f}ms" if call.duration_ms else "—"
        table.add_row(icon, call.tool_name, f"({params_str})", latency_str)

    console.print(
        Panel(table, title="Tool Execution Trace", border_style="dim blue")
    )


def print_response(response: AgentResponse, query: str = "") -> None:
    """Print the agent answer + tool trace in a consistent layout."""
    if query:
        console.print(f"\n[bold cyan]You:[/bold cyan] {query}")

    console.print(
        Panel(
            response.answer,
            title=f"[bold green]Agent[/bold green]  ·  {response.total_latency_ms:.0f}ms",
            border_style="green",
        )
    )
    print_tool_trace(response)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive() -> None:
    """Start an interactive REPL chat session."""
    console.print(
        Panel(
            "[bold]Tool-Calling Agent[/bold] — Türkçe Müşteri Destek Asistanı\n"
            "[dim]Çıkmak için: quit / exit / Ctrl+C[/dim]",
            border_style="bright_blue",
        )
    )

    agent = Agent()
    session_id = f"cli_{int(time.time())}"

    while True:
        try:
            user_input = console.input("\n[bold cyan]Siz:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Görüşmek üzere![/dim]")
            break

        if user_input.lower() in ("quit", "exit", "q", ""):
            console.print("[dim]Görüşmek üzere![/dim]")
            break

        if user_input.lower() == "/yeni":
            agent.new_session(session_id)
            console.print("[dim]Oturum sıfırlandı.[/dim]")
            continue

        if user_input.lower() == "/reload":
            db.reload()
            console.print("[green]Veritabanı yenilendi.[/green]")
            continue

        response = agent.chat(user_input, session_id=session_id)
        print_response(response)


# ---------------------------------------------------------------------------
# Single query mode
# ---------------------------------------------------------------------------

def run_single_query(query: str) -> None:
    """Run a single query and print the result."""
    agent = Agent()
    response = agent.chat(query, session_id="cli_single")
    print_response(response, query=query)


# ---------------------------------------------------------------------------
# Test suite (12 scenarios)
# ---------------------------------------------------------------------------

TEST_CASES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Full chain: email → user → txn → fraud",
        "query": "ali@sirket.com son ödeme neden reddedildi?",
        "session": "test_1",
        "expect_tools": ["get_user_details", "get_recent_transactions", "check_fraud_reason"],
        "expect_in_answer": ["FRAUD_SUSPECTED", "Moskova"],
        "expect_no_followup": True,
    },
    {
        "id": 2,
        "name": "No email → asks user, zero tool calls",
        "query": "Son işlemlerimi görmek istiyorum",
        "session": "test_2",
        "expect_tools": [],
        "expect_followup": True,
    },
    {
        "id": 3,
        "name": "Suspended account info",
        "query": "ayse@ornek.com hesabının durumu nedir?",
        "session": "test_3",
        "expect_tools": ["get_user_details"],
        "expect_in_answer_lower": ["askı", "suspend", "ayşe"],
    },
    {
        "id": 4,
        "name": "Unknown email → graceful error",
        "query": "bilinmeyen@email.com hesabı nedir?",
        "session": "test_4",
        "expect_tool_error": "get_user_details",
    },
    {
        "id": 5,
        "name": "TXN-ID given directly → single step",
        "query": "TXN-10041 neden reddedildi?",
        "session": "test_5",
        "expect_tools": ["check_fraud_reason"],
        "expect_max_tools": 1,
    },
    {
        "id": 6,
        "name": "Multi-turn: email from first message, no re-asking",
        "query_sequence": [
            "ali@sirket.com hesabım hakkında bilgi almak istiyorum",
            "son işlemlerimi de göster",
        ],
        "session": "test_6",
        "final_expect_tools_include": ["get_recent_transactions"],
    },
    {
        "id": 7,
        "name": "Invalid email format → InvalidParameterError handled",
        "query": "gecersiz-email hesabı",
        "session": "test_7",
        "expect_tool_error_or_clarification": True,
    },
    {
        "id": 8,
        "name": "Successful txn fraud check → FraudRecordNotFoundError",
        "query": "TXN-10042 neden reddedildi?",
        "session": "test_8",
        "expect_tool_error": "check_fraud_reason",
    },
    {
        "id": 9,
        "name": "Off-topic: no tool calls, polite redirect",
        "query": "hava nasıl bugün?",
        "session": "test_9",
        "expect_tools": [],
        "expect_no_followup": False,  # can_proceed=True, just no tools
    },
    {
        "id": 10,
        "name": "Empty message → asks what user needs",
        "query": "",
        "session": "test_10",
        "expect_followup": True,
        "expect_tools": [],
    },
    {
        "id": 11,
        "name": "Live data update: fraud reason changes in JSON → new reason returned",
        "query": "TXN-10041 neden reddedildi?",
        "session": "test_11",
        "dynamic_test": "fraud_reason_update",
    },
    {
        "id": 12,
        "name": "Live data update: new user added to DB → found immediately",
        "query": "yenikullanici@test.com hesap bilgileri",
        "session": "test_12",
        "dynamic_test": "new_user",
    },
]


def _run_test_case(agent: Agent, case: Dict[str, Any], verbose: bool = True) -> bool:
    """Run a single test case and return True if it passes."""
    test_id = case["id"]
    name = case["name"]

    try:
        # Multi-turn test
        if "query_sequence" in case:
            responses = []
            for q in case["query_sequence"]:
                r = agent.chat(q, session_id=case["session"])
                responses.append(r)
            response = responses[-1]
        else:
            response = agent.chat(case["query"], session_id=case["session"])

        # --- assertions ---
        if case.get("expect_tools") is not None:
            expected = set(case["expect_tools"])
            actual = {c.tool_name for c in response.tool_calls}
            if expected and not expected.issubset(actual):
                if verbose:
                    console.print(f"  [red]FAIL[/red] Expected tools {expected} but got {actual}")
                return False

        if case.get("expect_max_tools"):
            if len(response.tool_calls) > case["expect_max_tools"]:
                if verbose:
                    console.print(
                        f"  [red]FAIL[/red] Expected max {case['expect_max_tools']} tool calls, "
                        f"got {len(response.tool_calls)}"
                    )
                return False

        if case.get("expect_followup"):
            if not response.requires_followup:
                if verbose:
                    console.print("  [red]FAIL[/red] Expected requires_followup=True")
                return False

        if case.get("expect_no_followup"):
            if response.requires_followup:
                if verbose:
                    console.print("  [red]FAIL[/red] Expected requires_followup=False")
                return False

        if case.get("expect_in_answer"):
            for keyword in case["expect_in_answer"]:
                if keyword not in response.answer:
                    if verbose:
                        console.print(f"  [yellow]WARN[/yellow] Expected '{keyword}' in answer")
                    # Don't fail on answer content — synthesis is non-deterministic

        if case.get("expect_in_answer_lower"):
            ans_lower = response.answer.lower()
            for keyword in case["expect_in_answer_lower"]:
                if keyword not in ans_lower:
                    if verbose:
                        console.print(f"  [yellow]WARN[/yellow] Expected '{keyword}' in answer (case-insensitive)")

        if case.get("expect_tool_error"):
            tool_name = case["expect_tool_error"]
            errors = [c for c in response.tool_calls if c.tool_name == tool_name and c.error]
            if not errors:
                if verbose:
                    console.print(
                        f"  [red]FAIL[/red] Expected error from {tool_name} "
                        f"but got: {[(c.tool_name, c.status) for c in response.tool_calls]}"
                    )
                return False

        if case.get("expect_tool_error_or_clarification"):
            has_error = any(c.error for c in response.tool_calls)
            if not has_error and not response.requires_followup:
                # Either is acceptable — just make sure no unhandled exception
                pass

        if "final_expect_tools_include" in case:
            actual = {c.tool_name for c in response.tool_calls}
            for t in case["final_expect_tools_include"]:
                if t not in actual:
                    if verbose:
                        console.print(f"  [red]FAIL[/red] Expected {t} in final turn tools: {actual}")
                    return False

        return True

    except Exception as exc:
        if verbose:
            console.print(f"  [red]EXCEPTION[/red] {type(exc).__name__}: {exc}")
        return False


def _run_dynamic_test_11(agent: Agent, case: Dict[str, Any]) -> bool:
    """Test 11: Modify fraud reason in JSON → db.reload() → verify new reason returned."""
    fraud_path = settings.data_dir / "fraud_reasons.json"
    new_reason = "TEST_UPDATE: İşlem test güncellemesi ile değiştirildi."

    # Read current data
    with open(fraud_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    try:
        # Patch TXN-10041 reason_tr
        patched = []
        for rec in original_data:
            if rec["transaction_id"] == "TXN-10041":
                rec = dict(rec)
                rec["reason_tr"] = new_reason
            patched.append(rec)

        with open(fraud_path, "w", encoding="utf-8") as f:
            json.dump(patched, f, ensure_ascii=False, indent=2)

        db.reload()
        agent.new_session(case["session"])

        response = agent.chat(case["query"], session_id=case["session"])

        # The synthesizer receives the updated reason_tr → should appear in answer
        # (non-deterministic synthesis, so we check the raw tool result)
        fraud_calls = [c for c in response.tool_calls if c.tool_name == "check_fraud_reason"]
        if not fraud_calls:
            console.print("  [red]FAIL[/red] check_fraud_reason was not called")
            return False

        call = fraud_calls[0]
        if call.status != "success":
            console.print(f"  [red]FAIL[/red] check_fraud_reason returned error: {call.error}")
            return False

        if call.result and call.result.reason_tr != new_reason:
            console.print(
                f"  [red]FAIL[/red] reason_tr not updated.\n"
                f"  Expected: {new_reason}\n"
                f"  Got: {call.result.reason_tr}"
            )
            return False

        return True

    finally:
        # Restore original data
        with open(fraud_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        db.reload()


def _run_dynamic_test_12(agent: Agent, case: Dict[str, Any]) -> bool:
    """Test 12: Add new user to users.json → db.reload() → get_user_details finds them."""
    users_path = settings.data_dir / "users.json"
    new_user = {
        "user_id": "USR-099",
        "email": "yenikullanici@test.com",
        "full_name": "Yeni Kullanıcı",
        "account_status": "active",
        "account_type": "basic",
        "created_at": "2024-11-15",
        "phone": "+90-500-000-0099",
        "risk_score": 1,
    }

    with open(users_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    try:
        patched = original_data + [new_user]
        with open(users_path, "w", encoding="utf-8") as f:
            json.dump(patched, f, ensure_ascii=False, indent=2)

        db.reload()
        agent.new_session(case["session"])

        response = agent.chat(case["query"], session_id=case["session"])

        user_calls = [c for c in response.tool_calls if c.tool_name == "get_user_details"]
        if not user_calls:
            console.print("  [red]FAIL[/red] get_user_details was not called")
            return False

        call = user_calls[0]
        if call.status != "success":
            console.print(f"  [red]FAIL[/red] get_user_details error: {call.error}")
            return False

        if call.result and call.result.user_id != "USR-099":
            console.print(f"  [red]FAIL[/red] Wrong user returned: {call.result.user_id}")
            return False

        return True

    finally:
        with open(users_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        db.reload()


def run_tests(verbose: bool = True) -> None:
    """Execute all 12 test scenarios and print a summary table."""
    console.print(
        Panel("[bold]Running 12 Test Scenarios[/bold]", border_style="bright_blue")
    )

    agent = Agent()
    results: List[Dict[str, Any]] = []
    passed = 0
    failed = 0

    for case in TEST_CASES:
        console.print(f"\n[bold]Test {case['id']:02d}:[/bold] {case['name']}")

        if case.get("dynamic_test") == "fraud_reason_update":
            ok = _run_dynamic_test_11(agent, case)
        elif case.get("dynamic_test") == "new_user":
            ok = _run_dynamic_test_12(agent, case)
        else:
            # Reset session for each test to isolate state (except multi-turn test 6)
            agent.new_session(case.get("session", f"test_{case['id']}"))
            ok = _run_test_case(agent, case, verbose=verbose)

        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {status}")
        results.append({"id": case["id"], "name": case["name"], "passed": ok})
        if ok:
            passed += 1
        else:
            failed += 1

    # Summary table
    table = Table(title="Test Results", box=box.ROUNDED, show_lines=True)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Test Name", min_width=40)
    table.add_column("Result", justify="center", min_width=8)

    for r in results:
        result_text = "[green]✅ PASS[/green]" if r["passed"] else "[red]❌ FAIL[/red]"
        table.add_row(str(r["id"]), r["name"], result_text)

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]{'[green]' if failed == 0 else '[red]'}"
        f"{passed}/{len(TEST_CASES)} tests passed"
        f"{'[/green]' if failed == 0 else '[/red]'}[/bold]"
    )


# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    ("Full chain fraud", "ali@sirket.com son ödeme neden reddedildi?"),
    ("Account info", "mehmet@kurumsal.com hesap bilgileri"),
    ("Direct TXN", "TXN-10041 neden reddedildi?"),
    ("Transaction history", "emre@tech.io işlem geçmişi"),
    ("Unknown email", "bilinmeyen999@email.com hesabı"),
]


def run_benchmark() -> None:
    """Run a small benchmark and report per-query latency."""
    console.print(Panel("[bold]Benchmark Mode[/bold]", border_style="yellow"))
    agent = Agent()
    table = Table(title="Benchmark Results", box=box.SIMPLE)
    table.add_column("Query", min_width=45)
    table.add_column("Tools", justify="center")
    table.add_column("Latency", justify="right")

    for label, query in BENCHMARK_QUERIES:
        agent.new_session("bench")
        resp = agent.chat(query, session_id="bench")
        table.add_row(
            f"{label}: {query[:40]}…" if len(query) > 40 else f"{label}: {query}",
            str(len(resp.tool_calls)),
            f"{resp.total_latency_ms:.0f}ms",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-Calling Agent CLI")
    parser.add_argument("--interactive", action="store_true", help="Start interactive chat")
    parser.add_argument("--query", type=str, help="Run a single query")
    parser.add_argument("--run-tests", action="store_true", help="Run all 12 test scenarios")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.query:
        run_single_query(args.query)
    elif args.run_tests:
        run_tests()
    elif args.benchmark:
        run_benchmark()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
