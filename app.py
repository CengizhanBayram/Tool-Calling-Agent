"""Streamlit chat UI for the tool-calling agent.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.agent import Agent
from src.config import settings
from src.database import db
from src.models import AgentResponse, ToolStatus

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Müşteri Destek Asistanı",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Ensure all required session state keys exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"ui_{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role", "content", "meta"}
    if "agent" not in st.session_state:
        st.session_state.agent = Agent()
    if "total_tool_calls" not in st.session_state:
        st.session_state.total_tool_calls = 0
    if "successful_tool_calls" not in st.session_state:
        st.session_state.successful_tool_calls = 0
    if "total_latency_ms" not in st.session_state:
        st.session_state.total_latency_ms = 0.0
    if "turn_count" not in st.session_state:
        st.session_state.turn_count = 0


_init_state()

# ---------------------------------------------------------------------------
# Helper: DB file info
# ---------------------------------------------------------------------------

def _db_file_info() -> List[Dict[str, Any]]:
    """Return last-modified times for each JSON data file."""
    files = ["users.json", "transactions.json", "fraud_reasons.json"]
    info = []
    for fname in files:
        path = settings.data_dir / fname
        if path.exists():
            mtime = path.stat().st_mtime
            info.append({"file": fname, "mtime": time.strftime("%H:%M:%S", time.localtime(mtime))})
        else:
            info.append({"file": fname, "mtime": "not found"})
    return info


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 Agent Kontrolü")

    # Session info
    st.subheader("Oturum")
    st.code(st.session_state.session_id, language=None)
    if st.button("🆕 Yeni Oturum", use_container_width=True):
        st.session_state.agent.new_session(st.session_state.session_id)
        st.session_state.session_id = f"ui_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.total_tool_calls = 0
        st.session_state.successful_tool_calls = 0
        st.session_state.total_latency_ms = 0.0
        st.session_state.turn_count = 0
        st.rerun()

    # Extracted params (from conversation memory)
    summary = st.session_state.agent.get_session_summary(st.session_state.session_id)
    st.subheader("Hafıza (Mevcut Oturum)")
    col1, col2 = st.columns(2)
    col1.metric("Mesajlar", summary.get("message_count", 0))
    col2.metric("Turlar", st.session_state.turn_count)
    if summary.get("cached_email"):
        st.info(f"📧 {summary['cached_email']}")
    if summary.get("cached_user_id"):
        st.info(f"👤 {summary['cached_user_id']}")
    if summary.get("cached_transaction_id"):
        st.info(f"💳 {summary['cached_transaction_id']}")

    st.divider()

    # DB status
    st.subheader("Veritabanı Durumu")
    for fi in _db_file_info():
        st.caption(f"📄 {fi['file']} — {fi['mtime']}")

    if st.button("🔄 DB Yenile", use_container_width=True):
        db.reload()
        st.success("Veritabanı yenilendi!")

    st.divider()

    # Statistics
    st.subheader("İstatistikler")
    total = st.session_state.total_tool_calls
    success = st.session_state.successful_tool_calls
    avg_lat = (
        st.session_state.total_latency_ms / max(st.session_state.turn_count, 1)
    )
    c1, c2 = st.columns(2)
    c1.metric("Toplam Araç", total)
    c2.metric("Başarı", f"{success}/{total}" if total else "—")
    st.metric("Ort. Gecikme", f"{avg_lat:.0f}ms" if st.session_state.turn_count else "—")

    st.divider()

    # Quick examples
    st.subheader("Örnek Sorgular")
    examples = [
        "ali@sirket.com son ödeme neden reddedildi?",
        "TXN-10041 neden reddedildi?",
        "ayse@ornek.com hesap durumu nedir?",
        "emre@tech.io son işlemlerim",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state["_pending_query"] = ex
            st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("💬 Müşteri Destek Asistanı")
st.caption("Hesap bilgileri, işlem geçmişi ve red nedenleri için sorularınızı yazın.")

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show tool trace for assistant messages that have meta
        meta: AgentResponse | None = msg.get("meta")
        if meta and meta.tool_calls:
            with st.expander(
                f"🔧 Araç İzleme — {len(meta.tool_calls)} çağrı "
                f"· {meta.total_latency_ms:.0f}ms",
                expanded=False,
            ):
                for call in meta.tool_calls:
                    if call.status == ToolStatus.SUCCESS:
                        icon, color = "✅", "green"
                    elif call.status == ToolStatus.ERROR:
                        icon, color = "❌", "red"
                    else:
                        icon, color = "⏭️", "orange"

                    params_str = ", ".join(
                        f"{k}={repr(v)}" for k, v in call.parameters.items()
                    )
                    st.markdown(
                        f":{color}[{icon} **{call.tool_name}**({params_str})]  "
                        f"`{call.duration_ms:.0f}ms`  ·  "
                        f"Deneme: {call.attempt_count}"
                    )
                    if call.error:
                        st.error(f"Hata: {call.error}")

            with st.expander("🔍 Ham Sonuçlar (Debug)", expanded=False):
                for step_id, call in enumerate(meta.tool_calls):
                    if call.result is not None:
                        st.json(
                            {
                                "tool": call.tool_name,
                                "status": str(call.status),
                                "result_type": type(call.result).__name__,
                                "error": call.error,
                            }
                        )

            col1, col2, col3 = st.columns(3)
            col1.metric("Toplam Gecikme", f"{meta.total_latency_ms:.0f}ms")
            col2.metric("Araç Çağrısı", len(meta.tool_calls))
            col3.metric(
                "Hata",
                sum(1 for c in meta.tool_calls if c.status == ToolStatus.ERROR),
            )


# ---------------------------------------------------------------------------
# Handle input (from chat box OR sidebar quick-example button)
# ---------------------------------------------------------------------------

pending = st.session_state.pop("_pending_query", None)
user_input: str | None = st.chat_input("Sorunuzu yazın…") or pending

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream a "thinking" indicator
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyor…"):
            response = st.session_state.agent.chat(
                user_input,
                session_id=st.session_state.session_id,
            )

        st.markdown(response.answer)

        # Tool trace
        if response.tool_calls:
            with st.expander(
                f"🔧 Araç İzleme — {len(response.tool_calls)} çağrı "
                f"· {response.total_latency_ms:.0f}ms",
                expanded=True,
            ):
                for call in response.tool_calls:
                    if call.status == ToolStatus.SUCCESS:
                        icon, color = "✅", "green"
                    elif call.status == ToolStatus.ERROR:
                        icon, color = "❌", "red"
                    else:
                        icon, color = "⏭️", "orange"

                    params_str = ", ".join(
                        f"{k}={repr(v)}" for k, v in call.parameters.items()
                    )
                    st.markdown(
                        f":{color}[{icon} **{call.tool_name}**({params_str})]  "
                        f"`{call.duration_ms:.0f}ms`  ·  "
                        f"Deneme: {call.attempt_count}"
                    )
                    if call.error:
                        st.error(f"Hata: {call.error}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Gecikme", f"{response.total_latency_ms:.0f}ms")
            col2.metric("Araç Çağrısı", len(response.tool_calls))
            col3.metric(
                "Hata",
                sum(1 for c in response.tool_calls if c.status == ToolStatus.ERROR),
            )

    # Persist to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response.answer, "meta": response}
    )

    # Update statistics
    st.session_state.turn_count += 1
    st.session_state.total_tool_calls += len(response.tool_calls)
    st.session_state.successful_tool_calls += sum(
        1 for c in response.tool_calls if c.status == ToolStatus.SUCCESS
    )
    st.session_state.total_latency_ms += response.total_latency_ms

    st.rerun()
