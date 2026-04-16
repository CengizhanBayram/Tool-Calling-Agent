"""Response synthesizer — generates the final natural-language answer.

Combines all tool results (successes and failures) into a coherent,
friendly Turkish response using a Gemini API call.
"""

from typing import Any, Dict, List

import google.generativeai as genai

from src.config import Settings, settings
from src.logger import log
from src.models import (
    FraudReason,
    ToolCall,
    ToolStatus,
    Transaction,
    TransactionStatus,
    UserDetails,
    ValidationResult,
)


SYNTHESIS_SYSTEM_PROMPT = """Sen profesyonel ve sıcakkanlı bir Türkçe müşteri destek asistanısın.
Görevin: sana verilen araç (tool) sonuçlarını kullanarak müşteriye net ve doğal bir Türkçe yanıt oluşturmak.

KURALLAR:
1. Her zaman Türkçe yanıt ver.
2. Müşteriye samimi ve yardımsever bir ton kullan — gerçek bir insanla konuşuyorsun.
3. Belirttiğin her bilgi için hangi araçtan geldiğini belirt: [Kaynak: araç_adı]
4. Bir araç hata döndürdüyse bunu doğal bir dille anlat — ham hata mesajlarını asla gösterme.
5. İsteği tamamlayamadıysan kullanıcıya ne bulamadığını ve ne yapabileceğini açıkla.
6. Dolandırıcılık / red nedeni için: nedeni açıkla, varsa çözüm adımlarını listele.
7. Araç sonuçlarında olmayan bilgileri asla uydurma.
8. Araç sonuçları boşsa (hiç araç çalışmadıysa): kullanıcının sorusuna genel bilgiyle yardımcı ol
   veya kibarca yönlendir.
9. Yanıtını şu uzunlukta tut:
   - Basit hesap sorgusunda: 2-3 cümle
   - İşlem geçmişinde: madde listesi
   - Dolandırıcılık açıklamasında: sebep + yapılacaklar listesi
10. Yanıtın sonunda başka bir konuda yardımcı olup olamayacağını sor.

ARAÇ SONUÇLARI FORMAT:
Her araç sonucu şu etiketle gelir: [ARAÇ: araç_adı | DURUM: başarılı/hata]
Başarılı sonuçlar gerçek veriyi içerir.
Hatalı sonuçlar hata mesajını içerir — bunu yumuşatarak aktar.

Yanıt uzunluğu ve tonu her zaman müşterinin sorusuna göre ayarla."""


class ResponseSynthesizer:
    """Generates the final Turkish customer support response.

    Builds a structured context string from all tool results (including
    partial failures) and asks Gemini to synthesise a natural reply.

    Args:
        cfg: Application settings. Defaults to the module-level singleton.
    """

    def __init__(self, cfg: Settings = settings) -> None:
        genai.configure(api_key=cfg.google_api_key)
        self._model_name = cfg.gemini_model
        self._max_tokens = cfg.max_tokens

    def synthesize(
        self,
        user_query: str,
        tool_calls: Dict[str, ToolCall],
        validation_results: Dict[str, ValidationResult],
    ) -> str:
        """Generate a natural Turkish answer from tool execution results.

        Args:
            user_query: The original user message.
            tool_calls: Mapping of step_id → :class:`~src.models.ToolCall`.
            validation_results: Mapping of step_id → :class:`~src.models.ValidationResult`.

        Returns:
            Natural-language Turkish response string.
        """
        tool_context = self._build_tool_context(tool_calls, validation_results)

        if tool_context:
            user_content = (
                f"Müşteri sorusu: {user_query}\n\n"
                f"Araç sonuçları:\n{tool_context}"
            )
        else:
            user_content = (
                f"Müşteri sorusu: {user_query}\n\n"
                f"[Bu soru için herhangi bir araç çalıştırılmadı — "
                f"genel bilgiyle veya yönlendirme ile yanıt ver.]"
            )

        log.debug(f"Synthesizing response for query: {user_query[:80]}…")

        # Try Gemini with one retry on rate-limit (429)
        for attempt in range(2):
            try:
                model = genai.GenerativeModel(
                    model_name=self._model_name,
                    system_instruction=SYNTHESIS_SYSTEM_PROMPT,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self._max_tokens,
                    ),
                )
                response = model.generate_content(user_content)
                return response.text.strip()
            except Exception as exc:
                err = str(exc)
                if "429" in err and attempt == 0:
                    import time
                    wait = 30
                    log.warning(f"Rate limit hit, waiting {wait}s before retry…")
                    time.sleep(wait)
                    continue
                log.error(f"Response synthesis failed (attempt {attempt+1}): {exc}")
                break

        # API unavailable — build a structured Turkish response from raw tool data
        return self._template_fallback(tool_calls)

    # ------------------------------------------------------------------
    # Template fallback (no API required)
    # ------------------------------------------------------------------

    def _template_fallback(self, tool_calls: Dict[str, ToolCall]) -> str:
        """Generate a structured Turkish response without any LLM API call.

        Used when Gemini is unavailable (rate limit, outage, etc.).
        Produces readable output directly from tool result dataclasses.
        """
        parts: List[str] = []
        has_content = False

        for call in tool_calls.values():
            if call.status == ToolStatus.ERROR:
                parts.append(f"⚠️ {call.error}")
                has_content = True
                continue
            if call.status == ToolStatus.SKIPPED or call.result is None:
                continue

            has_content = True
            r = call.result

            if isinstance(r, UserDetails):
                status_map = {"active": "Aktif ✅", "suspended": "Askıya Alınmış ⚠️", "closed": "Kapatılmış ❌"}
                status_val = r.account_status.value if hasattr(r.account_status, "value") else str(r.account_status)
                parts.append(
                    f"**Hesap Bilgileri** [Kaynak: get_user_details]\n"
                    f"- Ad Soyad: {r.full_name}\n"
                    f"- E-posta: {r.email}\n"
                    f"- Durum: {status_map.get(status_val, status_val)}\n"
                    f"- Hesap Türü: {r.account_type}\n"
                    f"- Üyelik: {r.created_at}"
                )

            elif isinstance(r, list) and r and isinstance(r[0], Transaction):
                lines = [f"**Son {len(r)} İşlem** [Kaynak: get_recent_transactions]"]
                for t in r:
                    icon = {"success": "✅", "failed": "❌", "pending": "⏳", "refunded": "🔄"}.get(
                        t.status.value if hasattr(t.status, "value") else str(t.status), "❓"
                    )
                    line = f"- {icon} `{t.transaction_id}` — {t.amount:.2f} {t.currency} — {t.merchant} — {t.created_at}"
                    if t.failure_code:
                        line += f" (**{t.failure_code}**)"
                    lines.append(line)
                parts.append("\n".join(lines))

            elif isinstance(r, list) and not r:
                parts.append("Bu hesaba ait işlem kaydı bulunmamaktadır.")

            elif isinstance(r, FraudReason):
                steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(r.resolution_steps))
                dispute = "✅ Evet, itiraz hakkınız bulunmaktadır." if r.can_dispute else "❌ Bu işlem için itiraz yapılamaz."
                parts.append(
                    f"**Red Nedeni** [Kaynak: check_fraud_reason]\n\n"
                    f"{r.reason_tr}\n\n"
                    f"- **Hata Kodu:** {r.failure_code}\n"
                    f"- **Risk Sinyalleri:** {', '.join(r.risk_signals)}\n"
                    f"- **İtiraz:** {dispute}\n\n"
                    f"**Yapmanız Gerekenler:**\n{steps}"
                )

        if not has_content:
            return (
                "Merhaba! Hesabınız veya işlemlerinizle ilgili bir sorunuz varsa "
                "yardımcı olmaktan memnuniyet duyarım. "
                "Lütfen kayıtlı e-posta adresinizi veya işlem numaranızı (TXN-XXXXX) paylaşın."
            )

        result = "\n\n".join(parts)
        result += "\n\n---\nBaşka bir konuda yardımcı olabilir miyim?"
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tool_context(
        self,
        tool_calls: Dict[str, ToolCall],
        validation_results: Dict[str, ValidationResult],
    ) -> str:
        """Serialize tool results into a text block for the LLM.

        Args:
            tool_calls: All step results from the executor.
            validation_results: Validation outcomes for successful calls.

        Returns:
            Multi-section string, one section per tool call.
        """
        sections: List[str] = []
        for step_id, call in tool_calls.items():
            vr = validation_results.get(step_id)
            if call.status == ToolStatus.SUCCESS and (vr is None or vr.valid):
                content = self._serialize_result(call.result)
                sections.append(
                    f"[ARAÇ: {call.tool_name} | DURUM: başarılı | adım: {step_id}]\n{content}"
                )
            elif call.status == ToolStatus.SUCCESS and vr and not vr.valid:
                sections.append(
                    f"[ARAÇ: {call.tool_name} | DURUM: doğrulama hatası | adım: {step_id}]\n"
                    f"Doğrulama hatası: {vr.error}"
                )
            elif call.status == ToolStatus.ERROR:
                sections.append(
                    f"[ARAÇ: {call.tool_name} | DURUM: hata | adım: {step_id}]\n"
                    f"Hata mesajı: {call.error}"
                )
            elif call.status == ToolStatus.SKIPPED:
                sections.append(
                    f"[ARAÇ: {call.tool_name} | DURUM: atlandı | adım: {step_id}]\n"
                    f"Bu adım üst bağımlılık hatası nedeniyle atlandı."
                )
        return "\n\n".join(sections)

    @staticmethod
    def _serialize_result(result: Any) -> str:
        """Convert a tool result dataclass or list to readable text.

        Args:
            result: Return value of any of the three tool functions.

        Returns:
            Human-readable multi-line string.
        """
        if isinstance(result, UserDetails):
            status_val = (
                result.account_status.value
                if hasattr(result.account_status, "value")
                else str(result.account_status)
            )
            status_label = {
                "active": "Aktif ✅",
                "suspended": "Askıya Alınmış ⚠️",
                "closed": "Kapatılmış ❌",
            }.get(status_val, status_val)
            return (
                f"Kullanıcı: {result.full_name} ({result.email})\n"
                f"ID: {result.user_id}\n"
                f"Hesap Durumu: {status_label}\n"
                f"Hesap Türü: {result.account_type}\n"
                f"Risk Skoru: {result.risk_score}\n"
                f"Üyelik Tarihi: {result.created_at}"
            )

        if isinstance(result, list) and result and isinstance(result[0], Transaction):
            lines = [f"Son {len(result)} işlem (yeniden eskiye):"]
            for t in result:
                status_icon = {
                    TransactionStatus.SUCCESS: "✅",
                    TransactionStatus.FAILED: "❌",
                    TransactionStatus.PENDING: "⏳",
                    TransactionStatus.REFUNDED: "🔄",
                }.get(t.status, "❓")
                line = (
                    f"  {status_icon} {t.transaction_id} | "
                    f"{t.amount:.2f} {t.currency} | "
                    f"{t.merchant} | "
                    f"{t.created_at} | "
                    f"Durum: {t.status}"
                )
                if t.failure_code:
                    line += f" [{t.failure_code}]"
                lines.append(line)
            return "\n".join(lines)

        if isinstance(result, list) and not result:
            return "Bu kullanıcı için işlem kaydı bulunamadı."

        if isinstance(result, FraudReason):
            steps = "\n".join(
                f"  {i + 1}. {s}" for i, s in enumerate(result.resolution_steps)
            )
            dispute_info = (
                "Evet — itiraz hakkınız mevcuttur."
                if result.can_dispute
                else "Hayır — bu işlem türü için itiraz yapılamaz."
            )
            return (
                f"Red Sebebi: {result.reason_tr}\n"
                f"Hata Kodu: {result.failure_code}\n"
                f"Risk Sinyalleri: {', '.join(result.risk_signals)}\n"
                f"İtiraz Edilebilir mi: {dispute_info}\n"
                f"Çözüm Adımları:\n{steps}\n"
                f"Tespit Zamanı: {result.detected_at}\n"
                f"İnceleyen Sistem: {result.reviewed_by}"
            )

        return str(result)
