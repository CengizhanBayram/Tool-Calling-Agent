# 🤖 Tool-Calling Agent — Production-Grade Build

Türkçe müşteri desteği için üretim kalitesinde çok aşamalı LLM agent sistemi.
Hesap sorgulama, işlem geçmişi ve fraud (dolandırıcılık) nedeni tespiti yapan
üç araç; bağımlılık grafiği tabanlı planlama, paralel yürütme ve Gemini 2.5 Flash
ile doğal dil yanıt sentezi içerir.

---

## İçindekiler

1. [Kurulum (Quick Start)](#1-kurulum-quick-start)
2. [Proje Yapısı](#2-proje-yapısı)
3. [Mimari Genel Bakış](#3-mimari-genel-bakış)
4. [Mimari Kararlar — Detaylı Açıklama](#4-mimari-kararlar--detaylı-açıklama)
   - [Neden özel pipeline, Claude/Gemini native tool_use değil?](#41-neden-özel-pipeline-claudegemini-native-tool_use-değil)
   - [State yönetimi nasıl kurgulandı?](#42-state-yönetimi-nasıl-kurgulandı)
   - [Bağımlılık grafiği ve $ref sistemi](#43-bağımlılık-grafiği-ve-ref-sistemi)
   - [Halüsinasyon önleme: Intent Classifier tasarımı](#44-halüsinasyon-önleme-intent-classifier-tasarımı)
   - [Hata izolasyonu ve graceful degradation](#45-hata-izolasyonu-ve-graceful-degradation)
   - [Canlı veri güncellemesi (db.reload)](#46-canlı-veri-güncellemesi-dbreload)
5. [Araçlar ve İstisna Hiyerarşisi](#5-araçlar-ve-i̇stisna-hiyerarşisi)
6. [Yeni Araç Ekleme (4 Adım)](#6-yeni-araç-ekleme-4-adım)
7. [Test Koşturma](#7-test-koşturma)

---

## 1. Kurulum (Quick Start)

### Gereksinimler

- Python 3.10+
- Google AI Studio API anahtarı → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Adım adım kurulum

```bash
# 1. Projeyi klonlayın / indirin
cd agent_project

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 3. Ortam değişkenlerini ayarlayın
cp .env.example .env
```

`.env` dosyasını açıp aşağıdaki satırı kendi anahtarınızla doldurun:

```env
GOOGLE_API_KEY=AIza-buraya-gercek-anahtarinizi-yazin
GEMINI_MODEL=gemini-2.5-flash
MAX_TOKENS=2048
DATA_DIR=./data
```

```bash
# 4a. Streamlit arayüzünü başlatın (önerilen)
streamlit run app.py
# Tarayıcıda açılır: http://localhost:8501

# 4b. Komut satırı — tek sorgu
python main.py --query "ali@sirket.com son ödeme neden reddedildi?"

# 4c. Komut satırı — interaktif sohbet
python main.py --interactive

# 4d. 12 senaryo otomatik test
python main.py --run-tests

# 4e. Performans benchmark
python main.py --benchmark
```

### Hızlı doğrulama

```bash
python -c "
from src.tools import get_user_details
u = get_user_details('ali@sirket.com')
print(u.full_name, u.account_status)
"
# Çıktı: Ali Yılmaz active
```

---

## 2. Proje Yapısı

```
agent_project/
├── data/
│   ├── users.json            # 8 kullanıcı kaydı
│   ├── transactions.json     # 20 işlem kaydı (6 farklı hata kodu)
│   └── fraud_reasons.json    # 7 fraud analizi
│
├── src/
│   ├── config.py             # Pydantic BaseSettings — tüm config buradan
│   ├── models.py             # Tüm dataclass'lar ve enum'lar
│   ├── logger.py             # Yapılandırılmış tool-call logger
│   ├── database.py           # MockDatabase — JSON oku/yaz/sorgula
│   ├── tools.py              # 3 core tool fonksiyonu + özel exception'lar
│   ├── tool_registry.py      # Gemini API tool tanımları + fonksiyon haritası
│   ├── intent_classifier.py  # ← TEMEL: Eksik param tespiti
│   ├── tool_planner.py       # ← TEMEL: Bağımlılık DAG'ı
│   ├── tool_executor.py      # ← TEMEL: Paralel yürütme + retry
│   ├── result_validator.py   # Çıktı şema doğrulama
│   ├── response_synthesizer.py # Gemini ile Türkçe yanıt + template fallback
│   ├── conversation.py       # Çok turlu hafıza yöneticisi
│   └── agent.py              # Ana orkestratör
│
├── tests/
│   ├── test_tools.py         # 29 birim test
│   ├── test_intent_classifier.py
│   ├── test_tool_executor.py # Retry, $ref, hata izolasyonu
│   └── test_agent.py         # Entegrasyon testleri (mock'lu)
│
├── main.py                   # CLI giriş noktası
├── app.py                    # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Mimari Genel Bakış

Her kullanıcı mesajı 5 aşamadan geçer:

```
KULLANICI MESAJI
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  AŞAMA 1 — INTENT CLASSIFIER                        │
│  • Gemini ile intent tespiti (fraud / history / ...) │
│  • Eksik parametre var mı? (email, txn_id, user_id)  │
│  • Eksikse → kullanıcıya sor, tool ÇAĞIRMA           │
└──────────────────────┬──────────────────────────────┘
                       │ can_proceed = True
                       ▼
┌─────────────────────────────────────────────────────┐
│  AŞAMA 2 — TOOL PLANNER                             │
│  • Hangi araçlar gerekli?                           │
│  • Bağımlılık sırası (DAG) kur                      │
│  • $ref placeholder'larını yerleştir                │
└──────────────────────┬──────────────────────────────┘
                       │ ExecutionPlan
                       ▼
┌─────────────────────────────────────────────────────┐
│  AŞAMA 3 — TOOL EXECUTOR                            │
│  • Bağımsız adımları ThreadPoolExecutor ile paralel  │
│  • Bağımlı adımları sıralı çalıştır                 │
│  • Geçici hatalarda 3×retry (exponential backoff)   │
│  • Domain hataları → direkt ERROR, retry yok         │
└──────────────────────┬──────────────────────────────┘
                       │ Dict[step_id → ToolCall]
                       ▼
┌─────────────────────────────────────────────────────┐
│  AŞAMA 4 — RESULT VALIDATOR                         │
│  • Tip kontrolü (UserDetails, List[Transaction] …)  │
│  • Zorunlu alan kontrolü                            │
│  • Uyarı / hata etiketleme                          │
└──────────────────────┬──────────────────────────────┘
                       │ ValidationResult'lar
                       ▼
┌─────────────────────────────────────────────────────┐
│  AŞAMA 5 — RESPONSE SYNTHESIZER                     │
│  • Gemini 2.5 Flash ile doğal Türkçe yanıt          │
│  • API limit/hata → template fallback (API'siz)     │
│  • Kaynak atıfı: [Kaynak: araç_adı]                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              KULLANICIYA YANIT
        (doğal dil + araç iz kaydı)
```

---

## 4. Mimari Kararlar — Detaylı Açıklama

### 4.1 Neden özel pipeline, Claude/Gemini native tool_use değil?

**Sorun:** Gemini ve Claude'un native tool_use özelliği LLM'in hangi araçları
hangi sırayla çağıracağına kendisinin karar vermesini sağlar. Bu yaklaşım
basit senaryolar için yeterlidir ama birkaç kritik sorunu çözmez:

| Konu | Native tool_use | Bu mimari |
|---|---|---|
| Eksik parametre | LLM bazen tahmin eder / uydurur | Classifier önce kontrol eder, asla tahmin yapılmaz |
| Bağımlılık sırası | LLM'e güvenilir | DAG planner kesin sıra garantisi verir |
| Paralel çalışma | Desteklenmiyor | ThreadPoolExecutor ile bağımsız adımlar paralel |
| Retry | Yok | 3× exponential backoff sadece geçici hatalarda |
| Hata izolasyonu | Bir araç hatası tüm akışı durdurur | Her araç bağımsız; biri hata verirse diğerleri çalışır |
| Gözlemlenebilirlik | Siyah kutu | Her adım loglanır: araç, parametre, süre, deneme sayısı |

**Karar:** Planlama ve yürütme LLM'den ayrı tutuldu. LLM yalnızca iki noktada
kullanılıyor: intent sınıflandırma ve nihai yanıt sentezi.

---

### 4.2 State yönetimi nasıl kurgulandı?

Agent'ın "hafızası" iki katmandan oluşur:

#### Katman 1 — Mesaj Geçmişi (ConversationManager)

```python
# src/conversation.py
self._histories: Dict[str, List[Dict]] = {}  # session_id → mesajlar
```

- Her session için son **20 mesaj** saklanır (bant genişliği tasarrufu).
- `IntentClassifier`, yeni mesajı sınıflandırmadan önce geçmişi tarar:
  email, `USR-XXX`, `TXN-XXXXX` pattern'larını regex ile çıkarır.
- Bu sayede kullanıcı 2. turda email yazmak zorunda kalmaz.

```
Tur 1: "ali@sirket.com hesabım hakkında bilgi ver"
         → email = "ali@sirket.com" geçmişe yazılır

Tur 2: "son işlemlerimi de göster"
         → Geçmişten email okunur → get_user_details çağrılmaz
         → Sadece get_recent_transactions çalışır
```

#### Katman 2 — Parametre Önbelleği (extracted params)

```python
# src/conversation.py
self._params: Dict[str, Dict] = {}  # session_id → {user_id, email, txn_id}
```

Başarılı tool çağrılarından elde edilen veriler önbelleğe alınır:

```python
# get_user_details başarılı olursa:
cache["user_id"] = call.result.user_id   # USR-001
cache["email"]   = call.result.email      # ali@sirket.com

# get_recent_transactions başarılı olursa:
failed = [t for t in txns if t.status == FAILED]
if failed:
    cache["transaction_id"] = failed[0].transaction_id  # TXN-10041
```

`ToolPlanner`, yeni tur geldiğinde bu önbellekten okur ve gereksiz
tool çağrılarını siler. Sonuç: kullanıcı aynı sorgu için email'ini
bir kez verir, sistem defalarca sormaz.

#### Thread güvenliği

```python
self._lock = threading.RLock()  # Yeniden-girilebilir kilit
```

`RLock` kullanıldı çünkü aynı thread içinde iç içe `get_history` →
`update_params` çağrıları yapılabiliyor. Yalın `Lock` bu durumda kilitlenmeye
yol açardı.

#### Oturum izolasyonu

Her `session_id` tamamen bağımsızdır. Streamlit'te her kullanıcı sekme
açtığında `uuid4` ile benzersiz session alır. `_histories` ve `_params`
dict'leri session_id ile anahtarlanır — paylaşılan global state yoktur.

---

### 4.3 Bağımlılık grafiği ve $ref sistemi

**Sorun:** `get_recent_transactions` için `user_id` gerekir, bu da
`get_user_details`'tan gelir. Ardışık çağrıları kodlamak kırılgandır.

**Çözüm:** Planlama aşamasında `PlanStep` nesneleri oluşturulur, parametre
değerleri henüz bilinmiyorsa `$ref:step_id.field` placeholder'ı kullanılır:

```python
# src/tool_planner.py
PlanStep(
    step_id="step_2",
    tool_name="get_recent_transactions",
    parameters={
        "user_id": "$ref:step_1.user_id",  # çalışma zamanında çözülür
        "limit": 5,
    },
    depends_on=["step_1"],
)
```

Yürütme sırasında `ToolExecutor._resolve_parameters()` bu placeholder'ları
tamamlanmış adımların sonuçlarından çözer:

```python
# src/tool_executor.py
def _resolve_parameters(self, step, completed):
    for key, value in step.parameters.items():
        if value.startswith("$ref:"):
            ref_step_id, ref_field = value[5:].split(".", 1)
            resolved[key] = self._extract_field(
                completed[ref_step_id].result, ref_field
            )
```

Özel alan: `first_failed` — listedeki ilk `status==FAILED` işlemin
`transaction_id`'sini döner. Bu sayede planlama aşamasında hangi
işlemin hata verdiği bilinmeden bile plan kurulabilir.

**Short-circuit optimizasyonu:** Kullanıcı doğrudan `TXN-10041` verirse
planlayıcı 1 adımlık plan oluşturur (3 yerine):

```
Email verilince:  get_user_details → get_recent_transactions → check_fraud_reason  (3 adım)
TXN-ID verilince: check_fraud_reason  (1 adım)
```

---

### 4.4 Halüsinasyon önleme: Intent Classifier tasarımı

LLM'lerin en tehlikeli davranışı parametre uydurmaktır. Örneğin kullanıcı
"son işlemim neden reddedildi" dediğinde LLM rastgele bir email uydurabilir.

**Çözüm — iki aşamalı koruma:**

**1. Aşama — Gemini ile sınıflandırma:**

```python
# src/intent_classifier.py — CLASSIFICATION_SYSTEM_PROMPT içinden:
"""
CRITICAL RULES:
3. NEVER assume or fabricate an email address, user_id, or transaction_id.
4. Turkish phrases like "hesabım", "işlemim" imply the user's OWN account
   — email is required unless already in history.
"""
```

Gemini'ye JSON formatında yanıt üretmesi emredilir:
```json
{
  "can_proceed": false,
  "missing_required_params": ["email"],
  "clarification_needed": "E-posta adresinizi paylaşır mısınız?"
}
```

**2. Aşama — Python kodu doğrulaması:**

LLM çıktısına güvenilmez; Python tarafında da kontrol yapılır:

```python
# Sınıflandırma sonrası Python güvenlik katmanı:
if intent in ("fraud_inquiry",):
    has_id = (
        bool(provided.get("email"))
        or bool(provided.get("user_id"))
        or bool(provided.get("transaction_id"))
    )
    can_proceed = has_id  # LLM'in dediği değil, Python'ın hesapladığı
```

**3. Aşama — Fallback heuristik:**

Gemini API'si çöktüğünde bile sistem çalışır:

```python
# src/intent_classifier.py
def _fallback_classification(self, message, history_params):
    # Regex ile email/TXN-ID/USR-ID çıkar
    # Anahtar kelimelerle intent tahmin et
    # can_proceed hesapla
    # Hiçbir zaman parametre uydurma
```

**Sonuç:** Agent'ın 5 aşama boyunca hiçbirinde parametre uydurma riski yoktur.

---

### 4.5 Hata izolasyonu ve graceful degradation

**Sorun:** Bir araç hata verirse diğer araçların ve nihai yanıtın çökmemesi gerekir.

**Çözüm — Üç katmanlı hata yönetimi:**

**1. Araç seviyesi — Exception hiyerarşisi:**

```python
# src/tools.py
class UserNotFoundError(Exception): ...      # Domain hatası → retry yok
class TransactionNotFoundError(Exception): ... # Domain hatası → retry yok
class InvalidParameterError(Exception): ...   # Domain hatası → retry yok
```

vs.

```python
# Geçici hatalar (ağ, zaman aşımı) → 3× retry, exponential backoff
TRANSIENT = (ConnectionError, TimeoutError, OSError)
```

**2. Adım seviyesi — Bağımlılık yayılımı:**

Bir adım hata verirse, ona bağımlı tüm sonraki adımlar otomatik olarak
`SKIPPED` durumuna alınır:

```python
# src/tool_executor.py
failed_step_ids.add(step.step_id)
# → Bağımlı adımlar: unrunnable listesine → SKIPPED
# → SKIPPED adımlar da failed_step_ids'e → cascade propagation
```

**3. Sentez seviyesi — Template fallback:**

Gemini API'si kullanılamaz durumdaysa (rate limit, süresi dolmuş anahtar vb.)
`_template_fallback()` metodu API çağrısı yapmadan doğrudan araç sonuçlarından
Türkçe yanıt üretir:

```python
# src/response_synthesizer.py
def _template_fallback(self, tool_calls) -> str:
    # UserDetails → hesap bilgi bloğu
    # List[Transaction] → madde listesi
    # FraudReason → red nedeni + çözüm adımları
    # Hiçbir Gemini çağrısı yapılmaz
```

**Sonuç — Kısmi başarı senaryosu:**

```
get_user_details    → ✅ başarılı
get_recent_transactions → ✅ başarılı
check_fraud_reason  → ❌ hata (fraud kaydı yok)
                       ↓
Yanıt: "Son 5 işleminizi getirdim [liste]. TXN-10042 numaralı
        işlem başarılı tamamlandığından red kaydı bulunmuyor."
```

---

### 4.6 Canlı veri güncellemesi (db.reload)

**Sorun:** Üretim ortamında veri değişebilir; uygulamayı yeniden başlatmak
yerine anlık güncelleme gerekir.

**Çözüm:**

```python
# src/database.py
def reload(self) -> None:
    with self._lock:      # RLock — uçuşta olan okuma işlemleri tamamlanır
        self._load_all()  # Tüm JSON dosyaları yeniden okunur
```

`RLock` sayesinde reload sırasında yarım kalan sorgular bitmesine izin
verilir, yeni sorgular yeni veriyle çalışır. Atomik geçiş garanti edilir.

Streamlit'te **"🔄 DB Yenile"** butonu, CLI'de `/reload` komutu bu metodu çağırır.

Testlerde (Test 11 ve 12) bu mekanizma şöyle doğrulanır:

```python
# JSON'u değiştir
with open(fraud_path, "w") as f:
    json.dump(patched, f)

db.reload()             # Anlık güncelleme

response = agent.chat(...)  # Yeni veriyle yanıt

# Sonra orijinale geri dön
db.reload()
```

---

## 5. Araçlar ve İstisna Hiyerarşisi

### Araçlar

| Araç | Girdi | Çıktı | Bağımlılık |
|---|---|---|---|
| `get_user_details` | `email: str` | `UserDetails` | — |
| `get_recent_transactions` | `user_id: str, limit: int` | `List[Transaction]` | `get_user_details` |
| `check_fraud_reason` | `transaction_id: str` | `FraudReason` | `get_recent_transactions` veya doğrudan TXN-ID |

### İstisna türleri

| İstisna | Ne zaman fırlatılır | Retry? |
|---|---|---|
| `InvalidParameterError` | Email formatı yanlış, USR-/TXN- formatı yanlış | ❌ |
| `UserNotFoundError` | Email ile eşleşen kullanıcı yok | ❌ |
| `TransactionNotFoundError` | İşlem ID sistemde yok | ❌ |
| `FraudRecordNotFoundError` | İşlem var ama fraud kaydı yok (örn. başarılı işlem) | ❌ |
| `ConnectionError` / `TimeoutError` | Ağ geçici hatası | ✅ 3× |

---

## 6. Yeni Araç Ekleme (4 Adım)

**1.** `src/tools.py` içine fonksiyon ekle:
```python
def get_account_limits(user_id: str) -> AccountLimits:
    # validate → db.find_user_by_id → return typed dataclass
```

**2.** `src/tool_registry.py` içine tanım ve harita kaydı ekle:
```python
{"name": "get_account_limits", "description": "...", "input_schema": {...}}
# TOOL_FUNCTION_MAP'e de ekle
```

**3.** `src/result_validator.py` içine doğrulayıcı ekle:
```python
def validate_account_limits(self, result) -> ValidationResult: ...
# validate_all'a da kaydet
```

**4.** `src/tool_planner.py` içinde ilgili intent için adım ekle:
```python
if intent == "account_limits":
    steps.append(PlanStep("step_N", "get_account_limits", ...))
```

---

## 7. Test Koşturma

### Birim testler (API anahtarı gerekmez)

```bash
python -m pytest tests/test_tools.py tests/test_tool_executor.py -v
# 39 test — sıfır API çağrısı
```

### Entegrasyon testler (mock'lu, API anahtarı gerekmez)

```bash
python -m pytest tests/test_intent_classifier.py tests/test_agent.py -v
# 18 test — Gemini mock'lanır
```

### Tümü

```bash
python -m pytest tests/ -v
# 57 test — tümü geçmeli
```

### 12 senaryolu E2E test (gerçek API)

```bash
python main.py --run-tests
```

| # | Senaryo | Kontrol edilen davranış |
|---|---|---|
| 1 | `ali@sirket.com son ödeme neden reddedildi?` | 3 araç zinciri |
| 2 | `son işlemlerimi görmek istiyorum` (email yok) | 0 araç, soru sor |
| 3 | `ayse@ornek.com hesap durumu` | Askıya alınmış hesap |
| 4 | `bilinmeyen@email.com` | UserNotFoundError → doğal yanıt |
| 5 | `TXN-10041 neden reddedildi?` | 1 araç (short-circuit) |
| 6 | 2 turlu: email 1. turda, soru 2. turda | Geçmişten email okunur |
| 7 | `gecersiz-email hesabı` | InvalidParameterError |
| 8 | Başarılı işlemin fraud kaydı | FraudRecordNotFoundError |
| 9 | `hava nasıl?` | 0 araç, nazik yönlendirme |
| 10 | Boş mesaj | Soru sor |
| 11 | fraud_reasons.json değiştir → db.reload() | Yeni veri döner |
| 12 | users.json'a yeni kullanıcı → db.reload() | Anında bulunur |
