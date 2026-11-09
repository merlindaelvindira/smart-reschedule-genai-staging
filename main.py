import os
import json
import secrets
import typing
import uvicorn
from fastapi import FastAPI, Request, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

# 1. ENVIRONMENT & CONFIGURATION
GEN_AI_API_KEY = os.getenv("GEN_AI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "18"))

if not GEN_AI_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("GEN_AI_API_KEY and GEMINI_API_KEY are required")

genai.configure(api_key=GEMINI_API_KEY)

model_available = True

CONFIRMATION_REPLIES = {
    "id": {
        "pending": "Terima kasih. Silakan tekan tombol Konfirmasi agar permintaan pindah jadwal Anda dikirim ke admin untuk ditinjau.",
        "none": "Saat ini belum ada jadwal yang menunggu konfirmasi. Jadwal mana yang ingin Anda pilih?",
    },
    "en": {
        "pending": "Thank you. Please press the Confirm button so your reschedule request can be sent to the admin for review.",
        "none": "There is no schedule waiting for confirmation right now. Which schedule would you like?",
    },
}


class ModelUnavailableError(Exception):
    pass

# 2. FASTAPI APPLICATION SETUP
app = FastAPI(
    title="GenAI Service - Gymnasyum Reschedule Assistant",
    description="Stateless GenAI service compliant with Application Server API.",
    version="1.0.0",
    docs_url="/docs",
)

# MIDDLEWARE
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/analyze" and request.method == "POST":
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"code": "UNAUTHORIZED", "message": "Missing or invalid Bearer token"}}
                )
            token = auth_header.split("Bearer ")[-1].strip()
            if not secrets.compare_digest(token.encode(), GEN_AI_API_KEY.encode()):
                return JSONResponse(
                    status_code=401,
                    content={"error": {"code": "UNAUTHORIZED", "message": "Invalid API Key token"}}
                )
        return await call_next(request)

app.add_middleware(AuthMiddleware)

# Override FastAPI default 422 validation error to match backend 400 specification
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "The analysis request body format is invalid or missing context fields."
            }
        }
    )

# 3. SCHEMAS
class AnalyzeRequest(BaseModel):
    model: str = Field(..., description="The ID of the GenAI model to use")
    context: dict = Field(..., description="Context payload from the Application Server")

class AnalyzeResponse(BaseModel):
    intent: str = Field(..., description="Detected intent: Reschedule, Cancel, Schedule inquiry, Confirmation, Other")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    extracted: dict = Field(default_factory=dict, description="Extracted entities")
    recommendedScheduleIds: list[str] = Field(default_factory=list, description="Recommended schedule UUIDs")
    reply: str = Field(..., description="Drafted response for the customer")
    nextState: str = Field(..., description="Next conversation state: active, awaiting_confirmation, or submitted")
    model: str = Field(..., description="The model used for generation")
    modelVersion: str = Field(..., description="The version of the model")

# 4. CORE AI PROCESSING PIPELINE
def analyze_reschedule_context(context: dict, model_name: str = GEMINI_MODEL, model_version: str = "v1") -> dict:
    global model_available

    messages = context.get("messages", [])
    if not messages:
        return {
            "intent": "Other",
            "confidence": 0.0,
            "extracted": {},
            "recommendedScheduleIds": [],
            "reply": "Pesan tidak ditemukan atau kosong.",
            "nextState": "active",
            "model": model_name,
            "modelVersion": model_version
        }

    full_chat_history = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

    # Ambil pesan terakhir untuk RAG
    user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    latest_user_message = user_messages[-1] if user_messages else ""

    # =========================================================
    # RAG RETRIEVAL (Terintegrasi dengan aman tanpa NameError)
    # =========================================================
    relevant_context = """
    1. Reschedule: Boleh jika ada sisa kuota. Tawarkan jadwal relevan.
    2. Cancel: Belum bisa diproses lewat chat. Arahkan customer ke menu booking atau resepsionis, dan jangan meminta konfirmasi pembatalan.
    3. Schedule Inquiry: Tanyakan spesifik kelas apa.
    4. Confirmation: Customer menyetujui jadwal yang sedang diusulkan.
    5. Other: Jawab pertanyaan pelanggan berdasarkan informasi fasilitas, harga, atau aturan yang tersedia di dalam konteks. Jika maklumat yang ditanyakan sama sekali tidak ada di sini, barulah arahkan ke resepsionis.
    """

    if globals().get('collection') is not None and latest_user_message:
        try:
            results = collection.query(query_texts=[latest_user_message], n_results=2)
            if results and results.get('documents') and results['documents'][0]:
                relevant_context += "\n\nINFORMASI TAMBAHAN DARI DATABASE:\n" + " | ".join(results['documents'][0])
        except Exception as e:
            print(f"⚠️ RAG Query Error: {e}")

    # =========================================================
    # DEBUGGING PRINT
    # =========================================================
    print("\n" + "="*50)
    print("DEBUG OUTPUT: RELEVANT CONTEXT (YANG DIKIRIM KE AI)")
    print("="*50)
    print(relevant_context)
    print("="*50 + "\n")
    # =========================================================

    source_sch = context.get("sourceSchedule", {})
    source_text = (
        f"Judul: {source_sch.get('title', 'N/A')} | Waktu: {source_sch.get('startsAt', 'N/A')}"
        if source_sch else "Tidak ada jadwal aktif."
    )

    session = context.get("session") or {}
    session_status = session.get("status", "active")
    proposed_schedule_id = session.get("proposedScheduleId")
    remaining_quota = (context.get("eligibility") or {}).get("remainingQuota", "N/A")

    candidate_schedules = context.get("candidateSchedules", [])
    valid_candidate_ids = []
    candidates_text = ""

    if candidate_schedules:
        for cand in candidate_schedules:
            sch = cand.get("schedule", {})
            sch_id = sch.get("id")
            if sch_id:
                valid_candidate_ids.append(sch_id)
                candidates_text += f"- ID: {sch_id} | Kelas: {sch.get('title')} | Mulai: {sch.get('startsAt')} | Sisa Kuota: {cand.get('remainingCapacity')}\n"
    else:
        candidates_text = "Tidak ada kandidat jadwal alternatif dari backend."

    # Validate variable name
    prompt = f"""
    Kamu adalah asisten admin dan konsultan kebugaran di Gymnasyum yang sangat ramah dan profesional.
    Analisis pesan pelanggan dan hasilkan respons dalam format JSON murni.

    INFORMASI ATURAN GYM & FASILITAS (RAG):
    {relevant_context}

    JADWAL BOOKING SAAT INI:
    {source_text}

    STATUS SESI: {session_status}
    SISA KUOTA MEMBER: {remaining_quota}

    KANDIDAT JADWAL:
    {candidates_text}

    RIWAYAT PERCAKAPAN:
    {full_chat_history}

    PENTING SOAL WAKTU: Abaikan zona waktu UTC (huruf Z) pada sistem. Anggap angka jam yang tertera (misalnya 18:00 atau 08:00) SUDAH merupakan waktu lokal gym. JANGAN tambahkan +7 jam. (Contoh: 18:00 adalah jam 6 sore / malam).

    ATURAN JSON KETAT:
    1. "intent": Pilih SATU dari: ["Reschedule", "Cancel", "Schedule inquiry", "Confirmation", "Other"].
    2. "recommendedScheduleIds": Jika "intent" adalah Reschedule DAN ada jadwal yang cocok dengan permintaan waktu/hari user, masukkan ID jadwalnya (maksimal 1). JIKA TIDAK ADA YANG COCOK ATAU BUKAN RESCHEDULE, KOSONGKAN []. JANGAN MENGARANG ID.
    3. "extracted": JSON object berisi (waktu, hari, alasan). Jika kosong, isi {{}}.
    4. "reply": Balas pesan user. Jika mengusulkan jadwal, sebutkan nama kelas dan waktunya sesuai angka jam di kandidat. WAJIB balas dengan BAHASA YANG SAMA dengan pesan user.
    5. "confidence": Float 0.0 - 1.0.
    6. "language": "id" jika pesan terakhir user berbahasa Indonesia, "en" jika berbahasa Inggris.
    7. Jika STATUS SESI adalah awaiting_confirmation dan pesan terakhir user menyetujui usulan, gunakan intent Confirmation. Jika user meminta jadwal lain, gunakan intent Reschedule.
    8. "reply" harus teks biasa tanpa markdown dan tidak boleh menyebut ID jadwal.
    9. Jangan menyatakan booking sudah berpindah. Perpindahan baru terjadi setelah admin menyetujui permintaan.
    10. Jika mengusulkan jadwal, "reply" wajib meminta konfirmasi customer.
    """

    try:
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )

        model_instance = genai.GenerativeModel(model_name)
        response = model_instance.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": GEMINI_TIMEOUT_SECONDS},
        )
    except Exception as e:
        model_available = False
        print(f"GenAI Execution Error: {e}")
        raise ModelUnavailableError() from e

    model_available = True
    llm_output = json.loads(response.text)

    raw_recommended_ids = llm_output.get("recommendedScheduleIds", [])
    if isinstance(raw_recommended_ids, list):
        final_recommended_ids = [sid for sid in raw_recommended_ids if sid in valid_candidate_ids]
    else:
        final_recommended_ids = []

    # Logic State
    final_intent = llm_output.get("intent", "Other")
    final_reply = str(llm_output.get("reply", "")).strip()[:2000]

    if final_intent == "Reschedule" and final_recommended_ids:
        next_state = "awaiting_confirmation"
    elif final_intent == "Confirmation":
        replies = CONFIRMATION_REPLIES.get(llm_output.get("language"), CONFIRMATION_REPLIES["id"])
        if session_status == "awaiting_confirmation" and proposed_schedule_id in valid_candidate_ids:
            final_intent = "Reschedule"
            final_recommended_ids = [proposed_schedule_id]
            next_state = "awaiting_confirmation"
            final_reply = replies["pending"]
        else:
            final_recommended_ids = []
            next_state = "active"
            final_reply = replies["none"]
    else:
        next_state = "active"
        final_recommended_ids = []

    return {
        "intent": final_intent,
        "confidence": float(llm_output.get("confidence", 0.9)),
        "extracted": llm_output.get("extracted", {}),
        "recommendedScheduleIds": final_recommended_ids,
        "reply": final_reply,
        "nextState": next_state,
        "model": model_name,
        "modelVersion": model_version
    }

# 5. ENDPOINTS & CONTROLLERS
@app.get("/health", tags=["Monitoring"], summary="Health check endpoint")
def health_check():
    return {
        "status": "ok" if model_available else "degraded",
        "model": GEMINI_MODEL,
        "modelVersion": "v1"
    }

@app.post(
    "/analyze",
    tags=["Core GenAI Engine"],
    response_model=AnalyzeResponse,
    summary="Analyze chat context and generate response"
)
def analyze_endpoint(request: AnalyzeRequest):

    if not request.context or "messages" not in request.context:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INVALID_REQUEST", "message": "Missing messages in context payload"}}
        )

    try:
        response_payload = analyze_reschedule_context(
            context=request.context,
            model_version="v1"
        )
        return response_payload
    except ModelUnavailableError:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "MODEL_UNAVAILABLE", "message": "The GenAI model is temporarily unavailable"}}
        )
    except Exception as e:
        print(f"Endpoint Error: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": {"code": "ANALYSIS_FAILED", "message": "The analysis process failed"}}
        )

# 6. APPLICATION RUNNER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8787))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
