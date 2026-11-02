import os
import re
import json
import joblib
from typing import Dict, Any, List
from fastapi import FastAPI, Header, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import google.generativeai as genai
import chromadb

# 1. ENVIRONMENT & SECURITY CONFIGURATION
GEN_AI_API_KEY = os.getenv("GEN_AI_API_KEY", "smart_reschedule_assistant_api_key")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    llm_model = None

# Initialize local ML & RAG models safely
try:
    loaded_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    loaded_classifier = joblib.load("models/best_intent_classifier.joblib")
except Exception:
    loaded_vectorizer = None
    loaded_classifier = None

try:
    chroma_client = chromadb.PersistentClient(path="./gym_vector_db")
    collection = chroma_client.get_or_create_collection(name="gym_policies")
except Exception:
    collection = None


def verify_bearer_token(authorization: str = Header(None)) -> str:
    """Validates Authorization: Bearer <token> against backend security requirement."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split("Bearer ")[-1].strip()
    if token != GEN_AI_API_KEY:
        return None
    return token


# 2. FASTAPI APPLICATION SETUP
app = FastAPI(
    title="GenAI Service - Gymnasyum Reschedule Assistant",
    description="Stateless GenAI service compliant with Application Server API.",
    version="1.0.0",
    docs_url="/docs",
)


# 3. OPENAPI SCHEMAS
class AnalyzeRequest(BaseModel):
    model: str = Field(..., description="The ID of the GenAI model to use")
    context: Dict[str, Any] = Field(..., description="Context payload from the Application Server")

class AnalyzeResponse(BaseModel):
    intent: str = Field(..., description="Detected intent: Reschedule, Cancel, Schedule inquiry, Confirmation, Other")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    extracted: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    recommendedScheduleIds: List[str] = Field(default_factory=list, description="Recommended schedule UUIDs")
    reply: str = Field(..., description="Drafted response for the customer")
    nextState: str = Field(..., description="Next conversation state: active, awaiting_confirmation, or submitted")
    model: str = Field(..., description="The model used for generation")
    modelVersion: str = Field(..., description="The version of the model")


# 4. CORE AI PROCESSING PIPELINE
def clean_text(text: str) -> str:
    """Cleans text by lowercasing, removing punctuation, and trimming extra spaces."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analyze_reschedule_context(context: dict, model_name: str = "gemini-model", model_version: str = "v1") -> dict:

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

    # Extract user messages & context strings
    user_messages = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    latest_user_message = user_messages[-1] if user_messages else messages[-1].get("content", "")
    full_chat_history = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages])

    # Local ML classification fallback logic
    cleaned_message = clean_text(latest_user_message)
    ml_confidence = 0.95
    mapped_intent = "Other"

    if loaded_vectorizer and loaded_classifier:
        try:
            message_vector = loaded_vectorizer.transform([cleaned_message])
            raw_intent = loaded_classifier.predict(message_vector)[0]
            if hasattr(loaded_classifier, "predict_proba"):
                ml_confidence = float(max(loaded_classifier.predict_proba(message_vector)[0]))
            
            intent_mapping = {
                "reschedule": "Reschedule",
                "cancel": "Cancel",
                "schedule_inquiry": "Schedule inquiry",
                "confirmation": "Confirmation"
            }
            mapped_intent = intent_mapping.get(str(raw_intent).lower(), "Other")
        except Exception:
            pass

    # RAG policy retrieval
    relevant_context = "Gunakan aturan umum gym."
    if collection:
        try:
            results = collection.query(query_texts=[cleaned_message], n_results=2)
            if results and results.get('documents') and results['documents'][0]:
                relevant_context = " | ".join(results['documents'][0])
        except Exception:
            pass

    # Extract source and candidate schedules from context
    source_sch = context.get("sourceSchedule", {})
    source_text = f"Judul: {source_sch.get('title', 'N/A')} | Waktu: {source_sch.get('startsAt', 'N/A')}" if source_sch else "Tidak ada jadwal aktif terdeteksi."

    candidate_schedules = context.get("candidateSchedules", [])
    valid_candidate_ids = [c.get("schedule", {}).get("id") for c in candidate_schedules if c.get("schedule", {}).get("id")]
    
    candidates_text = ""
    for cand in candidate_schedules:
        sch = cand.get("schedule", {})
        candidates_text += f"- ID: {sch.get('id')} | Kelas: {sch.get('title')} | Mulai: {sch.get('startsAt')} | Kuota: {cand.get('remainingCapacity')}\n"
    
    if not candidates_text:
        candidates_text = "Tidak ada kandidat jadwal alternatif yang tersedia dari backend."

    # Gemini LLM Prompt Construction
    prompt = f"""
Kamu adalah asisten admin dan konsultan kebugaran di Gymnasyum yang ramah dan profesional.
    
INFORMASI ATURAN GYM & FASILITAS (RAG):
{relevant_context}

JADWAL BOOKING SAAT INI:
{source_text}
    
RIWAYAT PERCAKAPAN:
{full_chat_history}
    
KANDIDAT JADWAL:
{candidates_text}
    
TUGAS UTAMA:
1. Buat SATU paragraf balasan WhatsApp dalam bahasa Indonesia yang ramah, sopan, dan berempati.
2. JIKA pelanggan bertanya tentang fasilitas/sauna/informasi umum, jawablah berdasarkan INFORMASI ATURAN GYM (RAG).
3. JIKA menawarkan jadwal pengganti, kamu WAJIB meminta konfirmasi eksplisit kepada pelanggan.
4. JANGAN PERNAH menyatakan booking sudah berhasil diubah (karena butuh persetujuan admin).
5. Pilih ID jadwal dari KANDIDAT JADWAL di atas yang sesuai. JANGAN MENGARANG ID.
    
OUTPUT JSON FORMAT ONLY:
{{
  "confidence": 0.95,
  "extracted": {{
    "requestedDate": "YYYY-MM-DD atau null",
    "requestedTime": "HH:MM atau null",
    "reason": "Alasan pelanggan atau null"
  }},
  "recommendedScheduleIds": ["SELECTED_CANDIDATE_ID_OR_EMPTY"],
  "reply": "Draft teks balasan WhatsApp..."
}}
"""

    confidence = max(ml_confidence, 0.85)
    try:
        # Call Gemini
        response = llm_model.generate_content(prompt)
        raw_response = response.text.strip()
        json_str = re.sub(r'```(?:json)?\s*|\s*```', '', raw_response).strip()
        llm_output = json.loads(json_str)
        confidence = float(llm_output.get("confidence", confidence))
    except Exception:
        # 6. SMART FALLBACK
        first_cand = valid_candidate_ids[:1] if valid_candidate_ids else []
     
        if mapped_intent == "Other":
            fallback_reply = "Halo kak! Terima kasih sudah menghubungi Gymnasyum. Untuk detail informasi fasilitas seperti sauna, silakan tanyakan ke resepsionis kami di lokasi ya."
        elif mapped_intent == "Schedule inquiry":
            fallback_reply = "Halo kak! Saya dapat membantu mengecek jadwal kelas. Boleh tahu jadwal kelas apa yang ingin dicek?"
        elif mapped_intent == "Cancel":
            fallback_reply = "Halo kak! Mohon konfirmasi kembali apakah kakak yakin ingin membatalkan sesi kelas ini?"
        elif first_cand:
            fallback_reply = "Halo kak! Saya menemukan jadwal pengganti yang sesuai. Apakah kakak ingin saya pindahkan ke jadwal tersebut?"
        else:
            fallback_reply = "Halo kak! Saat ini belum ada jadwal alternatif yang sesuai. Apakah ada pilihan waktu lain?"

        llm_output = {
            "extracted": {},
            "recommendedScheduleIds": first_cand,
            "reply": fallback_reply
        }

    # 7. FINAL VALIDATION & ASSEMBLY
    final_recommended_ids = [sid for sid in llm_output.get("recommendedScheduleIds", []) if sid in valid_candidate_ids]
    
    if final_recommended_ids:
        final_intent = "Reschedule"
        next_state = "awaiting_confirmation"
    else:
        final_intent = mapped_intent
        next_state = "active"

    reply_text = str(llm_output.get("reply", "")).strip() or "Halo kak, ada yang bisa saya bantu?"

    return {
        "intent": final_intent,
        "confidence": round(confidence, 2),
        "extracted": llm_output.get("extracted", {}),
        "recommendedScheduleIds": final_recommended_ids,
        "reply": reply_text[:2000],
        "nextState": next_state,
        "model": model_name,
        "modelVersion": model_version
    }


# 5. ENDPOINTS & CONTROLLERS
@app.get("/health", tags=["Monitoring"], summary="Health check endpoint for deployment monitoring")
def health_check():
    """Operational health monitoring endpoint."""
    return {
        "status": "ok",
        "model": "gemini-model",
        "modelVersion": "v1"
    }


@app.post(
    "/analyze",
    tags=["Core GenAI Engine"],
    response_model=AnalyzeResponse,
    summary="Analyze chat context and generate response"
)
def analyze_endpoint(request: AnalyzeRequest, token: str = Depends(verify_bearer_token)):
    """Main webhook endpoint invoked by Application Server."""
    
    # Security Token Check
    if token is None:
        return JSONResponse(
            status_code=401,
            content={"error": {"code": "UNAUTHORIZED", "message": "Unauthorized access: Missing or invalid Bearer token"}}
        )

    # Request Body Validation
    if not request.context or "messages" not in request.context:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": "INVALID_REQUEST", "message": "The analysis request is invalid - missing messages context"}}
        )

    try:
        response_payload = analyze_reschedule_context(
            context=request.context,
            model_name=request.model,
            model_version="v1"
        )
        return response_payload
    except Exception:
        return JSONResponse(
            status_code=502,
            content={"error": {"code": "ANALYSIS_FAILED", "message": "The analysis could not be completed"}}
        )


# 6. APPLICATION RUNNER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
