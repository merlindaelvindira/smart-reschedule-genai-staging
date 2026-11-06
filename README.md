# smart-reschedule-genai-staging
Stateless GenAI Microservice for Gymnasyum Smart Reschedule Assistant powered by FastAPI, Google Gemini 1.5 Flash, Scikit-learn, and ChromaDB RAG.

---

## 🛠️ Tech Stack

* **Framework:** FastAPI (Python 3.10+)
* **LLM Engine:** Google Gemini 3.6 Flash (`google-generativeai`)
* **Intent Classifier:** Scikit-learn (TF-IDF + Calibrated Classifier)
* **Vector Database (RAG):** ChromaDB
* **Server:** Uvicorn

---

## 🔑 Environment Variables

Make sure to configure the following environment variables before running the application:

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `GEN_AI_API_KEY` | Bearer token for Backend Authentication | `smart_reschedule_assistant_api_key` |
| `GEMINI_API_KEY` | Google Gemini API Authorization Key | *Required* |
| `PORT` | Server listening port | `8000` |

---
