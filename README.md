# GenAI Service - Gymnasyum Reschedule Assistant

A stateless Generative AI service built with **FastAPI**, **Google Gemini (3.6 Flash)**, and **ChromaDB**. This service acts as the core AI engine for the Gymnasyum booking system, capable of analyzing customer chat contexts, retrieving gym policies via RAG (Retrieval-Augmented Generation), and outputting structured JSON responses to seamlessly drive the backend state machine.

## Features

- **Context-Aware Intent Detection**: Accurately classifies customer messages into 5 specific intents: `Reschedule`, `Cancel`, `Schedule inquiry`, `Confirmation`, and `Other`.
- **RAG Integration (ChromaDB)**: Dynamically retrieves relevant gym rules, facility info, and pricing to inject into the LLM prompt, minimizing hallucinations.
- **State Machine Orchestration**: Translates AI intents into actionable backend states (`active`, `awaiting_confirmation`, `submitted`).
- **Bilingual Support**: Hardcoded, reliable confirmation flows in both Indonesian (ID) and English (EN).
- **Secure by Design**: Implements Bearer token authentication with `secrets.compare_digest` to prevent timing attacks.
- **High Reliability & Fallbacks**: Graceful error handling for model unavailability (503 Service Unavailable) and gateway issues (502 Bad Gateway), ensuring the system never crashes during failed LLM executions.

## Tech Stack

- **Framework**: FastAPI, Uvicorn
- **AI/LLM**: Google Generative AI SDK (`gemini-3.6-flash`)
- **Vector Database**: ChromaDB (for RAG)
- **Data Validation**: Pydantic

## Prerequisites

- Python 3.9+
- A valid Google Gemini API Key.
- (Optional but recommended) A pre-populated ChromaDB instance located in the server environment.

## Environment Variables

Create a `.env` file in the root directory (or configure your deployment server) with the following variables:

```env
# Required: Your Google AI Studio API Key
GEMINI_API_KEY="AIzaSy_YOUR_GEMINI_API_KEY"

# Required: Bearer token used by your backend to authorize requests to this FastAPI service
GEN_AI_API_KEY="your_secure_bearer_token"

# Optional: The Gemini model version (Defaults to gemini-3.6-flash)
GEMINI_MODEL="gemini-3.6-flash"

```

## Core Processing Logic

1. **Extraction**: Parses chat history, schedules, and backend session states.
2. **Retrieval**: Uses ChromaDB to fetch relevant context based on the user's latest message.
3. **Prompt Assembly**: Combines the strict 5-intent rules, available candidate schedules, remaining quota, and local time logic.
4. **Generation**: Instructs Gemini to output *pure JSON* with a drafted reply, confidence score, and extracted entities.
5. **Final Validation**: Evaluates the LLM's output against the provided candidate schedule IDs. Overrides AI outputs during `Confirmation` intents with hardcoded bilingual replies to ensure system stability.
