# Real-Time Voice Agent

An end-to-end real-time voice agent built with LiveKit (WebRTC), LangGraph, OpenAI, and RAG over uploaded documents.

---

## Architecture

```
Browser (React + LiveKit SDK)
    ↕  WebRTC audio
LiveKit Server
    ↕  agent job dispatch
Agent (agent.py)  ←→  LangGraph (graph_voice.py)  ←→  FAISS vector store
    ↕  STT: Sarvam                                      ↑
    ↕  TTS: Cartesia                             rag_creation/docs/
FastAPI (server.py) ← REST → Browser             uploads/
```

---

## Prerequisites

- Python 3.10+
- A LiveKit server (cloud at [livekit.io](https://livekit.io) or self-hosted)
- OpenAI API key
- Sarvam API key
- Cartesia API key

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd <repo-root>

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create your environment file

Copy the example env file and fill in your keys:

```bash
cp .env.example .env.local
```

Open `.env.local` and set every value:

```env
# LiveKit
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# OpenAI  (LLM + embeddings)
OPENAI_API_KEY=sk-...

# Sarvam  (STT)
SARVAM_API_KEY=your_sarvam_api_key

# Cartesia  (TTS)
CARTESIA_API_KEY=your_cartesia_api_key

# LangSmith  (optional — tracing)
LANGSMITH_API_KEY=your_langsmith_api_key
```

> **Note:** The app loads `.env.local`, not `.env`. Make sure the filename is exact.

### 4. Add your documents (optional)

Drop any `.pdf`, `.txt`, or `.docx` files you want baked into the knowledge base into:

```
src/rag_creation/docs/
```

Then build the initial vector index:

```bash
cd src
python -m rag_creation.ingest
```

---

## Running

All commands below assume you are inside the `src/` directory.

```bash
cd src
```

Open **two terminal tabs** — one for the API server, one for the agent.

### Tab 1 — FastAPI server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Tab 2 — LiveKit agent

```bash
python agent.py start
```

The agent registers itself with LiveKit and waits for a room dispatch. It will connect automatically when a user opens a session from the frontend.

---

## Frontend

Follow the instructions in the `frontend/` directory (React + Vite). Set `VITE_API_BASE_URL` in `frontend/.env` to point at your FastAPI server:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Then:

```bash
cd frontend
npm install
npm run dev
```

---

## How It Works

1. **Connect** — the browser calls `GET /token`, which creates a LiveKit room, sets the system prompt as room metadata, dispatches the agent, and returns a JWT.
2. **Voice** — the agent uses Sarvam for speech-to-text, sends the transcript to LangGraph, and speaks the response back via Cartesia TTS.
3. **RAG** — when the LangGraph `rag_tool` is called, it queries the FAISS index and returns relevant chunks along with their source filenames. Those filenames are sent back to the browser over a LiveKit data channel and shown in the "Sources Used" panel.
4. **Document upload** — the frontend uploads files to `POST /upload`. The server saves them to `src/uploads/` and rebuilds the FAISS index automatically.
5. **Prompt updates** — `POST /update-prompt` patches the LiveKit room metadata. The agent reads fresh metadata on every turn, so prompt changes take effect immediately without reconnecting.

---

## Project Structure

```
├── .env.example
├── requirements.txt
└── src/
    ├── server.py              # FastAPI — token, upload, delete, list-docs, update-prompt
    ├── agent.py               # LiveKit agent — STT → LangGraph → TTS
    ├── uploads/               # User-uploaded documents (auto-created)
    ├── index/                 # FAISS vector index (auto-created on first ingest)
    ├── Databases/
    │   └── petesinn.sqlite    # LangGraph conversation memory
    ├── graph/
    │   ├── graph_voice.py     # LangGraph state machine
    │   ├── tools_voice.py     # rag_tool definition
    │   └── memory.py          # SQLite checkpointer
    └── rag_creation/
        ├── ingest.py          # Chunking, embedding, FAISS index builder
        └── docs/              # Static knowledge base documents
```

---

## Known Limitations

- The FAISS index is rebuilt from scratch on every upload or delete. For large document sets this adds a few seconds of latency to those operations.
- Conversation memory is per-room. Reconnecting with the same room name resumes the previous conversation; a new room name starts fresh.
- The agent reads the system prompt from room metadata on each turn — there is a small window where a prompt update mid-turn may not apply until the next user message.
- STT language is hardcoded to `en-IN`. Change `language` in `agent.py` to target a different locale.