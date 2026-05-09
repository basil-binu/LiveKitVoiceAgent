import asyncio
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from livekit.api import AccessToken, LiveKitAPI, VideoGrants
from livekit.api.agent_dispatch_service import CreateAgentDispatchRequest
from livekit.protocol.room import CreateRoomRequest, UpdateRoomMetadataRequest

from rag_creation.ingest import build_vectorstore, UPLOADS_DIR

load_dotenv(".env.local")

app = FastAPI(title="Realtime Voice Agent")

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# RAG CONFIG  (absolute paths — no CWD issues)
# =========================================================

BASE_DIR   = Path(__file__).parent                      # src/
DOCS_PATH  = BASE_DIR / "rag_creation" / "docs"        # src/rag_creation/docs
UPLOADS_PATH = BASE_DIR / "uploads"                     # src/uploads
UPLOADS_PATH.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# =========================================================
# TOKEN ENDPOINT
# =========================================================

@app.get("/token")
async def get_token(
    room: str = "voice-agent",
    system_prompt: str = "",
):
    async with LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    ) as lk:

        try:
            await lk.room.create_room(CreateRoomRequest(name=room))
        except Exception:
            pass

        await lk.room.update_room_metadata(
            UpdateRoomMetadataRequest(
                room=room,
                metadata=json.dumps({"system_prompt": system_prompt}),
            )
        )

        await lk.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                room=room,
                agent_name="voice-agent",
            )
        )

    token = (
        AccessToken(
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )
        .with_grants(VideoGrants(room_join=True, room=room))
        .with_identity("guest-user")
        .to_jwt()
    )

    return {"token": token, "url": os.getenv("LIVEKIT_URL")}

# =========================================================
# LIVE PROMPT UPDATE
# =========================================================

@app.post("/update-prompt")
async def update_prompt(data: dict):
    room = data["room"]

    async with LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    ) as lk:
        await lk.room.update_room_metadata(
            UpdateRoomMetadataRequest(
                room=room,
                metadata=json.dumps({"system_prompt": data["system_prompt"]}),
            )
        )

    return {"success": True}

# =========================================================
# UPLOAD  →  save file + rebuild index
# =========================================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    dest = UPLOADS_PATH / file.filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        await asyncio.to_thread(build_vectorstore)
    except Exception as e:
        import traceback
        traceback.print_exc()
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed: {e}",
        )

    return {
        "success": True,
        "filename": file.filename,
        "message": "File saved and knowledge base updated.",
    }

# =========================================================
# DELETE  →  remove file + rebuild index
# =========================================================

@app.delete("/delete-doc/{filename}")
async def delete_document(filename: str):
    target = UPLOADS_PATH / filename

    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' not found in uploads.",
        )

    target.unlink()

    try:
        await asyncio.to_thread(build_vectorstore)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Index rebuild failed after delete: {e}",
        )

    return {
        "success": True,
        "filename": filename,
        "message": "File deleted and knowledge base updated.",
    }

# =========================================================
# LIST DOCS  →  what's currently in the KB
# =========================================================

@app.get("/list-docs")
async def list_documents():
    static_files = [
        {
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "type": f.suffix.lower(),
            "origin": "static",
        }
        for f in sorted(DOCS_PATH.iterdir())
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    uploaded_files = [
        {
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "type": f.suffix.lower(),
            "origin": "uploaded",
        }
        for f in sorted(UPLOADS_PATH.iterdir())
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    return {
        "total": len(static_files) + len(uploaded_files),
        "static": static_files,
        "uploaded": uploaded_files,
    }