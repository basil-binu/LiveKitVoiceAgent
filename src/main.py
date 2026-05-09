from livekit.api import AccessToken, VideoGrants, LiveKitAPI
from livekit.api.agent_dispatch_service import CreateAgentDispatchRequest
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(".env.local")

app = FastAPI(title="Pete's Inn Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/token")
async def get_token(room: str = "petes-inn"):
    async with LiveKitAPI(
        url=os.getenv("LIVEKIT_URL"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    ) as lk:
      
      await lk.agent_dispatch.create_dispatch(
          CreateAgentDispatchRequest(
              room=room,
              agent_name="petes-inn-agent",
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